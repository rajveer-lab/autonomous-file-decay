"""PageLab: a local web application for page replacement on real workloads.

    python3 -m mlpaging serve            # opens http://127.0.0.1:8765

Workloads come from the user: traces recorded from real programs on this
machine (Valgrind or strace) or imported trace files. Models are trained on
those workloads. Everything is stored under the data directory
(default ~/.mlpaging).

The server only listens on 127.0.0.1, and every API call must carry the
session token embedded in the page, so other websites cannot drive it.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import re
import secrets
import shlex
import tempfile
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import policies as P
from . import realtrace as R
from .learned import FEATURES, LearnedPolicy, LogReg, Tree, model_from_dict, opt_dataset
from .sim import simulate

POLICY_NAMES = ["FIFO", "LRU", "LFU", "CLOCK", "ARC", "LeCaR", "SIEVE", "S3-FIFO", "Learned", "OPT"]


# ------------------------------------------------------------------- store

class Store:
    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))
        for d in ("traces", "models", "runs"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        self.lock = threading.Lock()

    def _path(self, kind, id_, ext=".json"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", id_):
            raise KeyError(id_)
        return os.path.join(self.root, kind, id_ + ext)

    def list(self, kind):
        out = []
        for name in sorted(os.listdir(os.path.join(self.root, kind))):
            if name.endswith(".json"):
                with open(os.path.join(self.root, kind, name)) as f:
                    out.append(json.load(f))
        out.sort(key=lambda d: d.get("created", 0), reverse=True)
        return out

    def get(self, kind, id_):
        with open(self._path(kind, id_)) as f:
            return json.load(f)

    def put(self, kind, doc):
        with self.lock, open(self._path(kind, doc["id"]), "w") as f:
            json.dump(doc, f, indent=1)

    def delete(self, kind, id_):
        for ext in (".json", ".npy"):
            try:
                os.remove(self._path(kind, id_, ext))
            except OSError:
                pass

    def trace_path(self, id_):
        return self._path("traces", id_, ".npy")

    def save_trace(self, name, built, source):
        pages = built.pages
        id_ = "t" + uuid.uuid4().hex[:10]
        np.save(self.trace_path(id_), pages.astype(np.int32 if pages.max() < 2**31 else np.int64))
        doc = {"id": id_, "name": name, "created": time.time(), "refs": int(pages.size),
               "distinct": int(np.unique(pages).size), "raw_refs": int(built.raw_refs),
               "capped": bool(built.capped), "source": source}
        self.put("traces", doc)
        return doc

    def models(self):
        pre = json.loads(resources.files("mlpaging").joinpath("pretrained.json").read_text())
        pre["created"] = 0
        return self.list("models") + [pre]

    def model(self, id_):
        if id_ == "pretrained":
            return json.loads(resources.files("mlpaging").joinpath("pretrained.json").read_text())
        return self.get("models", id_)


# -------------------------------------------------------------------- jobs

class Jobs:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, kind, title, fn):
        jid = "j" + uuid.uuid4().hex[:10]
        job = {"id": jid, "kind": kind, "title": title, "status": "running", "progress": 0.0,
               "message": "starting", "started": time.time(), "result": None, "error": None}
        with self.lock:
            self.jobs[jid] = job

        def update(progress=None, message=None):
            if progress is not None:
                job["progress"] = max(0.0, min(1.0, progress))
            if message is not None:
                job["message"] = message

        def run():
            try:
                job["result"] = fn(update)
                job["status"], job["progress"], job["message"] = "done", 1.0, "finished"
            except Exception as e:     # report the failure to the page
                job["status"], job["error"] = "error", str(e) or e.__class__.__name__
                job["trace"] = traceback.format_exc(limit=3)
            job["ended"] = time.time()

        threading.Thread(target=run, daemon=True).start()
        return job

    def list(self):
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: j["started"], reverse=True)[:30]


# --------------------------------------------------------------- simulation

def _make_policy(name, model_doc, k, threshold, seed=0):
    if name == "Learned":
        return LearnedPolicy(model_from_dict(model_doc), k=k, threshold=threshold, seed=seed)
    return {
        "FIFO": P.FIFO, "LRU": P.LRU, "LFU": P.LFU, "CLOCK": P.CLOCK, "ARC": P.ARC,
        "LeCaR": lambda: P.LeCaR(seed=seed), "SIEVE": P.SIEVE, "S3-FIFO": P.S3FIFO, "OPT": P.OPT,
    }[name]()


def _sim_task(args):
    path, lo, hi, name, frames, model_doc, k, threshold = args
    pages = np.load(path, mmap_mode="r")[lo:hi].astype(np.int64)
    pages = _relabel(pages)
    pol = _make_policy(name, model_doc, k, threshold)
    t0 = time.perf_counter()
    r = simulate(pol, pages, frames)
    r["seconds"] = time.perf_counter() - t0
    r["policy"] = name
    if name == "Learned":
        r["model_evictions"] = pol.model_evictions
        r["fallbacks"] = pol.fallbacks
    return r


def _relabel(pages):
    _, inv = np.unique(pages, return_inverse=True)
    return inv.astype(np.int64)


def _executor():
    workers = max(1, (os.cpu_count() or 2) - 1)
    try:
        return cf.ProcessPoolExecutor(max_workers=workers)
    except (OSError, NotImplementedError):     # e.g. sandboxes without process pools
        return cf.ThreadPoolExecutor(max_workers=1)


def run_simulation(store, params, update):
    tr = store.get("traces", params["trace"])
    pages = np.load(store.trace_path(tr["id"]), mmap_mode="r")
    n = pages.size
    lo = int(n * float(params.get("start", 0.0)))
    hi = n
    seg = _relabel(np.asarray(pages[lo:hi]))
    distinct = int(seg.max()) + 1
    frames = sorted({max(1, min(distinct, int(f))) for f in params["frames"]})
    policies = [p for p in POLICY_NAMES if p in params["policies"]]
    if "LRU" not in policies:
        policies.insert(0, "LRU")
    model_doc = store.model(params.get("model", "pretrained"))
    k = int(params.get("k") or model_doc.get("k", 32))
    thr = float(params.get("threshold") if params.get("threshold") is not None else model_doc.get("threshold", 0.5))
    tasks = [(store.trace_path(tr["id"]), lo, hi, p, f, model_doc, k, thr) for f in frames for p in policies]
    update(0.0, f"{len(tasks)} simulations of {hi - lo:,} references")
    rows = []
    with _executor() as ex:
        futs = [ex.submit(_sim_task, t) for t in tasks]
        for i, fut in enumerate(cf.as_completed(futs)):
            rows.append(fut.result())
            update((i + 1) / len(tasks), f"{i + 1} of {len(tasks)} simulations done")
    by = {(r["policy"], r["capacity"]): r for r in rows}
    for r in rows:
        lru = by[("LRU", r["capacity"])]["faults"]
        r["vs_lru_pct"] = 100 * (r["faults"] - lru) / lru if lru else 0.0
        if ("OPT", r["capacity"]) in by:
            opt = by[("OPT", r["capacity"])]["faults"]
            r["gap_closed_pct"] = 100 * (lru - r["faults"]) / (lru - opt) if lru > opt else None
    rows.sort(key=lambda r: (r["capacity"], POLICY_NAMES.index(r["policy"])))
    run = {"id": "r" + uuid.uuid4().hex[:10], "created": time.time(), "trace": tr["id"],
           "trace_name": tr["name"], "start": lo, "end": hi, "distinct": distinct,
           "frames": frames, "policies": policies, "model": model_doc.get("id"),
           "model_name": model_doc.get("name"), "k": k, "threshold": thr, "rows": rows}
    store.put("runs", run)
    return {"run": run["id"]}


def run_training(store, params, update):
    ids = params["traces"]
    frac = float(params.get("train_fraction", 0.5))
    horizon = float(params.get("horizon", 2.0))
    kind = params.get("kind", "logreg")
    fractions = [0.01, 0.02, 0.05, 0.10]
    Xs, Ds, names = [], [], []
    for i, tid in enumerate(ids):
        tr = store.get("traces", tid)
        pages = np.load(store.trace_path(tid), mmap_mode="r")
        seg = _relabel(np.asarray(pages[: max(2, int(pages.size * frac))]))
        distinct = int(seg.max()) + 1
        names.append(f"{tr['name']} (first {round(100 * frac)}%)")
        for j, f in enumerate(fractions):
            c = max(4, round(f * distinct))
            if c >= distinct:
                continue
            update((i * len(fractions) + j) / (len(ids) * len(fractions)) * 0.9,
                   f"replaying {tr['name']} under OPT with {c} frames")
            X, D = opt_dataset(seg, c, max_rows=25_000, seed=i * 10 + j)
            Xs.append(X)
            Ds.append(D)
    if not Xs:
        raise ValueError("the selected traces are too small to train on")
    X, D = np.vstack(Xs), np.concatenate(Ds)
    update(0.92, f"fitting {kind} on {len(D):,} OPT-labelled rows")
    t0 = time.time()
    if kind == "tree":
        m = Tree().fit(X, (D > horizon).astype(float))
    else:
        m = LogReg().fit(X, (D > horizon).astype(float)).compile()
    doc = {"id": "m" + uuid.uuid4().hex[:10], "created": time.time(),
           "name": params.get("name") or f"{kind} on {', '.join(n.split(' (')[0] for n in names)}",
           **m.to_dict(), "features": FEATURES, "horizon": horizon,
           "k": 32, "threshold": 0.5, "trained_on": names, "train_fraction": frac,
           "rows": int(len(D)), "label_rate": float((D > horizon).mean()),
           "train_seconds": round(time.time() - t0, 3), "traces": ids}
    store.put("models", doc)
    return {"model": doc["id"]}


def run_import(store, path, name, fmt, cap, instructions, unit, update, cleanup=True):
    try:
        update(0.05, "reading file")
        built, used = R.load(path, fmt=fmt, cap=cap, instructions=instructions, unit=unit,
                             progress=lambda n: update(min(0.9, n / cap), f"{n:,} page references read"))
        doc = store.save_trace(name, built, {"kind": "file", "format": used,
                                             "file": os.path.basename(name)})
        return {"trace": doc["id"]}
    finally:
        if cleanup:
            try:
                os.unlink(path)
            except OSError:
                pass


def run_record(store, params, update):
    argv = shlex.split(params["command"])
    tool = params.get("tool", "valgrind")
    cap = int(params.get("cap", 2_000_000))
    update(0.02, f"running {argv[0]} under {tool}")
    built, info = R.record(argv, tool=tool, cap=cap, instructions=bool(params.get("instructions")),
                           cwd=params.get("cwd") or None,
                           progress=lambda n: update(min(0.95, n / cap), f"{n:,} page references captured"))
    name = params.get("name") or f"{os.path.basename(argv[0])} ({tool})"
    doc = store.save_trace(name, built, {"kind": "recorded", **info})
    return {"trace": doc["id"]}


# ------------------------------------------------------------------ server

class App:
    def __init__(self, data_dir):
        self.store = Store(data_dir)
        self.jobs = Jobs()
        self.token = secrets.token_urlsafe(24)


def make_handler(app: App, port: int):
    allowed_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
    page = resources.files("mlpaging").joinpath("static/app.html").read_text()

    class Handler(BaseHTTPRequestHandler):
        server_version = "PageLab"

        def log_message(self, *args):
            pass

        # --- helpers
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=_jsonable).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self):
            origin = self.headers.get("Origin")
            if origin is not None and origin not in allowed_origins:
                return False
            return secrets.compare_digest(self.headers.get("X-Token", ""), app.token)

        def _json(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        # --- routes
        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                html = page.replace("__TOKEN__", app.token)
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if not self._authorised():
                return self._send(403, {"error": "forbidden"})
            s = app.store
            if url.path == "/api/state":
                return self._send(200, {"traces": s.list("traces"), "models": s.models(),
                                        "runs": [_run_summary(r) for r in s.list("runs")],
                                        "jobs": app.jobs.list(), "tools": R.tools(),
                                        "policies": POLICY_NAMES, "data_dir": s.root,
                                        "cpus": os.cpu_count()})
            m = re.fullmatch(r"/api/runs/([\w-]+)", url.path)
            if m:
                return self._send(200, s.get("runs", m.group(1)))
            m = re.fullmatch(r"/api/runs/([\w-]+)\.csv", url.path)
            if m:
                run = s.get("runs", m.group(1))
                keys = ["policy", "capacity", "accesses", "faults", "fault_rate", "vs_lru_pct",
                        "gap_closed_pct", "decision_us", "seconds"]
                lines = [",".join(keys)] + [",".join("" if r.get(k) is None else str(r.get(k)) for k in keys)
                                            for r in run["rows"]]
                return self._send(200, ("\n".join(lines) + "\n").encode(), "text/csv")
            m = re.fullmatch(r"/api/traces/([\w-]+)/preview", url.path)
            if m:
                pages = np.load(s.trace_path(m.group(1)), mmap_mode="r")
                step = max(1, pages.size // 1500)
                return self._send(200, {"sample": pages[::step][:1500].tolist(), "step": step})
            return self._send(404, {"error": "not found"})

        def do_DELETE(self):
            if not self._authorised():
                return self._send(403, {"error": "forbidden"})
            m = re.fullmatch(r"/api/(traces|models|runs)/([\w-]+)", urlparse(self.path).path)
            if not m:
                return self._send(404, {"error": "not found"})
            app.store.delete(m.group(1), m.group(2))
            return self._send(200, {"ok": True})

        def do_POST(self):
            if not self._authorised():
                return self._send(403, {"error": "forbidden"})
            url = urlparse(self.path)
            s, jobs = app.store, app.jobs
            try:
                if url.path == "/api/traces/upload":
                    q = {k: v[0] for k, v in parse_qs(url.query).items()}
                    name = q.get("name") or "uploaded trace"
                    suffix = ".gz" if name.endswith(".gz") else ".txt"
                    fd, tmp = tempfile.mkstemp(suffix=suffix)
                    remaining = int(self.headers.get("Content-Length") or 0)
                    with os.fdopen(fd, "wb") as f:
                        while remaining > 0:
                            chunk = self.rfile.read(min(1 << 20, remaining))
                            if not chunk:
                                break
                            f.write(chunk)
                            remaining -= len(chunk)
                    job = jobs.start("import", f"Import {name}", lambda u: run_import(
                        s, tmp, name, q.get("format", "auto"), int(q.get("cap", 2_000_000)),
                        q.get("instructions") == "1", q.get("unit", "address"), u))
                    return self._send(202, job)
                body = self._json()
                if url.path == "/api/traces/record":
                    if not body.get("command", "").strip():
                        return self._send(400, {"error": "enter a command to run"})
                    job = jobs.start("record", f"Record {body['command'][:60]}",
                                     lambda u: run_record(s, body, u))
                    return self._send(202, job)
                if url.path == "/api/train":
                    if not body.get("traces"):
                        return self._send(400, {"error": "pick at least one trace"})
                    job = jobs.start("train", "Train model", lambda u: run_training(s, body, u))
                    return self._send(202, job)
                if url.path == "/api/simulate":
                    if not body.get("trace") or not body.get("frames"):
                        return self._send(400, {"error": "pick a trace and at least one memory size"})
                    tr = s.get("traces", body["trace"])
                    job = jobs.start("simulate", f"Simulate {tr['name']}",
                                     lambda u: run_simulation(s, body, u))
                    return self._send(202, job)
            except KeyError as e:
                return self._send(404, {"error": f"not found: {e}"})
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            return self._send(404, {"error": "not found"})

    return Handler


def _run_summary(r):
    return {k: r[k] for k in ("id", "created", "trace", "trace_name", "frames", "policies",
                              "model_name", "start", "end") if k in r}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


def serve(port=8765, data_dir="~/.mlpaging", open_browser=True):
    app = App(data_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, port))
    url = f"http://127.0.0.1:{port}/"
    print(f"PageLab running at {url}  (data: {app.store.root})  Ctrl+C to stop", flush=True)
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return app
