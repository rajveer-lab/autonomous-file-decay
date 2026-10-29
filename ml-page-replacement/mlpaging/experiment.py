"""End-to-end pipeline: verify, train, tune on validation, evaluate on test."""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

from . import policies as P
from . import traces as T
from .learned import FEATURES, GROUPS, LearnedPolicy, LogReg, Tree, model_bytes, opt_dataset
from .sim import simulate

FRACTIONS = [0.01, 0.02, 0.05, 0.10]
_TRACES: dict[str, T.Trace] = {}


def capacity(trace: T.Trace, frac: float) -> int:
    return max(8, round(frac * trace.distinct))


def verify_textbook() -> dict:
    tb = T.textbook().pages
    got = {p.name: simulate(p, tb, 3)["faults"] for p in (P.FIFO(), P.LRU(), P.OPT())}
    expected = {"FIFO": 15, "LRU": 12, "OPT": 9}
    assert got == expected, f"textbook check failed: {got} != {expected}"
    return got


# ----------------------------------------------------------------- workers

def _run(job):
    name, frac, policy = job
    tr = _TRACES[name]
    c = capacity(tr, frac)
    r = simulate(policy, tr.pages, c)
    r.update(trace=name, family=tr.family, fraction=frac)
    if isinstance(policy, LearnedPolicy):
        r["fallbacks"] = policy.fallbacks
    return r


def run_jobs(jobs, procs):
    if procs <= 1:
        return [_run(j) for j in jobs]
    with mp.get_context("fork").Pool(procs) as pool:
        return pool.map(_run, jobs, chunksize=1)


def _dataset(job):
    name, frac = job
    tr = _TRACES[name]
    return opt_dataset(tr.pages, capacity(tr, frac), seed=sum(map(ord, name)) % 1000)


def build_dataset(train, procs, fractions):
    """OPT-labelled rows: features X and distance-to-next-use D (in units of C)."""
    jobs = [(t.name, f) for t in train for f in fractions]
    if procs <= 1:
        parts = [_dataset(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(procs) as pool:
            parts = pool.map(_dataset, jobs, chunksize=1)
    X = np.vstack([p[0] for p in parts])
    D = np.concatenate([p[1] for p in parts])
    return X, D


def target(D, spec):
    """Classifier labels for horizon h, or log-distance for regression."""
    return np.log1p(D) if spec == "reg" else (D > spec).astype(float)


def thresholds(spec):
    """Confidence thresholds to tune: probabilities, or distances in units of C."""
    return [0.5, 0.7, 0.9] if spec != "reg" else [math.log1p(x) for x in (0.5, 1.0, 2.0)]


def relative_to_lru(rows):
    """Mean over (trace, size) of faults / LRU faults, per policy."""
    lru = {(r["trace"], r["fraction"]): r["faults"] for r in rows if r["policy"] == "LRU"}
    out = {}
    for r in rows:
        base = lru.get((r["trace"], r["fraction"]))
        if base:
            out.setdefault(r["policy"], []).append(r["faults"] / base)
    return {k: float(np.mean(v)) for k, v in out.items()}


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------- pipeline

def main(out="results", quick=False, procs=None, seed=0):
    procs = procs or max(1, (os.cpu_count() or 2))
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    say(f"textbook check (3 frames): {verify_textbook()}")

    train, val, test = (T.workload_set(s) for s in ("train", "val", "test"))
    if quick:
        train, val, test = train[:4], val[:4], test[:5]
    fractions = FRACTIONS[1:3] if quick else FRACTIONS
    for tr in train + val + test:
        _TRACES[tr.name] = tr
    say("traces: " + ", ".join(f"{t.name} ({len(t)} refs, {t.distinct} pages)" for t in train + val + test))

    # 1. training data: one OPT replay, labels made per horizon
    t0 = time.time()
    X, D = build_dataset(train, procs, fractions)
    say(f"training data: {len(D)} OPT-labelled rows from {len(train)} traces in {time.time() - t0:.1f}s")
    specs = [("logreg", 1.0), ("tree", "reg")] if quick else \
        [("logreg", h) for h in (0.5, 1.0, 2.0)] + [("tree", h) for h in (0.5, 1.0, 2.0)] + [("tree", "reg")]
    models = {}
    for kind, spec in specs:
        t0 = time.time()
        m = (LogReg if kind == "logreg" else Tree)().fit(X, target(D, spec)).compile()
        m.train_s = time.time() - t0
        models[(kind, spec)] = m

    def label(kind, spec):
        return f"{kind}-{'regression' if spec == 'reg' else f'H{spec}'}"

    def learned(model, spec, k=16, thr=None, name="Learned", guard=True):
        thr = thresholds(spec)[0] if thr is None else thr
        return LearnedPolicy(model, k=k, threshold=thr, seed=seed, name=name, guard=guard)

    # 2. model selection on validation traces, by simulated faults
    val_jobs = [(t.name, f, P.LRU()) for t in val for f in fractions]
    for (kind, spec), m in models.items():
        val_jobs += [(t.name, f, learned(m, spec, name=label(kind, spec))) for t in val for f in fractions]
    val_rows = run_jobs(val_jobs, procs)
    score = relative_to_lru(val_rows)
    best_kind, best_spec = min(models, key=lambda ks: score[label(*ks)])
    say("validation, faults / LRU (lower is better): " + ", ".join(
        f"{k}={v:.3f}" for k, v in sorted(score.items()) if k != "LRU"))
    best = models[(best_kind, best_spec)]

    # 3. tune sample size k and the confidence threshold on validation
    ks = [16] if quick else [8, 16, 32]
    thrs = thresholds(best_spec)[:1] if quick else thresholds(best_spec)
    tune_jobs = [(t.name, f, P.LRU()) for t in val for f in fractions]
    for k in ks:
        for i, thr in enumerate(thrs):
            tune_jobs += [(t.name, f, learned(best, best_spec, k, thr, name=f"k{k}-thr{i}"))
                          for t in val for f in fractions]
    tune_rows = run_jobs(tune_jobs, procs)
    tscore = relative_to_lru(tune_rows)
    cfg = min((s for s in tscore if s != "LRU"), key=tscore.get)
    best_k, best_thr = int(cfg.split("-")[0][1:]), thrs[int(cfg.split("thr")[1])]
    say(f"chosen: {label(best_kind, best_spec)}, k={best_k}, threshold={best_thr:.3f} "
        f"(validation faults / LRU = {tscore[cfg]:.3f})")
    write_csv(os.path.join(out, "validation.csv"), val_rows + tune_rows)

    # 4. test: every baseline and the chosen learned policy on unseen traces;
    #    "Learned (no guard)" shows what the regret guard adds
    def policy_set():
        return P.baselines(seed) + [
            learned(best, best_spec, best_k, best_thr, name="Learned"),
            learned(best, best_spec, best_k, best_thr, name="Learned (no guard)", guard=False)]
    test_jobs = [(t.name, f, pol) for t in test for f in fractions for pol in policy_set()]
    test_rows = run_jobs(test_jobs, procs)
    write_csv(os.path.join(out, "test.csv"), test_rows)

    # 5. feature ablation: retrain without each feature group
    abl_rows = []
    if not quick:
        M = Tree if best_kind == "tree" else LogReg
        abl_jobs = []
        for group, feats in GROUPS.items():
            cols = [i for i, f in enumerate(FEATURES) if f not in feats]
            m = M().fit(X, target(D, best_spec), cols=cols).compile()
            abl_jobs += [(t.name, f, learned(m, best_spec, best_k, best_thr, name=f"without {group}"))
                         for t in test for f in fractions]
        abl_rows = run_jobs(abl_jobs, procs)
        write_csv(os.path.join(out, "ablation.csv"), abl_rows)

    summary = {
        "textbook": verify_textbook(),
        "model": {"kind": best_kind, "target": label(best_kind, best_spec),
                  "k": best_k, "threshold": best_thr,
                  "params": best.n_params, "bytes": model_bytes(best),
                  "train_seconds": round(best.train_s, 3), "train_rows": int(len(D))},
        "metadata_bytes_per_page": {p.name: p.metadata_bytes for p in policy_set()},
        "fractions": fractions,
        "seconds": round(time.time() - t_start, 1),
        "log": log,
    }
    if best_kind == "logreg":
        summary["model"]["weights"] = dict(zip(FEATURES, map(float, best.w))) | {"bias": float(best.b)}
    else:
        summary["model"]["nodes"] = best.nodes
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    say(f"done in {time.time() - t_start:.0f}s")
    return test_rows, abl_rows, summary
