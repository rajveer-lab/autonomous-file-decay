"""Reproduce every number, table and figure in the case-study report.

    python experiments/run_all.py            # writes results/*.csv, paper/figures, paper/tables

All metrics are measured, none are estimated: circuits are produced by the
QReclaim pipeline, lowered to {CX, U} by Qiskit (optimization level 1), and
verified by exact simulation where the width allows it.
"""

from __future__ import annotations

import csv
import pathlib
import platform
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib                                              # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import qiskit                                                  # noqa: E402
from qiskit_aer import AerSimulator                            # noqa: E402

import json                                                    # noqa: E402
import statistics                                              # noqa: E402

from qreclaim import (ABLATION, NR, QR, SB, SCOPE, CompileError, Program,   # noqa: E402
                      compile_neutral, compile_program, measure_circuit, with_budget)
from qreclaim.benchmarks import (SUITES, VERIFY, adder, bv, grover_sat,  # noqa: E402
                                 hidden_shift, qaoa_ring, repetition_code)
from qreclaim.sim import simulate, tvd                         # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
for d in (RES, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

# Categorical slots 1-4 of the reference palette, in fixed order, plus marker
# and dash as secondary encodings so the figures survive grayscale printing.
STYLE = {
    "NR":       dict(color="#2a78d6", marker="o", ls="-",  label="NR (no reuse)"),
    "SCOPE":    dict(color="#eb6834", marker="s", ls="--", label="SCOPE (lexical)"),
    "SB":       dict(color="#52514e", marker="x", ls="-.", label="SB (strong baseline)"),
    "QR-min":   dict(color="#1baf7a", marker="^", ls="-",  label="QReclaim-min"),
    "QR-dn":    dict(color="#eda100", marker="D", ls=":",  label="QReclaim-DN"),
}
# Benchmark identity (budget and scaling figures) uses categorical slots 5-8
# so it never collides with the configuration colours above; ADD falls back
# to neutral ink.
BENCH = {
    "BV": dict(color="#6250d6", marker="o", ls="-"),
    "HS": dict(color="#e87ba4", marker="s", ls="--"),
    "RC": dict(color="#008300", marker="^", ls="-."),
    "GRV": dict(color="#e34948", marker="D", ls=":"),
    "ADD": dict(color="#52514e", marker="v", ls="-"),
}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 3.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def size_of(params: str) -> int:
    return int(params.split(",")[0].split("=")[1])


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
def run_main() -> list[dict]:
    rows = []
    for suite, benches in SUITES.items():
        for b in benches:
            nr = compile_program(b.build(), NR)
            mnr = measure_circuit(nr)
            sc = compile_program(b.build(), SCOPE)
            msc = measure_circuit(sc)
            sb = compile_program(b.build(), SB)
            msb = measure_circuit(sb)
            qr = compile_program(b.build(), QR)
            mqr = measure_circuit(qr)
            t0 = time.perf_counter()
            dn = compile_neutral(b.build, mnr["depth"], qr.nqubits, nr.nqubits, "depth")
            search_ms = 1e3 * (time.perf_counter() - t0)
            exp = None
            for name, c, m in (("NR", nr, mnr), ("SCOPE", sc, msc), ("SB", sb, msb),
                               ("QR-min", qr, mqr), ("QR-dn", dn, dn.metrics)):
                verified = ""
                if c.nqubits <= 20:
                    try:
                        d = simulate(c.instrs, c.phys, c.nqubits, c.program.ncbits)
                        exp = exp if exp is not None else b.expected()
                        verified = f"{tvd(d, exp):.1e}"
                    except RuntimeError:          # too many measurement branches
                        verified = ""
                rows.append(dict(bench=b.name, params=b.params, size=size_of(b.params), config=name,
                                 ir_instrs=len(b.build().instrs),
                                 search_ms=round(search_ms, 2) if name == "QR-dn" else "",
                                 frees_validated=getattr(c, "frees_validated", ""),
                                 verified_tvd=verified,
                                 **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}))
            print(f"main  {b.name:4s} {b.params:10s} NR={mnr['qubits']} SCOPE={msc['qubits']} "
                  f"SB={msb['qubits']} QRmin={mqr['qubits']} QRdn={dn.metrics['qubits']}", flush=True)
    write_csv(RES / "main.csv", rows)
    return rows


def run_ablation() -> list[dict]:
    rows = []
    for b in (bv(32), hidden_shift(16), repetition_code(7, 7), adder(16), grover_sat(8, 16),
              qaoa_ring(16)):
        for cfg in [SCOPE, SB] + ABLATION:
            m = measure_circuit(compile_program(b.build(), cfg))
            rows.append(dict(bench=b.name, params=b.params, config=cfg.name, qubits=m["qubits"],
                             depth=m["depth"], cx=m["cx"], reset=m["reset"]))
    write_csv(RES / "ablation.csv", rows)
    print("ablation done", flush=True)
    return rows


def run_budget() -> list[dict]:
    rows = []
    for b in (bv(32), hidden_shift(16), repetition_code(7, 7), grover_sat(8, 16)):
        lo = compile_program(b.build(), QR).nqubits
        hi = compile_program(b.build(), NR).nqubits
        ref = measure_circuit(compile_program(b.build(), NR))["depth"]
        for B in range(lo, hi + 1):
            m = measure_circuit(compile_program(b.build(), with_budget(QR, B)))
            rows.append(dict(bench=b.name, params=b.params, budget=B, qubits=m["qubits"],
                             depth=m["depth"], reset=m["reset"], nr_depth=ref, nr_qubits=hi))
    write_csv(RES / "budget.csv", rows)
    print("budget done", flush=True)
    return rows


def run_verify() -> list[dict]:
    rows = []
    aer = AerSimulator(method="statevector", seed_simulator=1234)
    shots = 8192
    for b in VERIFY:
        exp = b.expected()
        for name, cfg in (("NR", NR), ("SCOPE", SCOPE), ("QR-min", QR)):
            c = compile_program(b.build(), cfg)
            measure_circuit(c)
            checks = sum(1 for i in c.instrs if i.op in ("alloc", "free"))
            row = dict(bench=b.name, params=b.params, config=name, qubits=c.nqubits,
                       gc_checks=checks, tvd_exact="", tvd_aer="", status="")
            if c.nqubits <= 20:
                dist = simulate(c.instrs, c.phys, c.nqubits, c.program.ncbits)
                row["tvd_exact"] = f"{tvd(dist, exp):.1e}"
                counts = aer.run(qiskit.transpile(c.qc, aer), shots=shots).result().get_counts()
                emp = {}
                for k, v in counts.items():
                    key = k.replace(" ", "")[::-1]            # Qiskit prints c[0] last
                    emp[key] = emp.get(key, 0) + v / shots
                row["tvd_aer"] = f"{tvd(emp, exp):.3f}"
                row["status"] = "pass" if tvd(dist, exp) < 1e-9 else "FAIL"
            else:
                row["status"] = "skipped (width)"
            rows.append(row)
        print(f"verify {b.name} {b.params}", flush=True)
    write_csv(RES / "verify.csv", rows)
    return rows


def run_scaling() -> list[dict]:
    rows = []
    cases = ([("BV", bv(n)) for n in (128, 256, 512, 1024, 2048, 4096)]
             + [("HS", hidden_shift(n)) for n in (32, 64, 128, 256, 512, 1024)]
             + [("ADD", adder(n)) for n in (64, 128, 256, 512, 1024)]
             + [("GRV", grover_sat(n, 2 * n)) for n in (8, 10, 12, 14)])
    for name, b in cases:
        best = None
        for _ in range(5):
            c = compile_program(b.build(), QR)
            if best is None or c.compile_time < best.compile_time:
                best = c
        m = measure_circuit(best)
        rows.append(dict(bench=name, params=b.params, instrs=len(best.instrs),
                         compile_ms=round(1e3 * best.compile_time, 2),
                         qiskit_ms=round(m["qiskit_ms"], 2),
                         **{f"pass_{k}": round(1e3 * v, 2) for k, v in sorted(best.pass_times.items())}))
        print(f"scale {name} {b.params} instrs={len(best.instrs)} "
              f"ours={1e3 * best.compile_time:.1f}ms qiskit={m['qiskit_ms']:.1f}ms", flush=True)
    write_csv(RES / "scaling.csv", rows)
    return rows


def run_diagnostics() -> list[tuple[str, str]]:
    out = []

    p = Program("clone")
    x = p.qreg("x", 1)
    p.cx(x[0], x[0])
    out.append(("E1 aliasing", p))

    p = Program("deadline")
    x = p.qreg("x", 2)
    o = p.qubit("out")
    t = p.ancilla("t")
    p.ccx(x[0], x[1], t)
    p.h(x[0])
    p.cx(t, o)
    out.append(("E2 control modified while ancilla live", p))

    p = Program("leak")
    o = p.qubit("out")
    t = p.ancilla("t")
    p.h(t)
    p.cx(t, o)
    out.append(("E3 unreclaimable (entangled) ancilla", p))

    msgs = []
    for title, prog in out:
        try:
            compile_program(prog, QR)
            msgs.append((title, "accepted"))
        except CompileError as e:
            msgs.append((title, str(e)))
    with (RES / "diagnostics.txt").open("w") as f:
        for title, m in msgs:
            f.write(f"{title}:\n  {m}\n")
    return msgs


# --------------------------------------------------------------------------
def fig_qubits(rows: list[dict]) -> None:
    order = ["BV", "HS", "RC", "ADD", "GRV", "QAOA"]
    xlab = {"BV": "secret bits $n$", "HS": "half-width $n$", "RC": "distance $d$ ($r=d$)",
            "ADD": "operand bits $n$", "GRV": "variables $n$ ($m=2n$)", "QAOA": "ring size $n$"}
    fig, axes = plt.subplots(1, 6, figsize=(7.16, 1.6))
    for ax, bname in zip(axes, order):
        for cfg in STYLE:
            pts = sorted((r["size"], r["qubits"]) for r in rows if r["bench"] == bname and r["config"] == cfg)
            xs, ys = zip(*pts)
            st = STYLE[cfg]
            ax.plot(xs, ys, color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"],
                    lw=1.1, ms=3)
        ax.set_title(bname)
        ax.set_xlabel(xlab[bname], fontsize=6.5)
        ax.set_yscale("log", base=2)
        if bname in ("BV", "HS", "ADD", "QAOA"):
            ax.set_xscale("log", base=2)
        else:
            ax.set_xticks(sorted({r["size"] for r in rows if r["bench"] == bname}))
        ax.grid(True, which="major")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("physical qubits")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(w_pad=0.4)
    fig.savefig(FIG / "fig_qubits.pdf")
    plt.close(fig)


def fig_budget(rows: list[dict]) -> None:
    benches = []
    for r in rows:
        key = (r["bench"], r["params"])
        if key not in benches:
            benches.append(key)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.7))
    for bname, params in benches:
        st = BENCH[bname]
        pts = [r for r in rows if r["bench"] == bname and r["params"] == params]
        hi = pts[0]["nr_qubits"]
        ref = pts[0]["nr_depth"]
        xs = [r["budget"] / hi for r in pts]
        axes[0].plot(xs, [r["depth"] / ref for r in pts], color=st["color"], ls=st["ls"],
                     marker=st["marker"], markevery=4, label=f"{bname} ({params})")
        axes[1].plot(xs, [r["reset"] for r in pts], color=st["color"], ls=st["ls"],
                     marker=st["marker"], markevery=4)
    axes[0].set_yscale("log", base=2)
    axes[0].set_yticks([1, 2, 4, 8])
    axes[0].set_yticklabels(["1", "2", "4", "8"])
    axes[0].minorticks_off()
    axes[0].axhline(1.0, color="#52514e", lw=0.6, ls="-")
    axes[0].set_ylabel("depth / depth(NR)")
    axes[1].set_ylabel("mid-circuit resets")
    for ax in axes:
        ax.set_xlabel("budget $B$ / width(NR)")
        ax.grid(True)
        ax.spines[["top", "right"]].set_visible(False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "fig_budget.pdf")
    plt.close(fig)


def fig_scaling(rows: list[dict]) -> None:
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for b in ("BV", "HS", "ADD", "GRV"):
        st = BENCH[b]
        pts = sorted((r["instrs"], r["compile_ms"], r["qiskit_ms"]) for r in rows if r["bench"] == b)
        xs, ours, qk = zip(*pts)
        ax.plot(xs, ours, color=st["color"], marker=st["marker"], ls="-")
        ax.plot(xs, qk, color=st["color"], marker=st["marker"], ls=":", mfc="white")
    xs = np.array([3e2, 3e4])
    ax.plot(xs, xs * 0.01, color="#8a8984", lw=0.6, ls="--")
    ax.text(1.1e4, 1.1e4 * 0.01 * 0.45, "slope 1\n(linear)", color="#52514e", fontsize=6,
            va="top", ha="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("output instructions")
    ax.set_ylabel("time (ms)")
    ax.grid(True)
    ax.spines[["top", "right"]].set_visible(False)
    bench_h = [Line2D([], [], color=BENCH[b]["color"], marker=BENCH[b]["marker"], ls="-", label=b)
               for b in ("BV", "HS", "ADD", "GRV")]
    style_h = [Line2D([], [], color="#52514e", ls="-", label="QReclaim (all passes)"),
               Line2D([], [], color="#52514e", ls=":", marker="o", mfc="white", label="Qiskit transpile O1")]
    fig.legend(handles=bench_h + style_h, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(FIG / "fig_scaling.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
def tex_main(rows: list[dict]) -> None:
    pick = {"BV": "n=128", "HS": "n=64", "RC": "d=11,r=11", "ADD": "n=64", "GRV": "n=10,m=20",
            "QAOA": "n=64"}
    get = {(r["bench"], r["params"], r["config"]): r for r in rows}
    lines = []
    for b, pr in pick.items():
        nr, sc, sb, qm, dn = (get[(b, pr, c)] for c in ("NR", "SCOPE", "SB", "QR-min", "QR-dn"))
        lines.append(
            f"{b}-{size_of(pr)} & {nr['qubits']} & {sc['qubits']} & {sb['qubits']} & "
            f"\\textbf{{{qm['qubits']}}} & {dn['qubits']} & "
            f"{nr['depth']} & {sb['depth']} & {qm['depth']} & {dn['depth']} & "
            f"{nr['cx']} & {qm['cx']} & {qm['reset']}/{dn['reset']} & {qm['compile_ms']:.1f} \\\\")
    (TAB / "tab_main.tex").write_text("\n".join(lines) + "\n")


def tex_ablation(rows: list[dict]) -> None:
    benches = []
    for r in rows:
        if (r["bench"], r["params"]) not in benches:
            benches.append((r["bench"], r["params"]))
    cfgs = []
    for r in rows:
        if r["config"] not in cfgs:
            cfgs.append(r["config"])
    get = {(r["bench"], r["params"], r["config"]): r for r in rows}
    lines = []
    for cfg in cfgs:
        cells = [f"{get[(b, p, cfg)]['qubits']}/{get[(b, p, cfg)]['depth']}" for b, p in benches]
        name = {"QReclaim": "\\QR{} (full)"}.get(cfg, cfg.replace("-", "$-$", 1) if cfg.startswith("-") else cfg)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if cfg == "SB":
            lines.append("\\midrule")
    (TAB / "tab_ablation.tex").write_text("\n".join(lines) + "\n")


def tex_mutation() -> dict:
    path = RES / "mutation.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        rows = list(csv.DictReader(f))
    names = {"alias": "alias an operand", "clobber-early": "overwrite control (early)",
             "clobber-late": "overwrite control (late)", "clobber-after": "overwrite after last use",
             "double-flip": "flip and restore control", "reset-control": "reset a control",
             "superpose": "H on ancilla"}
    lines, tot = [], dict(n=0, f=0, tp=0, fn=0, fp=0, tn=0)
    for op in ["alias", "clobber-early", "clobber-late", "double-flip", "reset-control", "superpose",
               "clobber-after"]:
        rs = [r for r in rows if r["operator"] == op]
        tp = sum(1 for r in rs if r["faulty"] == "1" and r["verdict"] == "rejected")
        fn = sum(1 for r in rs if r["faulty"] == "1" and r["verdict"] == "accepted")
        fp = sum(1 for r in rs if r["faulty"] == "0" and r["verdict"] == "rejected")
        tn = sum(1 for r in rs if r["faulty"] == "0" and r["verdict"] == "accepted")
        for k, v in (("n", len(rs)), ("f", tp + fn), ("tp", tp), ("fn", fn), ("fp", fp), ("tn", tn)):
            tot[k] += v
        lines.append(f"{names[op]} & {len(rs)} & {tp + fn} & {tp} & {fn} & {fp} & {tn} \\\\")
    lines.append("\\midrule")
    lines.append(f"total & {tot['n']} & {tot['f']} & {tot['tp']} & {tot['fn']} & {tot['fp']} & {tot['tn']} \\\\")
    (TAB / "tab_mutation.tex").write_text("\n".join(lines) + "\n")
    tot["gc_fail"] = sum(1 for r in rows if r["runtime_check"] == "GC-FAIL")
    tot["accepted_simulated"] = sum(1 for r in rows if r["runtime_check"] == "gc-ok")
    return tot


def tex_qrisp(rows: list[dict]) -> None:
    path = RES / "qrisp.csv"
    if not path.exists():
        return
    with path.open() as f:
        qr_rows = list(csv.DictReader(f))
    get = {(r["bench"], r["params"], r["config"]): r for r in rows}
    lines = []
    for r in qr_rows:
        if r["config"] != "Qrisp":
            continue
        ws = next(x for x in qr_rows if (x["bench"], x["params"], x["config"]) == (r["bench"], r["params"], "Qrisp+ws2"))
        ours = get.get((r["bench"], r["params"], "QR-min"))
        if ours is None:
            ours = measure_circuit(compile_program(
                {"HS": hidden_shift, "QAOA": qaoa_ring, "ADD": adder}[r["bench"]](size_of(r["params"])).build()
                if r["bench"] != "GRV" else grover_sat(size_of(r["params"]), 2 * size_of(r["params"])).build(), QR))
        lines.append(f"{r['bench']}-{size_of(r['params'])} & {ours['qubits']}/{ours['depth']}/{ours['cx']} & "
                     f"{r['qubits']}/{r['depth']}/{r['cx']} & {ws['qubits']}/{ws['depth']}/{ws['cx']} \\\\")
    (TAB / "tab_qrisp.tex").write_text("\n".join(lines) + "\n")


def summary(main_rows, abl, verify, sca, bud, mut) -> None:
    get = {(r["bench"], r["params"], r["config"]): r for r in main_rows}
    keys = sorted({(r["bench"], r["params"]) for r in main_rows})
    red = lambda a, b: 1 - a / b                                   # noqa: E731
    out = {"instances": len(keys)}
    for cfg in ("SCOPE", "SB", "QR-min", "QR-dn"):
        rs = [red(int(get[(b, p, cfg)]["qubits"]), int(get[(b, p, "NR")]["qubits"])) for b, p in keys]
        out[f"mean_width_reduction_{cfg}_vs_NR_pct"] = round(100 * statistics.mean(rs), 1)
        out[f"geomean_width_ratio_{cfg}_vs_NR"] = round(statistics.geometric_mean([1 - x for x in rs]), 3)
    for base in ("SCOPE", "SB"):
        rs = [red(int(get[(b, p, "QR-min")]["qubits"]), int(get[(b, p, base)]["qubits"])) for b, p in keys]
        out[f"mean_width_reduction_QR-min_vs_{base}_pct"] = round(100 * statistics.mean(rs), 1)
    out["qr_min_depth_ratio_geomean"] = round(statistics.geometric_mean(
        [int(get[(b, p, "QR-min")]["depth"]) / int(get[(b, p, "NR")]["depth"]) for b, p in keys]), 2)
    out["qr_dn_max_depth_ratio"] = round(max(
        int(get[(b, p, "QR-dn")]["depth"]) / int(get[(b, p, "NR")]["depth"]) for b, p in keys), 3)
    out["time_r10_ratio_dn_vs_nr_max"] = round(max(
        int(get[(b, p, "QR-dn")]["time_r10"]) / int(get[(b, p, "NR")]["time_r10"]) for b, p in keys), 3)
    out["time_r10_ratio_dn_vs_nr_min"] = round(min(
        int(get[(b, p, "QR-dn")]["time_r10"]) / int(get[(b, p, "NR")]["time_r10"]) for b, p in keys), 3)
    out["max_compile_ms_main"] = round(max(float(r["compile_ms"]) for r in main_rows), 1)
    out["max_dn_search_ms"] = round(max(float(r["search_ms"]) for r in main_rows if r["search_ms"] != ""), 1)
    sim_main = [r for r in main_rows if r["verified_tvd"] != ""]
    out["main_outputs_simulated"] = len(sim_main)
    out["main_outputs_total"] = len(main_rows)
    out["main_max_tvd"] = max(float(r["verified_tvd"]) for r in sim_main)
    sim_ver = [r for r in verify if r["tvd_exact"] != ""]
    out["verify_programs_simulated"] = len(sim_ver)
    out["verify_programs_total"] = len(verify)
    out["verify_max_tvd"] = max(float(r["tvd_exact"]) for r in sim_ver)
    out["verify_max_tvd_aer"] = max(float(r["tvd_aer"]) for r in sim_ver)
    out["frees_validated_qr_min_main"] = sum(int(r["frees_validated"]) for r in main_rows if r["config"] == "QR-min")
    x = np.log([float(r["instrs"]) for r in sca])
    y = np.log([float(r["compile_ms"]) for r in sca])
    out["scaling_slope_all"] = round(float(np.polyfit(x, y, 1)[0]), 2)
    for b in sorted({r["bench"] for r in sca}):
        pts = [r for r in sca if r["bench"] == b]
        out[f"scaling_slope_{b}"] = round(float(np.polyfit(np.log([float(r["instrs"]) for r in pts]),
                                                           np.log([float(r["compile_ms"]) for r in pts]), 1)[0]), 2)
    out["us_per_instr_median"] = round(statistics.median(
        [1e3 * float(r["compile_ms"]) / float(r["instrs"]) for r in sca]), 1)
    for key in sorted({(r["bench"], r["params"]) for r in bud}):
        pts = [r for r in bud if (r["bench"], r["params"]) == key]
        ref, hi = int(pts[0]["nr_depth"]), int(pts[0]["nr_qubits"])
        first = next((r for r in pts if int(r["depth"]) <= ref), None)
        if first:
            out[f"budget_knee_{key[0]}_pct_of_NR_width"] = round(100 * int(first["budget"]) / hi)
    out.update({f"mutation_{k}": v for k, v in mut.items()})
    if mut:
        out["mutation_recall_pct"] = round(100 * mut["tp"] / max(1, mut["tp"] + mut["fn"]), 1)
        out["mutation_precision_pct"] = round(100 * mut["tp"] / max(1, mut["tp"] + mut["fp"]), 1)
    (RES / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    m = {
        "NInst": out["instances"],
        "RedQRvsNR": out["mean_width_reduction_QR-min_vs_NR_pct"],
        "RedDNvsNR": out["mean_width_reduction_QR-dn_vs_NR_pct"],
        "RedQRvsSCOPE": out["mean_width_reduction_QR-min_vs_SCOPE_pct"],
        "RedQRvsSB": out["mean_width_reduction_QR-min_vs_SB_pct"],
        "GeoQR": out["geomean_width_ratio_QR-min_vs_NR"],
        "DepthQR": out["qr_min_depth_ratio_geomean"],
        "DNMaxDepth": out["qr_dn_max_depth_ratio"],
        "TimeTenMin": out["time_r10_ratio_dn_vs_nr_min"],
        "TimeTenMax": out["time_r10_ratio_dn_vs_nr_max"],
        "MaxCompileMs": out["max_compile_ms_main"],
        "MaxDNSearchMs": out["max_dn_search_ms"],
        "MainSim": out["main_outputs_simulated"], "MainTotal": out["main_outputs_total"],
        "VerSim": out["verify_programs_simulated"], "VerTotal": out["verify_programs_total"],
        "MaxTVD": f"{max(out['main_max_tvd'], out['verify_max_tvd']):.1e}".replace("e-", "\\times10^{-") + "}",
        "VerMaxAer": f"{out['verify_max_tvd_aer']:.3f}",
        "FreesValidated": out["frees_validated_qr_min_main"],
        "Slope": out["scaling_slope_all"], "UsPerInstr": out["us_per_instr_median"],
        "KneeBV": out.get("budget_knee_BV_pct_of_NR_width", "--"),
        "KneeHS": out.get("budget_knee_HS_pct_of_NR_width", "--"),
        "KneeRC": out.get("budget_knee_RC_pct_of_NR_width", "--"),
        "KneeGRV": out.get("budget_knee_GRV_pct_of_NR_width", "--"),
        "MutN": out.get("mutation_n", 0), "MutFaulty": out.get("mutation_f", 0),
        "MutFP": out.get("mutation_fp", 0), "MutRecall": out.get("mutation_recall_pct", 0),
        "MutPrecision": out.get("mutation_precision_pct", 0),
    }
    lines = []
    for k, v in m.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (TAB / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2))


def tex_verify(rows: list[dict]) -> None:
    lines = []
    seen = []
    for r in rows:
        key = (r["bench"], r["params"])
        if key in seen:
            continue
        seen.append(key)
        cells = []
        for cfg in ("NR", "SCOPE", "QR-min"):
            v = next(x for x in rows if (x["bench"], x["params"], x["config"]) == (*key, cfg))
            cells.append(f"{v['qubits']}" + ("$^\\dagger$" if v["tvd_exact"] == "" else ""))
        qr = next(x for x in rows if (x["bench"], x["params"], x["config"]) == (*key, "QR-min"))
        lines.append(f"{key[0]} ({key[1].replace(',', ', ')}) & " + " & ".join(cells)
                     + f" & {qr['gc_checks']} & {float(qr['tvd_exact']):.1e}"
                     + f" & {float(qr['tvd_aer']):.3f} \\\\")
    (TAB / "tab_verify.tex").write_text("\n".join(lines) + "\n")


def load(name: str) -> list[dict]:
    def conv(v):
        for t in (int, float):
            try:
                return t(v)
            except ValueError:
                pass
        return v
    with (RES / name).open() as f:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main() -> None:
    if "--figures-only" in sys.argv:
        main_rows, bud, sca = load("main.csv"), load("budget.csv"), load("scaling.csv")
        abl, verify = load("ablation.csv"), load("verify.csv")
    else:
        env = (f"Python {platform.python_version()}, Qiskit {qiskit.__version__}, "
               f"NumPy {np.__version__}, {platform.system()} {platform.machine()}")
        (RES / "environment.txt").write_text(env + "\n")
        print(env)
        for t, m in run_diagnostics():
            print(f"{t}: {m}")
        verify = run_verify()
        main_rows = run_main()
        abl = run_ablation()
        bud = run_budget()
        sca = run_scaling()
        main_rows, bud, sca = load("main.csv"), load("budget.csv"), load("scaling.csv")
        abl, verify = load("ablation.csv"), load("verify.csv")
    fig_qubits(main_rows)
    fig_budget(bud)
    fig_scaling(sca)
    tex_main(main_rows)
    tex_ablation(abl)
    tex_verify(verify)
    tex_qrisp(main_rows)
    mut = tex_mutation()
    summary(main_rows, abl, verify, sca, bud, mut)
    print("all done")


if __name__ == "__main__":
    main()
