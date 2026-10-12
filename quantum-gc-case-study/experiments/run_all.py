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

from qreclaim import (ABLATION, NR, QR, SCOPE, CompileError, Program,   # noqa: E402
                      compile_depth_neutral, compile_program, measure_circuit, with_budget)
from qreclaim.benchmarks import (SUITES, VERIFY, adder, bv, grover_sat,  # noqa: E402
                                 hidden_shift, repetition_code)
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
            qr = compile_program(b.build(), QR)
            mqr = measure_circuit(qr)
            t0 = time.perf_counter()
            dn = compile_depth_neutral(b.build, mnr["depth"], qr.nqubits, nr.nqubits)
            search_ms = 1e3 * (time.perf_counter() - t0)
            for name, m in (("NR", mnr), ("SCOPE", msc), ("QR-min", mqr), ("QR-dn", dn.metrics)):
                rows.append(dict(bench=b.name, params=b.params, size=size_of(b.params), config=name,
                                 ir_instrs=len(b.build().instrs),
                                 search_ms=round(search_ms, 2) if name == "QR-dn" else "",
                                 **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}))
            print(f"main  {b.name:4s} {b.params:10s} NR={mnr['qubits']} SCOPE={msc['qubits']} "
                  f"QRmin={mqr['qubits']} QRdn={dn.metrics['qubits']}", flush=True)
    write_csv(RES / "main.csv", rows)
    return rows


def run_ablation() -> list[dict]:
    rows = []
    for b in (bv(32), hidden_shift(16), repetition_code(7, 7), adder(16), grover_sat(8, 16)):
        for cfg in ABLATION:
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
        for _ in range(3):
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
    out.append(("E1 aliasing (no-cloning)", p))

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
    order = ["BV", "HS", "RC", "ADD", "GRV"]
    xlab = {"BV": "secret bits $n$", "HS": "half-width $n$", "RC": "distance $d$ ($r=d$)",
            "ADD": "operand bits $n$", "GRV": "variables $n$ ($m=2n$)"}
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.75))
    for ax, bname in zip(axes, order):
        for cfg in STYLE:
            pts = sorted((r["size"], r["qubits"]) for r in rows if r["bench"] == bname and r["config"] == cfg)
            xs, ys = zip(*pts)
            st = STYLE[cfg]
            ax.plot(xs, ys, color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"])
        ax.set_title(bname)
        ax.set_xlabel(xlab[bname])
        ax.set_yscale("log", base=2)
        if bname in ("BV", "HS", "ADD"):
            ax.set_xscale("log", base=2)
        else:
            ax.set_xticks(sorted({r["size"] for r in rows if r["bench"] == bname}))
        ax.grid(True, which="major")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("physical qubits")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(w_pad=0.6)
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
    pick = {"BV": "n=128", "HS": "n=64", "RC": "d=11,r=11", "ADD": "n=64", "GRV": "n=10,m=20"}
    get = {(r["bench"], r["params"], r["config"]): r for r in rows}
    lines = []
    for b, pr in pick.items():
        nr, sc, qm, dn = (get[(b, pr, c)] for c in ("NR", "SCOPE", "QR-min", "QR-dn"))

        def red(v):
            return f"{100 * (1 - v / nr['qubits']):.0f}\\%"

        lines.append(
            f"{b} ({pr.replace(',', ', ')}) & {nr['qubits']} & {sc['qubits']} & "
            f"\\textbf{{{qm['qubits']}}} ({red(qm['qubits'])}) & {dn['qubits']} ({red(dn['qubits'])}) & "
            f"{nr['depth']} & {sc['depth']} & {qm['depth']} & {dn['depth']} & "
            f"{nr['cx']} & {qm['cx']} & {qm['reset']} & {qm['compile_ms']:.1f} \\\\")
    (TAB / "tab_main.tex").write_text("\n".join(lines) + "\n")


def tex_ablation(rows: list[dict]) -> None:
    benches = []
    for r in rows:
        if (r["bench"], r["params"]) not in benches:
            benches.append((r["bench"], r["params"]))
    cfgs = [c.name for c in ABLATION]
    get = {(r["bench"], r["params"], r["config"]): r for r in rows}
    lines = []
    for cfg in cfgs:
        cells = [f"{get[(b, p, cfg)]['qubits']} / {get[(b, p, cfg)]['depth']}" for b, p in benches]
        lines.append(f"{cfg.replace('+', '+\\,')} & " + " & ".join(cells) + " \\\\")
    (TAB / "tab_ablation.tex").write_text("\n".join(lines) + "\n")


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
        fig_qubits(main_rows)
        fig_budget(bud)
        fig_scaling(sca)
        tex_main(main_rows)
        tex_ablation(load("ablation.csv"))
        tex_verify(load("verify.csv"))
        return
    env = (f"Python {platform.python_version()}, Qiskit {qiskit.__version__}, "
           f"NumPy {np.__version__}, {platform.system()} {platform.machine()}")
    (RES / "environment.txt").write_text(env + "\n")
    print(env)
    msgs = run_diagnostics()
    for t, m in msgs:
        print(f"{t}: {m}")
    verify = run_verify()
    main_rows = run_main()
    abl = run_ablation()
    bud = run_budget()
    sca = run_scaling()
    fig_qubits(main_rows)
    fig_budget(bud)
    fig_scaling(sca)
    tex_main(main_rows)
    tex_ablation(abl)
    tex_verify(verify)
    print("all done")


if __name__ == "__main__":
    main()
