"""Tables (Markdown) and figures from the result CSVs."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np

ORDER = ["FIFO", "LRU", "LFU", "CLOCK", "ARC", "LeCaR", "SIEVE", "S3-FIFO",
         "SampledLRU-16", "Learned", "OPT"]
# validated categorical palette (light mode), assigned in fixed order
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
PLOT = ["LRU", "CLOCK", "ARC", "LeCaR", "S3-FIFO", "Learned"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("faults", "accesses", "capacity", "decisions"):
            if k in r and r[k] != "":
                r[k] = int(r[k])
        for k in ("fault_rate", "fraction", "decision_us"):
            if k in r and r[k] != "":
                r[k] = float(r[k])
    return rows


def metrics(rows):
    """Per policy: fault reduction vs LRU, LRU-to-OPT gap closed, worst case."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["trace"], r["fraction"])][r["policy"]] = r
    out = defaultdict(lambda: defaultdict(list))
    for cell in by.values():
        lru, opt = cell["LRU"]["faults"], cell["OPT"]["faults"]
        for name, r in cell.items():
            f = r["faults"]
            out[name]["reduction"].append(100 * (lru - f) / lru)
            if lru - opt > 0:
                out[name]["gap"].append(100 * (lru - f) / (lru - opt))
            out[name]["worse"].append(100 * (f - lru) / lru)
            out[name]["fault_rate"].append(r["fault_rate"])
            if r.get("decisions"):
                out[name]["us"].append(r["decision_us"])
    return out


def fmt(x, d=1):
    return "–" if x is None or x != x else f"{x:.{d}f}"


def main(out="results", figures="figures"):
    rows = read(os.path.join(out, "test.csv"))
    with open(os.path.join(out, "summary.json")) as f:
        summary = json.load(f)
    os.makedirs(figures, exist_ok=True)
    m = metrics(rows)
    names = [n for n in ORDER if n in m]
    traces = list(dict.fromkeys(r["trace"] for r in rows))
    fracs = sorted({r["fraction"] for r in rows})
    md = ["# Results", ""]
    md.append(f"Test traces: {', '.join(traces)}. Memory sizes: "
              f"{', '.join(f'{100 * f:g}%' for f in fracs)} of each trace's distinct pages. "
              "Every number below comes from `results/test.csv`.")
    md += ["", "## Summary over all test traces and memory sizes", "",
           "| Policy | Mean fault rate | Fault reduction vs LRU (mean) | LRU-to-OPT gap closed (mean) | Worst case vs LRU | Decision time per eviction (Python) | Metadata per page |",
           "|---|---|---|---|---|---|---|"]
    meta = summary["metadata_bytes_per_page"]
    for n in names:
        d = m[n]
        md.append(f"| {n} | {fmt(np.mean(d['fault_rate']), 3)} | {fmt(np.mean(d['reduction']))}% | "
                  f"{fmt(np.mean(d['gap']) if d['gap'] else None)}% | {fmt(max(d['worse']))}% | "
                  f"{fmt(np.median(d['us']) if d['us'] else None, 2)} µs | {meta.get(n, '–')} B |")
    md += ["", "Fault reduction = (F_LRU − F_policy) / F_LRU × 100. Gap closed = (F_LRU − F_policy) / (F_LRU − F_OPT) × 100, "
           "averaged over cells where LRU and OPT differ. Worst case = the largest extra-fault percentage over LRU in any single cell. "
           "Decision time is the median time to pick a victim in this Python simulator; compare policies against each other, not against a kernel.", ""]

    # per-trace table at each memory size
    md += ["## Fault rate per trace", ""]
    for frac in fracs:
        md += [f"### Memory = {100 * frac:g}% of distinct pages", "",
               "| Trace | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for tr in traces:
            cells = {r["policy"]: r for r in rows if r["trace"] == tr and r["fraction"] == frac}
            best = min(cells[n]["faults"] for n in names if n != "OPT" and n in cells)
            vals = []
            for n in names:
                v = cells[n]["fault_rate"]
                s = f"{v:.3f}"
                vals.append(f"**{s}**" if n != "OPT" and cells[n]["faults"] == best else s)
            md.append(f"| {tr} | " + " | ".join(vals) + " |")
        md.append("")
    md.append("Bold marks the fewest faults among the online policies (OPT needs the future).")

    model = summary["model"]
    md += ["", "## Learned policy", "",
           f"- Model: {model['kind']}, {model['params']} parameters ({model['bytes']} bytes as float32), "
           f"trained on {model['train_rows']} OPT-labelled rows in {model['train_seconds']} s.",
           f"- Label horizon H = {model['horizon']} × memory size; sample k = {model['k']} pages per fault; "
           f"fallback to LRU-in-sample when the best score is below {model['threshold']}.",
           f"- Chosen on validation traces only; the test traces were not used for any choice.", ""]

    abl_path = os.path.join(out, "ablation.csv")
    if os.path.exists(abl_path):
        abl = read(abl_path)
        base = {(r["trace"], r["fraction"]): r["faults"] for r in rows if r["policy"] == "LRU"}
        learned = {(r["trace"], r["fraction"]): r["faults"] for r in rows if r["policy"] == "Learned"}
        md += ["## Feature ablation (test traces)", "",
               "| Model | Fault reduction vs LRU (mean) | Change vs full model |", "|---|---|---|"]
        full = np.mean([100 * (base[k] - learned[k]) / base[k] for k in learned])
        md.append(f"| all features | {full:.1f}% | – |")
        for name in dict.fromkeys(r["policy"] for r in abl):
            red = np.mean([100 * (base[(r['trace'], r['fraction'])] - r["faults"]) / base[(r['trace'], r['fraction'])]
                           for r in abl if r["policy"] == name])
            md.append(f"| {name} | {red:.1f}% | {red - full:+.1f} points |")
        md.append("")

    md += ["## How the run went", ""] + [f"- {line}" for line in summary["log"]]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    _figures(rows, traces, fracs, m, figures)
    print(f"wrote {out}/summary.md and figures in {figures}/")


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def _figures(rows, traces, fracs, m, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "text.color": INK,
                         "axes.labelcolor": MUTED})
    # 1. fault rate vs memory size, one panel per test trace
    cols = 4
    rws = (len(traces) + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(11, 2.6 * rws), squeeze=False)
    xs = [100 * f for f in fracs]
    for ax, tr in zip(axes.flat, traces):
        cells = defaultdict(dict)
        for r in rows:
            if r["trace"] == tr:
                cells[r["policy"]][r["fraction"]] = r["fault_rate"]
        for i, n in enumerate(PLOT):
            ax.plot(xs, [cells[n][f] for f in fracs], color=PALETTE[i], lw=1.6,
                    marker="o", ms=3.5, label=n, zorder=3)
        ax.plot(xs, [cells["OPT"][f] for f in fracs], color=MUTED, lw=1.2, ls="--",
                label="OPT (offline bound)", zorder=2)
        ax.set_title(tr, loc="left", color=INK)
        ax.set_xscale("log")
        ax.set_xticks(xs, [f"{x:g}%" for x in xs])
        ax.minorticks_off()
        _style(ax)
    for ax in list(axes.flat)[len(traces):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("fault rate")
    for ax in axes[-1]:
        ax.set_xlabel("memory, % of distinct pages")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(figures, "fault_rate_vs_memory.png"), dpi=170)
    plt.close(fig)

    # 2. mean fault reduction vs LRU per trace
    show = [n for n in ("CLOCK", "ARC", "LeCaR", "S3-FIFO", "Learned") if n in m]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    width = 0.8 / len(show)
    x = np.arange(len(traces))
    for i, n in enumerate(show):
        vals = []
        for tr in traces:
            cell = [r for r in rows if r["trace"] == tr]
            lru = {r["fraction"]: r["faults"] for r in cell if r["policy"] == "LRU"}
            vals.append(np.mean([100 * (lru[r["fraction"]] - r["faults"]) / lru[r["fraction"]]
                                 for r in cell if r["policy"] == n]))
        ax.bar(x + (i - (len(show) - 1) / 2) * width, vals, width * 0.92,
               color=PALETTE[PLOT.index(n)], label=n, zorder=3)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(x, traces, fontsize=8)
    ax.set_ylabel("fault reduction vs LRU, %")
    _style(ax)
    ax.legend(ncol=len(show), frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "reduction_vs_lru.png"), dpi=170)
    plt.close(fig)
