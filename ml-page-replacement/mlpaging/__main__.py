"""Command line: python -m mlpaging {serve,import,record,verify,run,report,demo,msr}."""

import argparse
import sys

from . import experiment, report
from . import policies as P
from . import traces as T
from .sim import simulate


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mlpaging")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="check FIFO/LRU/OPT on the textbook string")
    r = sub.add_parser("run", help="train, tune and evaluate everything")
    r.add_argument("--quick", action="store_true", help="smaller run for a smoke test")
    r.add_argument("--out", default="results")
    r.add_argument("--figures", default="figures")
    r.add_argument("--procs", type=int, default=None)
    rp = sub.add_parser("report", help="rebuild tables and figures from results/")
    rp.add_argument("--out", default="results")
    rp.add_argument("--figures", default="figures")
    d = sub.add_parser("demo", help="run the baselines on one trace")
    d.add_argument("--trace", default="textbook",
                   choices=["textbook", "loop", "zipf", "phase", "scanhot", "mixed", "matmul", "mergesort", "bfs"])
    d.add_argument("--frames", type=int, default=None)
    m = sub.add_parser("msr", help="run the baselines on an MSR Cambridge trace file")
    m.add_argument("path")
    m.add_argument("--fraction", type=float, default=0.05)
    m.add_argument("--limit", type=int, default=500_000)
    sv = sub.add_parser("serve", help="start the PageLab web application")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--data", default="~/.mlpaging", help="where workloads, models and runs are kept")
    sv.add_argument("--no-browser", action="store_true")
    im = sub.add_parser("import", help="add a trace file to PageLab's workloads")
    im.add_argument("path")
    im.add_argument("--format", default="auto")
    im.add_argument("--name")
    im.add_argument("--data", default="~/.mlpaging")
    rc = sub.add_parser("record", help="record a program's page references into PageLab")
    rc.add_argument("--tool", choices=["valgrind", "strace"], default="valgrind")
    rc.add_argument("--cap", type=int, default=2_000_000)
    rc.add_argument("--data", default="~/.mlpaging")
    rc.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        from .app import serve
        serve(args.port, args.data, not args.no_browser)
        return 0
    if args.cmd in ("import", "record"):
        from . import realtrace as RT
        from .app import Store
        store = Store(args.data)
        if args.cmd == "import":
            built, fmt = RT.load(args.path, fmt=args.format)
            doc = store.save_trace(args.name or args.path.rsplit("/", 1)[-1], built,
                                   {"kind": "file", "format": fmt, "file": args.path})
        else:
            cmd = [c for c in args.command if c != "--"]
            built, info = RT.record(cmd, tool=args.tool, cap=args.cap)
            doc = store.save_trace(f"{cmd[0]} ({args.tool})", built, {"kind": "recorded", **info})
        print(f"added {doc['name']}: {doc['refs']:,} references, {doc['distinct']:,} pages (id {doc['id']})")
        return 0

    if args.cmd == "verify":
        print(experiment.verify_textbook(), "(expected FIFO 15, LRU 12, OPT 9)")
    elif args.cmd == "run":
        experiment.main(out=args.out, quick=args.quick, procs=args.procs)
        report.main(out=args.out, figures=args.figures)
    elif args.cmd == "report":
        report.main(out=args.out, figures=args.figures)
    elif args.cmd in ("demo", "msr"):
        if args.cmd == "msr":
            tr = T.load_msr(args.path, limit=args.limit)
            c = max(8, round(args.fraction * tr.distinct))
        else:
            gen = {"textbook": T.textbook, "loop": T.loop, "zipf": T.zipf, "phase": T.phase_change,
                   "scanhot": T.scan_hot, "mixed": T.mixed, "matmul": T.matmul,
                   "mergesort": T.mergesort, "bfs": T.bfs}[args.trace]
            tr = gen()
            c = args.frames or (3 if args.trace == "textbook" else max(8, round(0.05 * tr.distinct)))
        print(f"{tr.name}: {len(tr)} references, {tr.distinct} distinct pages, {c} frames")
        for pol in P.baselines():
            r = simulate(pol, tr.pages, c)
            print(f"  {pol.name:<14} faults {r['faults']:>8}  fault rate {r['fault_rate']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
