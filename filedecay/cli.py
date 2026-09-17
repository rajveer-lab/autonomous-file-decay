"""Command-line interface for autonomous-file-decay."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from filedecay import actions, config, engine
from filedecay.store import Store

DAY = 86400


def _load(root: Path) -> tuple[config.Config, Store]:
    cfg = config.load_config(root)
    store = Store(cfg.db_path)
    return cfg, store


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = config.init_config(root, force=args.force)
    print(f"Initialized decay tracking at {root}")
    print(f"Config: {cfg_path}")
    print("Edit the [thresholds] there, then run:")
    print(f"  filedecay --root {root} scan   # dry run")
    print(f"  filedecay --root {root} run --apply")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    cfg, store = _load(Path(args.root))
    with store:
        planned = engine.run_pass(cfg, store, apply=False)
        _print_plan(planned, dry_run=True)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg, store = _load(Path(args.root))
    with store:
        if args.daemon:
            print(f"Watching {cfg.root} every {cfg.interval_seconds}s (Ctrl-C to stop)")
            try:
                while True:
                    planned = engine.run_pass(cfg, store, apply=args.apply)
                    _print_plan(planned, dry_run=not args.apply, timestamped=True)
                    time.sleep(cfg.interval_seconds)
            except KeyboardInterrupt:
                print("\nStopped.")
        else:
            planned = engine.run_pass(cfg, store, apply=args.apply)
            _print_plan(planned, dry_run=not args.apply)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg, store = _load(Path(args.root))
    with store:
        rows = store.list_files()
        if not rows:
            print("No files tracked yet. Run `filedecay scan` or `filedecay run --apply`.")
            return 0
        now = time.time()
        header = f"{'STATE':<12} {'IDLE':>8}  {'PATH'}"
        print(header)
        print("-" * len(header))
        for row in rows:
            if row["state"] == "quarantined":
                idle = f"{(now - row['state_changed_at']) / DAY:.1f}d*"
            elif row["state"] in ("deleted", "missing"):
                idle = "-"
            else:
                idle = f"{(now - row['last_mtime']) / DAY:.1f}d"
            print(f"{row['state']:<12} {idle:>8}  {row['rel_path']}")
        print("\n(* = time spent in quarantine, not original idle time)")
    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    cfg, store = _load(Path(args.root))
    with store:
        rows = store.list_actions()
        if not rows:
            print("No actions recorded yet.")
            return 0
        for row in rows:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["timestamp"]))
            file_row = store.get_file_by_id(row["file_id"])
            name = file_row["rel_path"] if file_row else f"file#{row['file_id']}"
            undone = " (undone)" if row["undone"] else ""
            print(f"[{ts}] {row['action']:<10} {name} -- {row['reason']}{undone}")
    return 0


def cmd_touch(args: argparse.Namespace) -> int:
    cfg, store = _load(Path(args.root))
    with store:
        rel = args.path
        row = store.get_file(rel)
        abs_path = cfg.root / rel
        if row is not None and row["state"] != "active":
            print(
                f"'{rel}' is currently {row['state']}; run `filedecay undo {rel}` first."
            )
            return 1
        if not abs_path.exists():
            print(f"No such file: {abs_path}")
            return 1
        now = time.time()
        os.utime(abs_path, (now, now))
        print(f"Touched '{rel}' -- decay clock reset.")
    return 0


def cmd_undo(args: argparse.Namespace) -> int:
    cfg, store = _load(Path(args.root))
    with store:
        row = store.get_file(args.path)
        if row is None:
            print(f"'{args.path}' is not tracked.")
            return 1
        try:
            restored = actions.restore(cfg, store, row)
        except actions.ActionError as e:
            print(f"Cannot undo: {e}")
            return 1
        print(f"Restored '{args.path}' -> {restored}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Populate the root with sample files backdated to different ages, so
    a single `run --apply` shows every stage of the pipeline at once."""
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config.init_config(root)

    samples = {
        "notes/todo.md": 0,
        "notes/old_meeting_notes.md": 10,
        "reports/q1_report.csv": 25,
        "reports/q2_report.csv": 50,
        "scratch/temp_dump.log": 70,
    }
    now = time.time()
    for rel, age_days in samples.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"sample content for {rel}\ncreated by `filedecay demo`\n")
        ts = now - age_days * DAY
        os.utime(path, (ts, ts))

    print(f"Demo files created under {root}:")
    for rel, age_days in samples.items():
        print(f"  {rel:<32} ({age_days}d old)")
    print("\nNow try:")
    print(f"  filedecay --root {root} scan")
    print(f"  filedecay --root {root} run --apply")
    print(f"  filedecay --root {root} status")
    return 0


def _print_plan(planned, dry_run: bool, timestamped: bool = False) -> None:
    prefix = time.strftime("[%H:%M:%S] ", time.localtime()) if timestamped else ""
    if not planned:
        print(f"{prefix}Nothing to do.")
        return
    verb = "Would" if dry_run else "Did"
    for p in planned:
        print(f"{prefix}{verb} {p.from_state} -> {p.to_state}: {p.rel_path}  ({p.reason})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="filedecay",
        description=(
            "Autonomous file decay: watches a directory, tracks how active "
            "each file is, and automatically compresses/archives/quarantines/"
            "deletes files that fall out of use -- with a full safety net to "
            "undo any action short of final deletion."
        ),
    )
    p.add_argument("--root", default=".", help="Watched directory (default: current directory)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="Initialize .decay/ config & state store")
    sp.add_argument("--force", action="store_true", help="Overwrite an existing config")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("scan", help="Dry run: show what WOULD happen, change nothing")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("run", help="Evaluate decay rules once, or loop as a daemon")
    sp.add_argument("--apply", action="store_true", help="Actually perform actions (default: dry run)")
    sp.add_argument("--daemon", action="store_true", help="Loop forever using scan.interval_seconds")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("status", help="Show every tracked file and its current stage")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("journal", help="Show the full action history (audit log)")
    sp.set_defaults(func=cmd_journal)

    sp = sub.add_parser("touch", help="Reset a file's decay clock (mark it freshly active)")
    sp.add_argument("path", help="Path relative to --root")
    sp.set_defaults(func=cmd_touch)

    sp = sub.add_parser("undo", help="Restore a file to active/original form from any stage")
    sp.add_argument("path", help="Original relative path, as shown by `status`")
    sp.set_defaults(func=cmd_undo)

    sp = sub.add_parser("demo", help="Populate --root with sample files at various ages")
    sp.set_defaults(func=cmd_demo)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
