# autonomous-file-decay

A background daemon watches a directory, tracks how "active" each file is,
and automatically compresses / archives / deletes files that fall out of
use — based on rules you define, with a safety net to undo any action.

This is a working **prototype**: pure Python standard library, no external
dependencies, runs anywhere Python 3.11+ is installed.

## The pipeline

```
active  --idle-->  compressed  --idle-->  archived  --idle-->  quarantined  --grace period-->  deleted
  |                                                                 |
  '------------------------- undo (any stage) <---------------------'
```

- **active** — a normal, live file. Its "idle clock" is its mtime: editing
  it (or running `filedecay touch`) resets the clock.
- **compressed** — gzipped in place (`foo.txt` -> `foo.txt.decaygz`), same
  directory.
- **archived** — moved into `.decay/archive/`, preserving the original
  relative path.
- **quarantined** — moved into `.decay/quarantine/`. This is a soft delete:
  the file is out of your way but still fully recoverable.
- **deleted** — permanently removed. The *only* irreversible step, and it
  only happens after a file has sat untouched in quarantine for a
  configurable grace period.

A very stale file (idle longer than several thresholds at once — e.g. a
pre-existing old file the tool has never seen) jumps straight to the
deepest applicable stage on its first scan rather than crawling through
every stage one scan at a time.

Every non-delete action is logged to an append-only journal and can be
undone with `filedecay undo <path>`, which restores the file to its
original path in its original, decompressed form.

## Quickstart

```bash
# See it work immediately with backdated sample files (no waiting on real time):
python3 -m filedecay --root /tmp/decay_demo demo

# Preview what would happen -- changes nothing:
python3 -m filedecay --root /tmp/decay_demo scan

# Apply it for real:
python3 -m filedecay --root /tmp/decay_demo run --apply

# See current state of every tracked file:
python3 -m filedecay --root /tmp/decay_demo status

# Full audit trail:
python3 -m filedecay --root /tmp/decay_demo journal

# Bring a file back from any stage short of permanent deletion:
python3 -m filedecay --root /tmp/decay_demo undo reports/q2_report.csv
```

On a real directory:

```bash
python3 -m filedecay --root ~/Downloads init      # writes .decay/config.toml
python3 -m filedecay --root ~/Downloads scan       # dry run, see what's stale
python3 -m filedecay --root ~/Downloads run --apply --daemon   # run forever
```

If you `pip install -e .`, a `filedecay` console script is also available
(same subcommands, no `python3 -m` needed).

## Configuration

`filedecay init` writes `.decay/config.toml` in the watched directory:

```toml
[thresholds]
compress_after_days   = 7
archive_after_days    = 21
quarantine_after_days = 45
delete_after_days     = 15   # days spent IN quarantine, not overall idle time

[scan]
interval_seconds = 3600
ignore = [".git", ".decay", "__pycache__", "node_modules", ".venv"]

pinned = []   # glob patterns that never decay, e.g. ["important/*", "*.key"]
```

Edit it and re-run — no restart machinery needed, config is re-read every
pass.

## Design notes

- **State lives in SQLite** (`.decay/state.db`): one row per tracked file
  with its current stage, plus an `actions` table that's an append-only
  journal (what happened, when, and why) so `undo` and `status` never have
  to guess.
- **Idle time is mtime-based**, not atime-based. atime is frequently
  disabled by `noatime`/`relatime` mounts and, worse, would be bumped by
  the tool's own scans, creating a false "still active" signal. mtime only
  moves when a file is genuinely written to.
- **Nothing is destroyed on the first pass.** The only truly irreversible
  action (permanent delete) requires a file to have already spent a full
  grace period sitting untouched in quarantine — a deliberate two-step
  safety net.
- **Dry run by default.** `filedecay run` without `--apply` only ever
  prints what it would do.

## Project layout

```
filedecay/
  config.py    # load/write .decay/config.toml
  scanner.py   # walk the directory, read mtime/size, ignore/pin matching
  decay.py     # pure decision logic: idle time -> next stage
  store.py     # SQLite state + action journal
  actions.py   # compress / archive / quarantine / delete / restore
  engine.py    # ties the above into one scan+decide(+apply) pass
  cli.py       # `filedecay` command-line interface
tests/         # unittest suite (stdlib only, no pytest required)
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## What's a prototype vs. what a production version would add

- Real-time watching (`inotify`/`watchdog`) instead of poll-based scanning.
- Per-file custom rules (not just global thresholds), e.g. by extension or
  directory.
- A read-only web/TUI dashboard over `status`/`journal`.
- Cross-platform packaging (systemd unit / launchd agent / Windows service)
  for `run --daemon`.
