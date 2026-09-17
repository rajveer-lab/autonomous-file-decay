"""Config loading for autonomous-file-decay.

Config lives at ``<root>/.decay/config.toml``. Everything is plain TOML,
read with the stdlib ``tomllib`` -- no extra dependencies required.
"""
from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

DEFAULT_CONFIG_TOML = """\
# autonomous-file-decay configuration
#
# All thresholds are in days of *inactivity* (idle time), where idle time
# for a live file is `now - file mtime`. Editing a file (or running
# `filedecay touch <path>`) resets its clock.

[thresholds]
compress_after_days   = 7     # gzip the file in place
archive_after_days    = 21    # move the (compressed) file into .decay/archive/
quarantine_after_days = 45    # move into .decay/quarantine/ -- a soft delete,
                               # still fully recoverable with `filedecay undo`
delete_after_days     = 15    # days spent IN QUARANTINE before permanent
                               # deletion. This is the only irreversible step,
                               # and it only ever happens after the grace
                               # period above.

[scan]
interval_seconds = 3600
ignore = [".git", ".decay", "__pycache__", "node_modules", ".venv"]

# Glob patterns (relative to the watched root) that never decay, no matter
# how idle they are.
pinned = []
"""


@dataclasses.dataclass
class Thresholds:
    compress_after_days: float
    archive_after_days: float
    quarantine_after_days: float
    delete_after_days: float


@dataclasses.dataclass
class Config:
    root: Path
    thresholds: Thresholds
    interval_seconds: int
    ignore: list[str]
    pinned: list[str]

    @property
    def decay_dir(self) -> Path:
        return self.root / ".decay"

    @property
    def archive_dir(self) -> Path:
        return self.decay_dir / "archive"

    @property
    def quarantine_dir(self) -> Path:
        return self.decay_dir / "quarantine"

    @property
    def db_path(self) -> Path:
        return self.decay_dir / "state.db"


def config_path(root: Path) -> Path:
    return root / ".decay" / "config.toml"


def init_config(root: Path, force: bool = False) -> Path:
    """Create .decay/{config.toml,archive/,quarantine/} under root."""
    decay_dir = root / ".decay"
    decay_dir.mkdir(parents=True, exist_ok=True)
    (decay_dir / "archive").mkdir(exist_ok=True)
    (decay_dir / "quarantine").mkdir(exist_ok=True)
    cfg_path = decay_dir / "config.toml"
    if cfg_path.exists() and not force:
        return cfg_path
    cfg_path.write_text(DEFAULT_CONFIG_TOML)
    return cfg_path


def load_config(root: Path) -> Config:
    root = root.resolve()
    cfg_path = config_path(root)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No config found at {cfg_path}. Run `filedecay --root {root} init` first."
        )
    data = tomllib.loads(cfg_path.read_text())
    th = data.get("thresholds", {})
    sc = data.get("scan", {})
    thresholds = Thresholds(
        compress_after_days=float(th.get("compress_after_days", 7)),
        archive_after_days=float(th.get("archive_after_days", 21)),
        quarantine_after_days=float(th.get("quarantine_after_days", 45)),
        delete_after_days=float(th.get("delete_after_days", 15)),
    )
    return Config(
        root=root,
        thresholds=thresholds,
        interval_seconds=int(sc.get("interval_seconds", 3600)),
        ignore=list(sc.get("ignore", [])),
        pinned=list(sc.get("pinned", [])),
    )
