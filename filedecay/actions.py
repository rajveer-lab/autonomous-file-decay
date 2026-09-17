"""Filesystem-mutating actions: compress, archive, quarantine, delete, restore.

Every action except `delete` is fully reversible via `restore`, which is the
safety net the project is built around: nothing is ever permanently removed
until it has already spent a full grace period sitting untouched in
quarantine.
"""
from __future__ import annotations

import gzip
import shutil
import sqlite3
import time
from pathlib import Path

from filedecay.config import Config
from filedecay.store import Store


class ActionError(Exception):
    pass


def _gzip_file(src: Path, dst: Path) -> None:
    with src.open("rb") as fin, gzip.open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def _gunzip_file(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)


def compress(cfg: Config, store: Store, row: sqlite3.Row, reason: str) -> str:
    """Gzip the file in place: foo.txt -> foo.txt.decaygz."""
    src = cfg.root / row["current_rel_path"]
    dst = src.with_name(src.name + ".decaygz")
    _gzip_file(src, dst)
    src.unlink()
    new_rel = str(dst.relative_to(cfg.root))
    store.log_action(row["id"], "compress", row["current_rel_path"], new_rel, reason)
    store.update_file(row["id"], state="compressed", current_rel_path=new_rel, compressed=1)
    return new_rel


def archive(cfg: Config, store: Store, row: sqlite3.Row, reason: str) -> str:
    """Ensure the artifact is compressed, then move it into .decay/archive/,
    preserving the original relative directory structure."""
    current_rel = row["current_rel_path"]
    src = cfg.root / current_rel
    compressed = bool(row["compressed"])
    if not compressed:
        tmp_dst = src.with_name(src.name + ".decaygz")
        _gzip_file(src, tmp_dst)
        src.unlink()
        src = tmp_dst
        compressed = True

    archive_rel = Path("archive") / row["rel_path"]
    if archive_rel.suffix != ".decaygz":
        archive_rel = archive_rel.with_name(archive_rel.name + ".decaygz")
    dest = cfg.decay_dir / archive_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))

    new_rel = str(dest.relative_to(cfg.root))
    store.log_action(row["id"], "archive", current_rel, new_rel, reason)
    store.update_file(row["id"], state="archived", current_rel_path=new_rel, compressed=1)
    return new_rel


def quarantine(cfg: Config, store: Store, row: sqlite3.Row, reason: str) -> str:
    """Move the current artifact (whatever form it's in) into
    .decay/quarantine/ -- a soft delete with a grace period before the file
    is permanently removed."""
    current_rel = row["current_rel_path"]
    src = cfg.root / current_rel
    ts = int(time.time())
    quarantine_name = f"{row['id']}__{ts}__{Path(current_rel).name}"
    dest = cfg.quarantine_dir / quarantine_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))

    new_rel = str(dest.relative_to(cfg.root))
    store.log_action(row["id"], "quarantine", current_rel, new_rel, reason)
    store.update_file(row["id"], state="quarantined", current_rel_path=new_rel)
    return new_rel


def delete(cfg: Config, store: Store, row: sqlite3.Row, reason: str) -> None:
    """Permanently remove a quarantined file. The one irreversible action,
    and it only ever runs after a file has sat untouched in quarantine for
    the configured grace period."""
    src = cfg.root / row["current_rel_path"]
    src.unlink(missing_ok=True)
    store.log_action(row["id"], "delete", row["current_rel_path"], None, reason)
    store.update_file(row["id"], state="deleted", current_rel_path="")


def restore(cfg: Config, store: Store, row: sqlite3.Row) -> Path:
    """Undo: bring a compressed/archived/quarantined file back to its
    original path, in its original (decompressed) form, active again."""
    if row["state"] == "active":
        raise ActionError(f"'{row['rel_path']}' is already active -- nothing to undo")
    if row["state"] == "deleted":
        raise ActionError(
            f"'{row['rel_path']}' was permanently deleted and cannot be restored"
        )

    current_rel = row["current_rel_path"]
    src = cfg.root / current_rel
    if not src.exists():
        raise ActionError(f"expected artifact missing on disk: {current_rel}")

    original_abs = cfg.root / row["rel_path"]
    if original_abs.exists():
        raise ActionError(
            f"cannot restore: a file already exists at '{row['rel_path']}'"
        )
    original_abs.parent.mkdir(parents=True, exist_ok=True)

    if row["compressed"]:
        _gunzip_file(src, original_abs)
        src.unlink()
    else:
        shutil.move(str(src), str(original_abs))

    store.log_action(row["id"], "restore", current_rel, row["rel_path"], "manual undo")
    store.update_file(
        row["id"],
        state="active",
        current_rel_path=row["rel_path"],
        compressed=0,
        last_mtime=original_abs.stat().st_mtime,
    )
    return original_abs
