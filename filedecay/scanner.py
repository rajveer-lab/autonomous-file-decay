"""Directory walking and activity signal collection."""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable, Iterator


def is_ignored(rel_path: Path, ignore_patterns: Iterable[str]) -> bool:
    parts = rel_path.parts
    rel_str = str(rel_path)
    for pattern in ignore_patterns:
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
        if fnmatch.fnmatch(rel_str, pattern):
            return True
    return False


def is_pinned(rel_path: Path, pinned_patterns: Iterable[str]) -> bool:
    rel_str = str(rel_path)
    return any(fnmatch.fnmatch(rel_str, pattern) for pattern in pinned_patterns)


def iter_files(root: Path, ignore_patterns: Iterable[str]) -> Iterator[Path]:
    """Yield relative paths of every non-ignored, non-directory file under root."""
    patterns = list(ignore_patterns)
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        if is_ignored(rel, patterns):
            continue
        yield rel


def file_signature(path: Path) -> tuple[int, float]:
    """Return (size_bytes, mtime). mtime is the activity clock: it only
    moves when the file is actually written to (or explicitly `touch`ed),
    which is far more portable than atime (often disabled by noatime/relatime
    mounts, and would otherwise be bumped by our own scans)."""
    st = path.stat()
    return st.st_size, st.st_mtime
