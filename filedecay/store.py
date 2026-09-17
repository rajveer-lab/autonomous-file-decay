"""SQLite-backed state store: current status of every tracked file, plus an
append-only action journal that makes every non-delete action undoable."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rel_path TEXT UNIQUE NOT NULL,      -- canonical original path (identity)
    state TEXT NOT NULL DEFAULT 'active',
    current_rel_path TEXT NOT NULL,     -- where the artifact lives right now
    compressed INTEGER NOT NULL DEFAULT 0,
    size INTEGER,
    last_mtime REAL,                    -- last observed real activity
    state_changed_at REAL,
    created_at REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id),
    action TEXT NOT NULL,
    from_rel_path TEXT,
    to_rel_path TEXT,
    reason TEXT,
    timestamp REAL NOT NULL,
    undone INTEGER NOT NULL DEFAULT 0
);
"""


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- files -----------------------------------------------------------

    def get_file(self, rel_path: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM files WHERE rel_path = ?", (rel_path,))
        return cur.fetchone()

    def get_file_by_id(self, file_id: int) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM files WHERE id = ?", (file_id,))
        return cur.fetchone()

    def list_files(self, states: Optional[tuple[str, ...]] = None) -> list[sqlite3.Row]:
        if states:
            placeholders = ",".join("?" for _ in states)
            cur = self.conn.execute(
                f"SELECT * FROM files WHERE state IN ({placeholders}) ORDER BY rel_path",
                states,
            )
        else:
            cur = self.conn.execute("SELECT * FROM files ORDER BY rel_path")
        return cur.fetchall()

    def insert_active(self, rel_path: str, size: int, mtime: float, now: Optional[float] = None) -> int:
        now = now if now is not None else time.time()
        cur = self.conn.execute(
            "INSERT INTO files (rel_path, state, current_rel_path, compressed, size, "
            "last_mtime, state_changed_at, created_at, updated_at) "
            "VALUES (?, 'active', ?, 0, ?, ?, ?, ?, ?)",
            (rel_path, rel_path, size, mtime, now, now, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def refresh_active(self, file_id: int, size: int, mtime: float, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        self.conn.execute(
            "UPDATE files SET size = ?, last_mtime = ?, updated_at = ? WHERE id = ?",
            (size, mtime, now, file_id),
        )
        self.conn.commit()

    def update_file(self, file_id: int, now: Optional[float] = None, **fields) -> None:
        now = now if now is not None else time.time()
        fields = dict(fields)
        fields["updated_at"] = now
        if "state" in fields:
            fields["state_changed_at"] = now
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(f"UPDATE files SET {cols} WHERE id = ?", (*fields.values(), file_id))
        self.conn.commit()

    # -- actions / journal -------------------------------------------------

    def log_action(
        self,
        file_id: int,
        action: str,
        from_rel_path: Optional[str],
        to_rel_path: Optional[str],
        reason: str,
        now: Optional[float] = None,
    ) -> int:
        now = now if now is not None else time.time()
        cur = self.conn.execute(
            "INSERT INTO actions (file_id, action, from_rel_path, to_rel_path, reason, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (file_id, action, from_rel_path, to_rel_path, reason, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_actions(self, file_id: Optional[int] = None) -> list[sqlite3.Row]:
        if file_id is not None:
            cur = self.conn.execute(
                "SELECT * FROM actions WHERE file_id = ? ORDER BY timestamp", (file_id,)
            )
        else:
            cur = self.conn.execute("SELECT * FROM actions ORDER BY timestamp")
        return cur.fetchall()

    def last_action(self, file_id: int) -> Optional[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM actions WHERE file_id = ? ORDER BY timestamp DESC LIMIT 1",
            (file_id,),
        )
        return cur.fetchone()

    def mark_undone(self, action_id: int) -> None:
        self.conn.execute("UPDATE actions SET undone = 1 WHERE id = ?", (action_id,))
        self.conn.commit()
