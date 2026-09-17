"""Ties scanner + decay rules + actions + store together into a single pass."""
from __future__ import annotations

import dataclasses
import time
from typing import Optional

from filedecay import actions, decay
from filedecay.config import Config
from filedecay.scanner import file_signature, is_pinned, iter_files
from filedecay.store import Store

_TRANSITION_FN = {
    "compressed": actions.compress,
    "archived": actions.archive,
    "quarantined": actions.quarantine,
}


@dataclasses.dataclass
class PlannedAction:
    rel_path: str
    file_id: Optional[int]
    from_state: str
    to_state: str
    idle_days: float
    reason: str


def run_pass(cfg: Config, store: Store, apply: bool, now: Optional[float] = None) -> list[PlannedAction]:
    """Run one scan+decide(+apply) pass. Always safe to call repeatedly;
    with apply=False nothing on disk or in the store is changed."""
    now = now if now is not None else time.time()
    planned: list[PlannedAction] = []
    seen: set[str] = set()

    ignore_patterns = list(cfg.ignore) + [".decay"]

    for rel in iter_files(cfg.root, ignore_patterns):
        if is_pinned(rel, cfg.pinned):
            continue
        rel_str = str(rel)
        seen.add(rel_str)

        abs_path = cfg.root / rel
        size, mtime = file_signature(abs_path)
        row = store.get_file(rel_str)

        if row is None:
            # Newly discovered file. A pre-existing dusty file (or a demo
            # fixture with a backdated mtime) should decay on the very first
            # pass rather than requiring one pass just to start tracking it.
            if apply:
                store.insert_active(rel_str, size=size, mtime=mtime, now=now)
                row = store.get_file(rel_str)
            else:
                idle_days = (now - mtime) / 86400
                next_state = decay.decide_from_active(idle_days, cfg.thresholds)
                if next_state is not None:
                    reason = f"idle {idle_days:.1f}d >= {next_state} threshold"
                    planned.append(PlannedAction(rel_str, None, "active", next_state, idle_days, reason))
                continue
        elif row["state"] != "active":
            # Shouldn't normally happen (the file would have moved), but a
            # user could recreate a file at a path we're already tracking.
            continue
        elif apply:
            store.refresh_active(row["id"], size=size, mtime=mtime, now=now)

        idle_days = (now - mtime) / 86400
        next_state = decay.decide_from_active(idle_days, cfg.thresholds)
        if next_state is not None:
            reason = f"idle {idle_days:.1f}d >= {next_state} threshold"
            planned.append(PlannedAction(rel_str, row["id"], "active", next_state, idle_days, reason))
            if apply:
                fresh_row = store.get_file(rel_str)
                _TRANSITION_FN[next_state](cfg, store, fresh_row, reason)

    # Files already off the beaten (still-active) path: progress them
    # further through the pipeline based on their frozen last-activity time
    # (compressed/archived) or their time spent in quarantine.
    for row in store.list_files(states=("compressed", "archived", "quarantined")):
        if row["state"] == "quarantined":
            days_in_quarantine = (now - row["state_changed_at"]) / 86400
            if decay.should_delete(days_in_quarantine, cfg.thresholds):
                reason = f"{days_in_quarantine:.1f}d in quarantine >= delete threshold"
                planned.append(
                    PlannedAction(row["rel_path"], row["id"], "quarantined", "deleted", days_in_quarantine, reason)
                )
                if apply:
                    actions.delete(cfg, store, row, reason)
        else:
            idle_days = (now - row["last_mtime"]) / 86400
            if decay.should_progress_to_quarantine(idle_days, cfg.thresholds):
                reason = f"idle {idle_days:.1f}d >= quarantine threshold"
                planned.append(
                    PlannedAction(row["rel_path"], row["id"], row["state"], "quarantined", idle_days, reason)
                )
                if apply:
                    actions.quarantine(cfg, store, row, reason)

    # Reconcile files that disappeared from disk outside of our pipeline.
    if apply:
        for row in store.list_files(states=("active",)):
            if row["rel_path"] not in seen:
                store.update_file(row["id"], state="missing")

    return planned
