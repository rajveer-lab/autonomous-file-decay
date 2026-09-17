"""Pure decision logic: given how idle a file is, what stage should it be in?

Kept dependency-free and side-effect-free so it's trivial to unit test.
"""
from __future__ import annotations

from typing import Optional

from filedecay.config import Thresholds

# active -> compressed -> archived -> quarantined -> deleted
STAGES = ("active", "compressed", "archived", "quarantined", "deleted")


def decide_from_active(idle_days: float, t: Thresholds) -> Optional[str]:
    """A still-active file's next stage, or None if it should stay active.

    Checked from most-severe to least: a file that hasn't been scanned in a
    long time (idle_days already past several thresholds) jumps straight to
    the deepest applicable stage rather than crawling through every stage
    one scan at a time.
    """
    if idle_days >= t.quarantine_after_days:
        return "quarantined"
    if idle_days >= t.archive_after_days:
        return "archived"
    if idle_days >= t.compress_after_days:
        return "compressed"
    return None


def should_delete(days_in_quarantine: float, t: Thresholds) -> bool:
    return days_in_quarantine >= t.delete_after_days


def should_progress_to_quarantine(idle_days: float, t: Thresholds) -> bool:
    """For files already compressed/archived: have they idled long enough
    (measured from their original last activity) to be quarantined too?"""
    return idle_days >= t.quarantine_after_days
