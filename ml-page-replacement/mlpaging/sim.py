"""Trace-driven simulator."""

from __future__ import annotations

import numpy as np

from .policies import Policy


def simulate(policy: Policy, trace: np.ndarray, capacity: int) -> dict:
    policy.reset(capacity, trace)
    access = policy.access
    faults = 0
    for t, p in enumerate(trace.tolist()):
        if not access(p, t):
            faults += 1
    n = int(trace.size)
    return {
        "policy": policy.name,
        "capacity": capacity,
        "accesses": n,
        "faults": faults,
        "fault_rate": faults / n,
        "decisions": policy.decisions,
        "decision_us": policy.decision_ns / max(policy.decisions, 1) / 1000,
    }
