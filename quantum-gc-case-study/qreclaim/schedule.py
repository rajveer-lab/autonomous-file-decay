"""Pressure-aware list scheduling (PALS).

Gates on disjoint qubits commute, so any topological order of the per-qubit
precedence DAG is semantically equivalent to the input.  PALS picks the
order that keeps the number of simultaneously live qubits low:

    priority 0 - ``free``         (closes a live range)
    priority 1 - gates, measure, reset on live qubits (original order)
    priority 2 - ``alloc``        (opens a live range; deferred, ordered by
                                   the position of the qubit's first use)

This sinks allocations towards their first use and hoists frees towards
the last use, which is what turns e.g. a Bernstein-Vazirani circuit into
the qubit-sequential form that needs only two physical qubits.
"""

from __future__ import annotations

import heapq

from .ir import Instr, operands, strip_markers


def schedule_pressure(instrs: list[Instr]) -> list[Instr]:
    ops = strip_markers(instrs)
    n = len(ops)
    preds = [0] * n
    succs: list[list[int]] = [[] for _ in range(n)]
    last: dict = {}
    first_use: dict = {}
    for i, ins in enumerate(ops):
        for q in operands(ins):
            if q in last:
                succs[last[q]].append(i)
                preds[i] += 1
            last[q] = i
            if ins.op != "alloc":
                first_use.setdefault(q, i)

    def prio(i: int) -> tuple:
        ins = ops[i]
        if ins.op == "free":
            return (0, i)
        if ins.op == "alloc":
            return (2, first_use.get(ins.tgts[0], i), i)
        return (1, i)

    heap = [(prio(i), i) for i in range(n) if preds[i] == 0]
    heapq.heapify(heap)
    out: list[Instr] = []
    while heap:
        _, i = heapq.heappop(heap)
        out.append(ops[i])
        for s in succs[i]:
            preds[s] -= 1
            if preds[s] == 0:
                heapq.heappush(heap, (prio(s), s))
    assert len(out) == n, "dependency cycle in instruction DAG"
    return out
