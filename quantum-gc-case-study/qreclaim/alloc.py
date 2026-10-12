"""Linear-scan qubit allocation.

Logical qubits are assigned to physical qubits in one pass over the
(scheduled) instruction list.  Because the program is a single basic block,
live ranges are intervals and the interference graph is an interval graph,
so greedy linear scan is optimal: it uses exactly max_p |LIVE(p)| physical
qubits for the given order.

Reuse policies for choosing among free physical qubits:

* ``none`` - never reuse (no garbage collection);
* ``lifo`` - most recently freed first (a stack-based qubit manager);
* ``fifo`` - least recently freed first;
* ``erf``  - earliest-ready-first: the free qubit whose last operation
             finishes earliest in the ASAP schedule, which minimises the
             false dependency a reuse introduces.

With a qubit *budget* B the allocator uses fresh qubits while fewer than B
are in use and only reuses (ERF) when the budget is reached, trading qubits
for depth.  Resets that precede no further reuse are removed afterwards
(dead-reset elimination).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from .ir import CompileError, Instr, operands


class BudgetError(CompileError):
    pass


@dataclass
class Allocation:
    phys: dict
    nqubits: int
    reuses: int
    asap_depth: int


def allocate(instrs: list[Instr], policy: str = "erf", budget: int | None = None) -> Allocation:
    phys: dict = {}
    ready: list[int] = []
    # ERF keeps the pool in a min-heap: a pooled qubit has no operations, so
    # its ready time cannot change while it waits there.
    heap: list = []
    pool: list[int] = []
    reuses = 0

    def pooled() -> bool:
        return bool(heap) if policy == "erf" else bool(pool)

    def take() -> int:
        if policy == "erf":
            return heapq.heappop(heap)[1]
        return pool.pop() if policy == "lifo" else pool.pop(0)

    for ins in instrs:
        if ins.op == "alloc":
            q = ins.tgts[0]
            if budget is not None:
                take_pool = len(ready) >= budget
            else:
                take_pool = policy != "none" and pooled()
            if take_pool:
                if not pooled():
                    raise BudgetError(f"qubit budget {budget} too small at allocation of {q}")
                p = take()
                reuses += 1
            else:
                p = len(ready)
                ready.append(0)
            phys[q] = p
        elif ins.op == "free":
            p = phys[ins.tgts[0]]
            if policy == "erf":
                heapq.heappush(heap, (ready[p], p))
            else:
                pool.append(p)
        elif ins.op != "scope_end":
            ps = [phys[q] for q in operands(ins)]
            t = max(ready[p] for p in ps) + 1
            for p in ps:
                ready[p] = t
    return Allocation(phys, len(ready), reuses, max(ready, default=0))


def drop_dead_resets(instrs: list[Instr], phys: dict) -> list[Instr]:
    """Remove resets of physical qubits that are never allocated again.

    The matching ``free`` is dropped too: such a qubit simply stays occupied
    (in its measured state) until the end of the program.
    """
    reused_later: set = set()
    free_at: dict = {}
    keep = [True] * len(instrs)
    for i in range(len(instrs) - 1, -1, -1):
        ins = instrs[i]
        if ins.op == "alloc":
            reused_later.add(phys[ins.tgts[0]])
        elif ins.op == "free":
            free_at[ins.tgts[0]] = i
        elif ins.op == "reset" and ins.tag == "gc" and phys[ins.tgts[0]] not in reused_later:
            keep[i] = False
            keep[free_at[ins.tgts[0]]] = False
    return [ins for ins, k in zip(instrs, keep) if k]
