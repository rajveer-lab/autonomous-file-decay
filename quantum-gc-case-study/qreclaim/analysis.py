"""Static analyses on QR-IR.

Forward pass - *basis-state abstract interpretation*.  Each qubit carries an
abstract state from the lattice

        0  (clean |0>)   <=   B  (basis value = f(deps), qfree-computed)
                         <=   T  (top: possibly in superposition)
        M  (measured; collapsed, becomes 0 again after a reset)

together with its *trace* (the qfree gates that computed it), its
*dependency set* Dep(a) (the controls those gates read) and its *deadline*
D(a): the first later instruction that changes the basis value of some
d in Dep(a).  Reversing the trace is a sound uncomputation for ``a`` at any
point p with L*(a) <= p < D(a) (Theorem 1 in the paper).

Backward pass - classical live-variable analysis, specialised to qubits:

        LIVE_in(i)  = USE(i)  U  (LIVE_out(i) - DEF(i))
        LIVE_out(i) = LIVE_in(i+1)

with DEF(i) = {q} for ``alloc q`` and USE(i) = operands(i).  After inlining
and unrolling, a quantum program is one basic block, so the system is
solved by a single backward sweep; it yields the last use L(q) of every
qubit and the register pressure |LIVE| at every program point.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .ir import Instr, operands

ZERO, BASIS, MEAS, TOP, FREED = "0", "B", "M", "T", "freed"


@dataclass
class QInfo:
    alloc: int = -1
    state: str = ZERO
    trace: list = field(default_factory=list)       # indices of qfree gates computing q
    deps: dict = field(default_factory=dict)        # dep qubit -> index it became a dep
    deadline: int | None = None                     # first write to a dep
    deadline_by: tuple | None = None                # (index, dep qubit) for diagnostics
    uses: list = field(default_factory=list)        # all non-alloc/free uses
    top_reason: str = ""                            # why reversal became impossible


class Analysis:
    def __init__(self, instrs: list[Instr]):
        self.instrs = instrs
        self.info: dict = {}
        self._forward()
        self._backward()

    # ------------------------------------------------------------------
    def _forward(self) -> None:
        info = self.info
        dependents: dict = defaultdict(set)   # d -> ancillas whose Dep contains d
        for i, ins in enumerate(self.instrs):
            if ins.op == "scope_end":
                continue
            if ins.op == "alloc":
                info[ins.tgts[0]] = QInfo(alloc=i)
                continue
            if ins.op == "free":
                info[ins.tgts[0]].state = FREED
                continue
            for q in operands(ins):
                info[q].uses.append(i)
            # A write to d ends the uncomputation window of every ancilla
            # that read d.  Only the first write matters, so the set is cleared.
            for w in ins.writes():
                for a in dependents.pop(w, ()):
                    if info[a].deadline is None:
                        info[a].deadline = i
                        info[a].deadline_by = (i, w)
            if ins.op == "gate":
                cat = ins.category
                if cat == "qfree" and len(ins.tgts) == 1:
                    t = ins.tgts[0]
                    ti = info[t]
                    if ti.state in (ZERO, BASIS) and ti.deadline is None:
                        ti.state = BASIS
                        ti.trace.append(i)
                        for c in ins.ctrls:
                            ti.deps.setdefault(c, i)
                            dependents[c].add(t)
                    else:
                        if not ti.top_reason:
                            ti.top_reason = (f"`{ins}` at [{i}] updates {t} after its "
                                             f"state became {ti.state}" if ti.state != BASIS else
                                             f"`{ins}` at [{i}] updates {t} after a control it "
                                             f"depends on was modified")
                        ti.state = TOP
                elif cat == "diag":
                    pass
                else:
                    for t in ins.tgts:
                        if not info[t].top_reason:
                            info[t].top_reason = f"`{ins}` at [{i}] puts {t} in superposition"
                        info[t].state = TOP
            elif ins.op == "measure":
                info[ins.tgts[0]].state = MEAS
            elif ins.op == "reset":
                q = ins.tgts[0]
                qi = info[q]
                qi.state = ZERO
                qi.trace.clear()
                for d in qi.deps:
                    dependents[d].discard(q)
                qi.deps.clear()
                qi.deadline = None
                qi.deadline_by = None

    # ------------------------------------------------------------------
    def _backward(self) -> None:
        live: set = set()
        self.last_use: dict = {}
        self.pressure = [0] * len(self.instrs)
        for i in range(len(self.instrs) - 1, -1, -1):
            ins = self.instrs[i]
            if ins.op in ("gate", "measure", "reset"):
                for q in operands(ins):                 # USE(i)
                    if q not in live:
                        live.add(q)
                        self.last_use[q] = i
            elif ins.op == "alloc":                      # DEF(i)
                q = ins.tgts[0]
                live.discard(q)
                self.last_use.setdefault(q, i)
            self.pressure[i] = len(live)

    def L(self, q) -> int:
        """Last use of q (its allocation point if it is never used)."""
        return self.last_use.get(q, self.info[q].alloc)
