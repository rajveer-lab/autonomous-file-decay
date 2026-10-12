"""Linear-resource checking: the static form of the no-cloning theorem.

QR-IR treats every qubit as a *linear* resource.  There is no copy
instruction, and the checker enforces the three typing rules of the paper:

* T-Gate   - the operands of one gate are pairwise distinct (a gate cannot
             act on "two copies" of the same qubit);
* T-Use    - a qubit is used only between its ``alloc`` and its ``free``;
* T-Free   - a qubit is freed at most once, and is never freed while a
             second logical qubit occupies the same physical slot.

The same checker runs on logical programs (before allocation) and on
physical programs (after allocation, via ``check_physical``), so a reuse
decision can never alias two live quantum states onto one qubit.
"""

from __future__ import annotations

from .ir import CompileError, Instr, operands


def check_linear(instrs: list[Instr]) -> None:
    live: set = set()
    for i, ins in enumerate(instrs):
        if ins.op == "scope_end":
            continue
        if ins.op == "alloc":
            q = ins.tgts[0]
            if q in live:
                raise CompileError(f"[{i}] `{ins}`: qubit {q} allocated twice")
            live.add(q)
            continue
        seen: set = set()
        for q in operands(ins):
            if q in seen:
                raise CompileError(
                    f"no-cloning violation at [{i}] `{ins}`: qubit {q} is used twice "
                    "by one operation; a gate cannot act on two copies of a qubit")
            seen.add(q)
            if q not in live:
                raise CompileError(f"[{i}] `{ins}`: qubit {q} used outside its lifetime")
        if ins.op == "free":
            live.discard(ins.tgts[0])


def check_physical(instrs: list[Instr], phys: dict) -> None:
    """Every physical qubit hosts at most one live logical qubit at a time."""
    occupant: dict = {}
    for i, ins in enumerate(instrs):
        if ins.op == "alloc":
            q = ins.tgts[0]
            p = phys[q]
            if p in occupant:
                raise CompileError(
                    f"[{i}] physical qubit {p} assigned to {q} while {occupant[p]} is live")
            occupant[p] = q
        elif ins.op == "free":
            occupant.pop(phys[ins.tgts[0]])
        else:
            for q in operands(ins):
                if occupant.get(phys[q]) is not q:
                    raise CompileError(f"[{i}] `{ins}`: {q} does not own physical qubit {phys[q]}")
