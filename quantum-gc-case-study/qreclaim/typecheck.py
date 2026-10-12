"""Linear-resource checking for QR-IR.

QR-IR treats every qubit as a *linear* resource, and the checkers enforce the
typing rules of the paper:

* T-Gate   - the operands of one operation are pairwise distinct (no
             aliasing of one qubit in two roles);
* T-Use    - a qubit is used only between its ``alloc`` and its ``free``;
* T-Free   - a qubit is freed at most once, only in a provably clean state
             (``check_frees_clean``), and never while a second logical
             qubit occupies the same physical slot (``check_physical``).

QR-IR has no copy instruction, so no program can clone a state; what the
checkers guarantee is that the compiler never aliases two roles onto one
qubit and never frees (and so later reuses) a qubit that still carries
information.
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
                    f"aliasing violation at [{i}] `{ins}`: qubit {q} appears twice in one "
                    "operation; QR-IR qubits are linear, so an operation cannot use one "
                    "qubit in two roles")
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


def _ctrl_key(ins: Instr) -> tuple:
    return tuple(sorted((c.uid, n) for c, n in zip(ins.ctrls, ins.neg)))


def check_frees_clean(instrs: list[Instr], kinds: tuple) -> int:
    """Statically discharge T-Free: every freed qubit of the given kinds is |0>.

    Translation validation by XOR cancellation.  Since its last ``alloc`` or
    ``reset`` the qubit a may only be (i) the target of X-type gates
    a ^= AND(controls), (ii) a control, or (iii) acted on by diagonal gates.
    If every control pattern occurs an even number of times and no control of
    a pattern is written between the first and the last gate with that
    pattern, every term is XORed in an even number of times with the same
    control values, so the XORs cancel on every computational-basis branch
    and a is |0> and unentangled.
    Returns the number of frees validated.
    """
    hist: dict = {}
    writes_at: dict = {}
    for i, ins in enumerate(instrs):
        for q in operands(ins):
            hist.setdefault(q, []).append(i)
        for w in ins.writes():
            writes_at.setdefault(w, []).append(i)
    checked = 0
    for i, ins in enumerate(instrs):
        if ins.op != "free" or ins.tgts[0].kind not in kinds:
            continue
        a = ins.tgts[0]
        events = [j for j in hist.get(a, []) if j < i]
        start = max((j for j in events if instrs[j].op in ("alloc", "reset")), default=-1)
        counts: dict = {}
        span: dict = {}                       # pattern -> (first, last, controls)
        for j in events:
            if j <= start:
                continue
            e = instrs[j]
            if e.op == "gate" and a in e.tgts:
                if e.name == "x" and len(e.tgts) == 1:
                    k = _ctrl_key(e)
                    counts[k] = counts.get(k, 0) + 1
                    f0 = span[k][0] if k in span else j
                    span[k] = (f0, j, e.ctrls)
                elif e.category != "diag":
                    raise CompileError(f"T-Free: `{e}` at [{j}] changes {a} irreversibly before it is freed at [{i}]")
            elif e.op == "gate":
                continue                      # a is only a control here
            else:
                raise CompileError(f"T-Free: {a} is {e.op}d at [{j}] but not reset before it is freed at [{i}]")
        if any(v % 2 for v in counts.values()):
            raise CompileError(f"T-Free: the computation of {a} is not cancelled before it is freed at [{i}]")
        for f0, f1, ctrls in span.values():
            for c in ctrls:
                if any(f0 < w < f1 for w in writes_at.get(c, [])):
                    raise CompileError(
                        f"T-Free: control {c} of {a} changes between its computation and uncomputation")
        checked += 1
    return checked
