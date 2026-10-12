"""Lowering of multi-controlled gates.

An ``x``/``z`` gate with k >= 3 controls is expanded into the Barenco-style
V-chain: k-2 HELPER qubits accumulate the conjunction of the controls, and
the final Toffoli (or CCZ) acts on the target.  Lowering only emits the
*compute* half of the chain; the uncompute half is produced by running the
same reclamation pass again over HELPER qubits.  Each expansion gets its own
scope, so lexical (scope-based) and eager reclamation coincide here.
"""

from __future__ import annotations

from .ir import HELPER, Instr, Program


def lower(instrs: list[Instr], prog: Program) -> list[Instr]:
    out: list[Instr] = []
    count = 0
    for ins in instrs:
        if not (ins.op == "gate" and ins.name in ("x", "z") and len(ins.ctrls) >= 3):
            out.append(ins)
            continue
        sid = prog.new_scope_id()
        cs, ng = ins.ctrls, ins.neg
        prev, prev_neg = cs[0], ng[0]
        for j in range(1, len(cs) - 1):
            h = prog.new_qubit(f"h{count}", HELPER, scope=sid)
            count += 1
            out.append(Instr("alloc", tgts=(h,), tag="lower"))
            out.append(Instr("gate", "x", (h,), (prev, cs[j]), (prev_neg, ng[j]), tag="lower"))
            prev, prev_neg = h, False
        out.append(Instr("gate", ins.name, ins.tgts, (prev, cs[-1]), (prev_neg, ng[-1]),
                         ins.params, tag=ins.tag))
        out.append(Instr("scope_end", sid=sid))
    return out
