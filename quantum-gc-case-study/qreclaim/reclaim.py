"""The reclamation pass: QReclaim's compile-time garbage collector.

Given a QR-IR program it decides, for every managed qubit, *whether* and
*where* the qubit can be returned to the free pool, and inserts the code
that makes this safe:

* qfree-computed ancillas (state B) are uncomputed by reversing their trace,
  then freed;
* qubits whose final operation is a measurement are measured as early as
  possible, reset and freed (early-measure reclamation);
* ancillas that are never read are deleted together with their computation
  (dead-garbage elimination, DGE);
* everything else (state T and never measured) cannot be reclaimed without
  violating no-cloning or leaving entangled garbage, and is reported as a
  static error.

Placement uses the *uncomputation-aware liveness* fixpoint

        L*(q) = max( L(q), max{ P(a) : a in R, q in Dep(a) } )
        P(a)  = L*(a)                                         (eager)

and is legal iff P(a) < D(a) for every reclaimed ancilla a.  The lazy
(scope-based) baseline instead places each uncomputation as late as its
scope, its deadline and its dependencies allow.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .analysis import BASIS, MEAS, TOP, ZERO, Analysis
from .ir import ANC, DATA, CompileError, Instr


@dataclass
class ReclaimConfig:
    placement: str = "eager"          # eager | lazy
    early_measure: bool = True
    dge: bool = True
    kinds: tuple = (ANC,)
    strict: bool = True


@dataclass
class ReclaimReport:
    uncomputed: list = field(default_factory=list)
    measured_reset: list = field(default_factory=list)
    freed_clean: list = field(default_factory=list)
    eliminated: list = field(default_factory=list)
    leaks: list = field(default_factory=list)
    gates_inserted: int = 0
    gates_removed: int = 0


def _scope_ends(instrs: list[Instr]) -> dict:
    ends = {0: len(instrs) - 1}
    for i, ins in enumerate(instrs):
        if ins.op == "scope_end":
            ends[ins.sid] = i
    return ends


def reclaim(instrs: list[Instr], cfg: ReclaimConfig) -> tuple[list[Instr], ReclaimReport]:
    A = Analysis(instrs)
    info = A.info
    rep = ReclaimReport()
    managed = [q for q in info if q.kind in cfg.kinds]

    # -- dead-garbage elimination -----------------------------------------
    removed: set = set()
    eliminated: set = set()
    if cfg.dge:
        for a in managed:
            ai = info[a]
            if ai.state in (ZERO, BASIS) and set(ai.uses) == set(ai.trace):
                removed.update(id(instrs[j]) for j in ai.trace)
                removed.add(id(instrs[ai.alloc]))
                eliminated.add(a)
                rep.gates_removed += len(ai.trace)
        rep.eliminated = sorted(eliminated, key=lambda q: q.uid)

    # -- classification ---------------------------------------------------
    to_unc = [a for a in managed if a not in eliminated and info[a].state == BASIS]
    to_free = [a for a in managed if a not in eliminated and info[a].state == ZERO]
    to_meas = []
    if cfg.early_measure:
        for q in info:
            qi = info[q]
            if (q.kind in cfg.kinds or q.kind == DATA) and qi.state == MEAS and qi.uses \
                    and instrs[qi.uses[-1]].op == "measure":
                to_meas.append(q)
    for a in managed:
        st = info[a].state
        if a in eliminated or st in (BASIS, ZERO) or (st == MEAS and a in to_meas):
            continue
        reason = info[a].top_reason if st == TOP else "it is measured but early-measure reclamation is disabled"
        rep.leaks.append((a, reason))
    if rep.leaks and cfg.strict:
        a, why = rep.leaks[0]
        raise CompileError(
            f"ancilla {a} can never be reclaimed: {why}, and it is not measured. "
            "Leaving it allocated would leave garbage entangled with the program state.")

    # -- placement --------------------------------------------------------
    unc_set = set(to_unc)
    dependents = defaultdict(list)          # q -> reclaimed ancillas whose Dep contains q
    for a in to_unc:
        for d in info[a].deps:
            dependents[d].append(a)
    L = {q: A.L(q) for q in info}
    P: dict = {}

    if cfg.placement == "eager":
        # Worklist fixpoint; processing in reverse order of computation is a
        # reverse topological order of the dependency DAG, so it usually
        # converges after a single sweep.
        order = sorted(to_unc, key=lambda a: info[a].trace[0], reverse=True)
        Lstar = dict(L)
        changed = True
        while changed:
            changed = False
            for a in order:
                v = max([L[a]] + [P[b] for b in dependents.get(a, ()) if b in P])
                if P.get(a) != v:
                    P[a] = v
                    changed = True
        for q, ds in dependents.items():
            Lstar[q] = max([L[q]] + [P[b] for b in ds])
    elif cfg.placement == "lazy":
        ends = _scope_ends(instrs)
        stack: set = set()

        def latest(a) -> int:
            if a in P:
                return P[a]
            if a in stack:
                raise CompileError(f"cyclic dependency while uncomputing {a}")
            stack.add(a)
            bound = ends.get(a.scope, len(instrs) - 1)
            if info[a].deadline is not None:
                bound = min(bound, info[a].deadline - 1)
            for d in info[a].deps:
                if d in unc_set:
                    bound = min(bound, latest(d))
            stack.discard(a)
            P[a] = bound
            return bound

        for a in to_unc:
            latest(a)
        Lstar = dict(L)
        for q, ds in dependents.items():
            Lstar[q] = max([L[q]] + [P[b] for b in ds])
    else:
        raise ValueError(cfg.placement)

    # -- legality: L*(a) <= P(a) < D(a) -------------------------------------
    for a in to_unc:
        ai = info[a]
        need = max([L[a]] + [P[b] for b in dependents.get(a, ())])
        if ai.deadline is not None and need >= ai.deadline:
            j, d = ai.deadline_by
            raise CompileError(
                f"cannot uncompute ancilla {a}: its control {d} is modified by "
                f"`{instrs[j]}` at [{j}] while {a} is still needed at [{need}]")
        if P[a] < need:
            raise CompileError(
                f"cannot uncompute ancilla {a} at the end of its scope: it is still "
                f"needed at [{need}] but must be uncomputed by [{P[a]}]")

    # -- build the reclamation blocks ---------------------------------------
    blocks = defaultdict(list)       # position -> [(kind, qubit, payload)]
    moved: set = set()
    for a in to_unc:
        blocks[P[a]].append(("U", a, None))
    for a in to_free:
        blocks[Lstar.get(a, L[a])].append(("F", a, None))
    for q in to_meas:
        uses = info[q].uses
        m = uses[-1]
        before = max(uses[:-1]) if len(uses) > 1 else info[q].alloc
        pos = max([before] + [P[b] for b in dependents.get(q, ())])
        moved.add(m)
        blocks[pos].append(("MR", q, instrs[m].cbit))

    def reads(blk) -> set:
        kind, q, _ = blk
        return set(info[q].deps) if kind == "U" else set()

    def writes(blk) -> set:
        kind, q, _ = blk
        return {q} if kind in ("U", "MR") else set()

    def ordered(blks: list) -> list:
        # A block that reads q must run before the block that writes q.
        # Ties follow reverse allocation order (Bennett-style LIFO).
        blks = sorted(blks, key=lambda b: -b[1].uid)
        out, pending = [], list(blks)
        while pending:
            for b in pending:
                if not any(writes(b) & reads(o) for o in pending if o is not b):
                    out.append(b)
                    pending.remove(b)
                    break
            else:
                raise CompileError("cyclic uncomputation order")
        return out

    def materialize(blk) -> list[Instr]:
        kind, q, payload = blk
        if kind == "U":
            body = [instrs[j].inverse() for j in reversed(info[q].trace)]
            rep.gates_inserted += len(body)
            rep.uncomputed.append(q)
            return body + [Instr("free", tgts=(q,), tag="gc")]
        if kind == "MR":
            rep.measured_reset.append(q)
            return [Instr("measure", tgts=(q,), cbit=payload, tag="gc"),
                    Instr("reset", tgts=(q,), tag="gc"),
                    Instr("free", tgts=(q,), tag="gc")]
        rep.freed_clean.append(q)
        return [Instr("free", tgts=(q,), tag="gc")]

    out: list[Instr] = []
    for i, ins in enumerate(instrs):
        if id(ins) not in removed and i not in moved:
            out.append(ins)
        for blk in ordered(blocks.get(i, [])):
            out.extend(materialize(blk))
    return out, rep
