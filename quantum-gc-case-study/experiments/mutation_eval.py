"""Evaluate QReclaim's static checks by bug injection.

Each mutation operator injects one fault into a correct benchmark program.
QReclaim either accepts the mutant (and compiles it) or rejects it with a
static diagnostic.  Ground truth comes from exact simulation:

* ALIAS mutants (an operation that names one qubit twice) are ill-formed and
  therefore always faulty.
* For the other operators an ancilla a is *reclaimable* if some
  insertion point p after its last use exists at which reversing its X-type
  trace returns a to |0> for EVERY computational-basis input.  We purify the
  input: each data qubit starts maximally entangled with a fresh reference
  qubit, so one exact simulation covers all 2^n inputs.  We try every p.  A
  mutant is faulty iff at least one of its ancillas (not only the one the
  operator targeted) is not reclaimable on its own.  (A static analysis is input-
  agnostic, so a mutant that is safe only for the hard-coded input of a
  benchmark still counts as faulty.)

A rejected faulty mutant is a true positive; a rejected reclaimable mutant is
a false alarm (the analysis is conservative); an accepted faulty mutant would
be a soundness bug.  Accepted mutants are additionally compiled and simulated
with the run-time GC assertions.

    python experiments/mutation_eval.py      # writes results/mutation.csv
"""

from __future__ import annotations

import copy
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qreclaim import QR, CompileError, compile_program                       # noqa: E402
from qreclaim.analysis import Analysis                                       # noqa: E402
from qreclaim.benchmarks import (adder, bv, grover_sat, hidden_shift,        # noqa: E402
                                 qaoa_ring, repetition_code)
from qreclaim.ir import ANC, Instr, operands                                 # noqa: E402
from qreclaim.sim import GCAssertionError, simulate                          # noqa: E402

SOURCES = [hidden_shift(3), qaoa_ring(4), adder(3), grover_sat(4, 4), bv(6), repetition_code(3, 2)]


def with_instrs(prog, instrs):
    p = copy.copy(prog)
    p.instrs = instrs
    p.qubits = list(prog.qubits)
    return p


def insert(instrs, k, new):
    return instrs[:k] + new + instrs[k:]


def mutants(prog):
    """Yield (operator, ancilla or None, mutated instruction list)."""
    instrs = prog.instrs
    for i, ins in enumerate(instrs):
        if ins.op == "gate" and ins.ctrls and len(ins.tgts) == 1:
            m = copy.copy(ins)
            m.tgts = (ins.ctrls[0],)
            yield "alias", None, instrs[:i] + [m] + instrs[i + 1:]
    A = Analysis(instrs)
    for a, ai in A.info.items():
        if a.kind != ANC or not ai.trace:
            continue
        first, last_use = ai.trace[0], A.L(a)
        deps = sorted(ai.deps, key=lambda q: q.uid)
        for d in deps:
            # clobber a control: right after the computation, just before the
            # last use, and just after the last use (the last one is benign)
            for label, k in (("early", first + 1), ("late", last_use), ("after", last_use + 1)):
                yield f"clobber-{label}", a, insert(instrs, k, [Instr("gate", "x", (d,))])
            # flip and restore the control before the last use (benign)
            if last_use > first + 1:
                flip = Instr("gate", "x", (d,))
                yield "double-flip", a, insert(insert(instrs, last_use, [flip]), first + 1, [copy.copy(flip)])
            yield "reset-control", a, insert(instrs, first + 1, [Instr("reset", tgts=(d,))])
        yield "superpose", a, insert(instrs, ai.trace[-1] + 1, [Instr("gate", "h", (a,))])


def reclaimable(prog, instrs, a) -> bool:
    """Ground truth: does some insertion point clean a, for every basis input?"""
    from qreclaim.ir import DATA, Qubit
    data = [ins.tgts[0] for ins in instrs if ins.op == "alloc" and ins.tgts[0].kind == DATA]
    refs = [Qubit(10**6 + i, f"ref{i}", "ref") for i in range(len(data))]
    purify = []
    for r, d in zip(refs, data):
        purify += [Instr("alloc", tgts=(r,)), Instr("gate", "h", (r,)), Instr("gate", "x", (d,), (r,), (False,))]
    allocs = [i for i, ins in enumerate(instrs) if ins.op == "alloc" and ins.tgts[0].kind == DATA]
    head = allocs[-1] + 1 if allocs else 0
    # measurements do not change basis values, so they are dropped for speed
    body = [ins for ins in instrs if ins.op != "measure"]
    shift = sum(1 for ins in instrs[:head] if ins.op == "measure")
    base = body[:head - shift] + purify + body[head - shift:]
    qubits = [ins.tgts[0] for ins in base if ins.op == "alloc"]
    phys = {q: i for i, q in enumerate(qubits)}
    uses = [i for i, ins in enumerate(base) if a in operands(ins) and ins.op != "alloc"]
    for p in range(max(uses), len(base)):
        # the trace starts after the last reset of a (a reset leaves it clean)
        resets = [i for i, ins in enumerate(base[:p + 1]) if ins.op == "reset" and ins.tgts == (a,)]
        since = resets[-1] + 1 if resets else 0
        trace = [ins for ins in base[since:p + 1]
                 if ins.op == "gate" and ins.name == "x" and ins.tgts == (a,)]
        undo = [t.inverse() for t in reversed(trace)] + [Instr("free", tgts=(a,))]
        try:
            simulate(base[:p + 1] + undo + base[p + 1:], phys, len(qubits), prog.ncbits)
            return True
        except GCAssertionError:
            continue
    return False


def main() -> None:
    rows = []
    for b in SOURCES:
        prog = b.build()
        compile_program(b.build(), QR)                  # the original is accepted
        for op, a, instrs in mutants(prog):
            mp = with_instrs(prog, instrs)
            try:
                c = compile_program(mp, QR)
                verdict, msg = "accepted", ""
            except CompileError as e:
                verdict, msg = "rejected", str(e).split(":")[0]
            if op == "alias":
                faulty = True
            else:
                ancs = [ins.tgts[0] for ins in instrs if ins.op == "alloc" and ins.tgts[0].kind == ANC]
                # the targeted ancilla first: it is the one most likely to fail
                ancs.sort(key=lambda q: q is not a)
                faulty = any(not reclaimable(prog, instrs, q) for q in ancs)
            sound = ""
            if verdict == "accepted" and c.nqubits <= 20:
                try:
                    simulate(c.instrs, c.phys, c.nqubits, c.program.ncbits)
                    sound = "gc-ok"
                except GCAssertionError:
                    sound = "GC-FAIL"
            rows.append(dict(bench=b.name, params=b.params, operator=op, ancilla=str(a or ""),
                             faulty=int(faulty), verdict=verdict, diagnostic=msg, runtime_check=sound))
        print(f"{b.name} {b.params}: {sum(1 for r in rows if r['bench'] == b.name)} mutants", flush=True)
    out = ROOT / "results" / "mutation.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ops = sorted({r["operator"] for r in rows})
    for op in ops + ["ALL"]:
        rs = [r for r in rows if op == "ALL" or r["operator"] == op]
        tp = sum(1 for r in rs if r["faulty"] and r["verdict"] == "rejected")
        fn = sum(1 for r in rs if r["faulty"] and r["verdict"] == "accepted")
        fp = sum(1 for r in rs if not r["faulty"] and r["verdict"] == "rejected")
        tn = sum(1 for r in rs if not r["faulty"] and r["verdict"] == "accepted")
        bad = sum(1 for r in rs if r["runtime_check"] == "GC-FAIL")
        print(f"{op:14s} n={len(rs):4d} faulty={tp + fn:4d} TP={tp:4d} FN={fn:3d} FP={fp:3d} TN={tn:4d} gc-fail={bad}")


if __name__ == "__main__":
    main()
