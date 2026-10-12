"""The QReclaim compilation pipeline and the configurations compared in the paper.

    front end (Program)
      -> [1] linear type check (no-cloning)
      -> [2] analysis + reclamation of ANC qubits      (level 1, high-level IR)
      -> [3] lowering of multi-controlled gates
      -> [4] analysis + reclamation of HELPER qubits   (level 2, lowered IR)
      -> [5] pressure-aware scheduling (optional)
      -> [6] linear type check again, linear-scan allocation, dead-reset elimination
      -> [7] physical no-aliasing check, code generation (Qiskit / OpenQASM 2)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from qiskit import transpile

from .alloc import allocate, drop_dead_resets
from .codegen import to_qiskit
from .ir import ANC, HELPER, Program, strip_markers
from .lower import lower
from .reclaim import ReclaimConfig, reclaim
from .schedule import schedule_pressure
from .typecheck import check_linear, check_physical


@dataclass(frozen=True)
class Config:
    name: str
    placement: str = "eager"
    early_measure: bool = True
    dge: bool = True
    schedule: bool = True
    policy: str = "erf"
    budget: int | None = None
    two_pass: bool = True        # ERF: find the minimum width, then re-allocate fresh-first under it


NR = Config("NR", placement="lazy", early_measure=False, dge=False, schedule=False, policy="none",
            two_pass=False)
SCOPE = Config("SCOPE", placement="lazy", early_measure=False, dge=False, schedule=False,
               policy="lifo", two_pass=False)
QR = Config("QReclaim")

ABLATION = [
    SCOPE,
    Config("+eager", placement="eager", early_measure=False, dge=False, schedule=False,
           policy="lifo", two_pass=False),
    Config("+early-meas", placement="eager", early_measure=True, dge=False, schedule=False,
           policy="lifo", two_pass=False),
    Config("+PALS", placement="eager", early_measure=True, dge=False, schedule=True,
           policy="lifo", two_pass=False),
    Config("+DGE", placement="eager", early_measure=True, dge=True, schedule=True,
           policy="lifo", two_pass=False),
    Config("+ERF", placement="eager", early_measure=True, dge=True, schedule=True,
           policy="erf", two_pass=False),
    Config("+two-pass", placement="eager", early_measure=True, dge=True, schedule=True,
           policy="erf", two_pass=True),
]


@dataclass
class Compiled:
    config: Config
    program: Program
    instrs: list
    phys: dict
    nqubits: int
    reuses: int
    compile_time: float
    pass_times: dict
    report1: object
    report2: object
    logical_qubits: int
    qc: object = None
    metrics: dict = field(default_factory=dict)


def compile_program(prog: Program, cfg: Config) -> Compiled:
    times: dict = {}
    t0 = time.perf_counter()

    def mark(name: str, start: float) -> float:
        now = time.perf_counter()
        times[name] = times.get(name, 0.0) + now - start
        return now

    t = time.perf_counter()
    check_linear(prog.instrs)
    t = mark("typecheck", t)
    l1, r1 = reclaim(prog.instrs, ReclaimConfig(cfg.placement, cfg.early_measure, cfg.dge, (ANC,)))
    t = mark("reclaim", t)
    l2 = lower(l1, prog)
    t = mark("lower", t)
    l3, r2 = reclaim(l2, ReclaimConfig(cfg.placement, False, False, (HELPER,)))
    t = mark("reclaim", t)
    l4 = schedule_pressure(l3) if cfg.schedule else strip_markers(l3)
    t = mark("schedule", t)
    check_linear(l4)
    t = mark("typecheck", t)
    if cfg.two_pass and cfg.budget is None:
        # Linear scan is optimal for width, so pass 1 gives the minimum W*;
        # pass 2 spreads work over W* qubits (fresh first, then ERF), which
        # never fails because at most W* logical qubits are ever live.
        width = allocate(l4, cfg.policy).nqubits
        al = allocate(l4, cfg.policy, width)
    else:
        al = allocate(l4, cfg.policy, cfg.budget)
    final = drop_dead_resets(l4, al.phys)
    t = mark("allocate", t)
    check_physical(final, al.phys)
    mark("typecheck", t)
    total = time.perf_counter() - t0
    nlog = sum(1 for i in final if i.op == "alloc")
    return Compiled(cfg, prog, final, al.phys, al.nqubits, al.reuses, total, times, r1, r2, nlog)


def with_budget(cfg: Config, budget: int) -> Config:
    return replace(cfg, name=f"{cfg.name}@{budget}", budget=budget)


def measure_circuit(c: Compiled, seed: int = 11) -> dict:
    """Lower to {CX, U} with Qiskit (optimization level 1) and collect metrics."""
    qc = to_qiskit(c.instrs, c.phys, c.nqubits, c.program.ncbits)
    c.qc = qc
    t0 = time.perf_counter()
    tq = transpile(qc, basis_gates=["cx", "u", "measure", "reset"], optimization_level=1,
                   seed_transpiler=seed)
    tt = time.perf_counter() - t0
    ops = tq.count_ops()
    m = {
        "qubits": c.nqubits,
        "cx": int(ops.get("cx", 0)),
        "u": int(ops.get("u", 0)),
        "gates": int(ops.get("cx", 0) + ops.get("u", 0)),
        "depth": int(tq.depth()),
        "measure": int(ops.get("measure", 0)),
        "reset": int(ops.get("reset", 0)),
        "compile_ms": 1e3 * c.compile_time,
        "qiskit_ms": 1e3 * tt,
        "logical_qubits": c.logical_qubits,
    }
    c.metrics = m
    return m


def compile_depth_neutral(build, ref_depth: int, lo: int, hi: int, base: Config = QR) -> Compiled:
    """Smallest qubit budget (found by bisection) whose circuit is no deeper than ref_depth."""
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        c = compile_program(build(), with_budget(base, mid))
        if measure_circuit(c)["depth"] <= ref_depth:
            best, hi = c, mid - 1
        else:
            lo = mid + 1
    if best is None:
        best = compile_program(build(), with_budget(base, hi + 1))
        measure_circuit(best)
    return best
