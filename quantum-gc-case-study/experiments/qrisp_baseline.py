#!/usr/bin/env python3
"""Qrisp as an external baseline for the QReclaim qubit-memory-management study.

The HS, QAOA (ring), GRV and ADD programs of ``qreclaim/benchmarks.py`` are
re-implemented in idiomatic Qrisp: temporaries are ``QuantumBool``s that are
never freed by hand; the functions that create them are decorated with
``@auto_uncompute`` so Qrisp synthesises the uncomputation, and
``QuantumSession.compile()`` performs Qrisp's qubit allocation and reuse.

Each program is compiled in two configurations:

* ``Qrisp``      ``qs.compile()``               (all defaults, workspace=0)
* ``Qrisp+ws2``  ``qs.compile(workspace=2)``    (two extra qubits granted to cut depth)

and measured exactly as ``qreclaim.pipeline.measure_circuit`` does:
physical qubits = circuit width, then
``transpile(qc, basis_gates=['cx','u','measure','reset'], optimization_level=1,
seed_transpiler=11)`` for the CX count and depth.

``compile_ms`` times ``qs.compile(...)`` and nothing else.  It is the median over
REPS freshly built sessions, taken after one untimed warm-up build and compile of
the same program and configuration (the first Qrisp compile of a new program
shape pays one-off caching costs of up to seconds; that value is kept in
``compile_cold_ms``).  Qrisp runs its uncomputation synthesis (``auto_uncompute``)
while the program is being built, so that cost is in ``build_ms``, not ``compile_ms``.

Correctness is checked on every row, on the transpiled circuit whose metrics are
reported (final measurements removed):

* width <= SV_MAX: exact Aer statevector.  All non-data qubits must be |0> with
  probability 1, the data register must have fidelity 1 with an independent reference
  state, and the distribution of the measured qubits must match the benchmark's own
  ``expected()`` distribution (total variation distance ~ 0).
* wider circuits (HS n=16): Aer matrix-product-state simulation after relabelling the
  qubits so that interacting pairs are adjacent; the amplitude of the single expected
  basis state (data = reference, ancillas = 0) must have modulus 1.

References: HS = the shift string with probability 1 (``hidden_shift()``);
QAOA = Qiskit statevector of H-all, CP(gamma) on every ring edge, RX(2 beta)-all;
GRV = an exact NumPy Grover iteration over the planted 3-SAT predicate;
ADD = the basis state a = a_val, b = (a_val + b_val) mod 2^n.

Usage, from a virtualenv with qrisp, qiskit and qiskit-aer installed:

    python experiments/qrisp_baseline.py [--out results/qrisp.csv] [--reps 5]

Only ``qreclaim.benchmarks`` (and the ``qreclaim.ir`` module it imports) is loaded from
the project.  ``qreclaim/__init__.py`` is not executed, and no bytecode is written.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata as md
import math
import os
import platform
import random
import statistics
import sys
import time
import types

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load_benchmarks():
    """Import qreclaim.benchmarks without running qreclaim/__init__.py (which pulls in the
    whole compiler); only the planted 3-SAT generator and expected() functions are needed."""
    if "qreclaim" not in sys.modules:
        pkg = types.ModuleType("qreclaim")
        pkg.__path__ = [os.path.join(ROOT, "qreclaim")]
        sys.modules["qreclaim"] = pkg
    return importlib.import_module("qreclaim.benchmarks")


bm = _load_benchmarks()

import numpy as np  # noqa: E402
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qrisp import (QuantumBool, QuantumVariable, auto_uncompute, cx, h, mcx, mcz,  # noqa: E402
                   measure, p, rx, x, z)

SV_MAX = 24          # widest circuit simulated with a dense statevector
TOL = 1e-9
CONFIGS = [("Qrisp", 0), ("Qrisp+ws2", 2)]


# --------------------------------------------------------------------------
# Benchmark programs in Qrisp
# --------------------------------------------------------------------------
def qubits_of(qv: QuantumVariable) -> list:
    return [qv[i] for i in range(qv.size)]


def build_hs(n: int, shift: list[int]):
    """HS(n): H, X^shift, O, X^shift, H, O, H, measure, where O = (-1)^{x.y}
    computed through one temporary t = x_i AND y_i per product (cf. hidden_shift())."""

    @auto_uncompute
    def ip_oracle(xv, yv):
        for i in range(n):
            t = QuantumBool()
            mcx([xv[i], yv[i]], t)
            z(t)
        # t's are local and not returned: auto_uncompute uncomputes them here

    xv = QuantumVariable(n, name="x")
    yv = QuantumVariable(n, name="y")
    zq = qubits_of(xv) + qubits_of(yv)
    h(xv); h(yv)
    for q, b in zip(zq, shift):
        if b:
            x(q)
    ip_oracle(xv, yv)
    for q, b in zip(zq, shift):
        if b:
            x(q)
    h(xv); h(yv)
    ip_oracle(xv, yv)            # the inner product is self-dual
    h(xv); h(yv)
    measure(xv); measure(yv)
    return xv.qs


def build_qaoa(n: int, gamma: float, beta: float):
    """QAOA-ring(n), p=1: H-all; for every ring edge (i, i+1 mod n) a temporary
    t = x_i AND x_j receives P(gamma); RX(2 beta)-all; measure (cf. qaoa_ring())."""

    @auto_uncompute
    def phase_separator(xv):
        for i in range(n):
            j = (i + 1) % n
            t = QuantumBool()
            mcx([xv[i], xv[j]], t)
            p(gamma, t)

    xv = QuantumVariable(n, name="x")
    h(xv)
    phase_separator(xv)
    rx(2 * beta, xv)
    measure(xv)
    return xv.qs


def build_grv(n: int, clauses: list, k_it: int):
    """GRV(n, m): Grover over a planted 3-SAT instance, one temporary per clause
    c = l1 OR l2 OR l3 (X then a negatively controlled MCX), phase flip by an MCZ over
    all clause temporaries, standard diffusion (cf. grover_sat())."""

    @auto_uncompute
    def oracle(xv):
        cs = []
        for lits in clauses:
            c = QuantumBool()
            x(c)
            # flip back to 0 iff every literal is false: control on x_v = 0 for a
            # positive literal and x_v = 1 for a negated one (char i <-> control i)
            mcx([xv[v] for v, _ in lits], c,
                ctrl_state="".join("0" if pos else "1" for _, pos in lits))
            cs.append(c[0])
        mcz(cs)

    def diffusion(xv):
        h(xv); x(xv)
        mcz(xv)
        x(xv); h(xv)

    xv = QuantumVariable(n, name="x")
    h(xv)
    for _ in range(k_it):
        oracle(xv)
        diffusion(xv)
    measure(xv)
    return xv.qs


def build_add(n: int, a_val: int, b_val: int):
    """ADD(n): carries c[i+1] = MAJ(a_i, b_i, c_i) in temporaries, sums written
    top-down into b, carry-out unused, measure b (cf. adder())."""
    a = QuantumVariable(n, name="a")
    b = QuantumVariable(n, name="b")
    for i in range(n):
        if (a_val >> i) & 1:
            x(a[i])
        if (b_val >> i) & 1:
            x(b[i])

    @auto_uncompute
    def add_into(a, b):
        c = [None] * (n + 1)
        for i in range(n):
            c[i + 1] = QuantumBool()
            mcx([a[i], b[i]], c[i + 1])
            if i > 0:
                mcx([a[i], c[i][0]], c[i + 1])
                mcx([b[i], c[i][0]], c[i + 1])
        for i in reversed(range(n)):
            cx(a[i], b[i])
            if i > 0:
                cx(c[i][0], b[i])

    add_into(a, b)
    measure(b)
    return a.qs


# --------------------------------------------------------------------------
# Specifications: build function + independent references
# --------------------------------------------------------------------------
class Spec:
    def __init__(self, bench, params, size, build, data, measured, ref_state, expected,
                 mps_order=None, basis_index=None):
        self.bench, self.params, self.size = bench, params, size
        self.build = build                 # () -> QuantumSession (fresh each call)
        self.data = data                   # qubit names, bit j of ref_state <-> data[j]
        self.measured = measured           # names of measured qubits, in key order
        self.ref_state = ref_state         # np.ndarray over the data qubits (None: basis_index)
        self.expected = expected           # benchmark's own expected() distribution
        self.mps_order = mps_order         # locality-friendly order for the MPS fallback
        self.basis_index = basis_index     # set when the reference is a basis state


def _key(bits) -> str:
    return "".join(str(int(b)) for b in bits)


def spec_hs(n: int, seed: int = 2) -> Spec:
    rng = random.Random(seed * 104729 + n)            # same formula as hidden_shift()
    shift = [rng.randint(0, 1) for _ in range(2 * n)]
    expected = bm.hidden_shift(n).expected()
    assert expected == {_key(shift): 1.0}, "shift formula drifted from hidden_shift()"
    data = [f"x.{i}" for i in range(n)] + [f"y.{i}" for i in range(n)]
    idx = sum(b << j for j, b in enumerate(shift))
    order = [nm for i in range(n) for nm in (f"x.{i}", f"y.{i}")]
    return Spec("HS", f"n={n}", n, lambda: build_hs(n, shift), data, data,
                None, expected, mps_order=order, basis_index=idx)


def spec_qaoa(n: int, gamma: float = 0.7, beta: float = 0.4) -> Spec:
    ref = QuantumCircuit(n)                           # independent, ancilla-free reference
    ref.h(range(n))
    for i in range(n):
        ref.cp(gamma, i, (i + 1) % n)
    for q in range(n):
        ref.rx(2 * beta, q)
    sv = Statevector(ref).data                        # bit j of the index = x_j
    probs = np.abs(sv) ** 2
    expected = {_key([(k >> j) & 1 for j in range(n)]): float(pk)
                for k, pk in enumerate(probs) if pk > 1e-14}
    if hasattr(bm, "qaoa_ring"):                      # cross-check against the benchmark
        bexp = bm.qaoa_ring(n, 1, gamma, beta).expected()
        assert _tvd(bexp, expected) < 1e-9, "QAOA reference disagrees with qaoa_ring()"
    data = [f"x.{i}" for i in range(n)]
    return Spec("QAOA", f"n={n}", n, lambda: build_qaoa(n, gamma, beta), data, data, sv,
                expected)


def spec_grv(n: int, seed: int = 5) -> Spec:
    m = 2 * n
    _, clauses = bm.planted_3sat(n, m, seed * 1009 + n * 13 + m)   # as in grover_sat()
    N = 1 << n
    sat = np.array([bm._sat([(k >> i) & 1 for i in range(n)], clauses) for k in range(N)])
    M = int(sat.sum())
    theta = math.asin(math.sqrt(M / N))
    k_it = max(1, int(math.floor(math.pi / (4 * theta))))
    psi = np.full(N, 1 / math.sqrt(N), dtype=complex)                # independent NumPy Grover
    s = psi.copy()
    for _ in range(k_it):
        psi = np.where(sat, -psi, psi)
        psi = 2 * np.vdot(s, psi) * s - psi
    expected = bm.grover_sat(n, m).expected()
    data = [f"x.{i}" for i in range(n)]
    return Spec("GRV", f"n={n},m={m}", n, lambda: build_grv(n, clauses, k_it), data, data, psi,
                expected)


def spec_add(n: int, seed: int = 3) -> Spec:
    rng = random.Random(seed * 31 + n)                # same formula as adder()
    a_val = rng.randrange(1 << n)
    b_val = rng.randrange(1 << n)
    s = (a_val + b_val) % (1 << n)
    expected = bm.adder(n).expected()
    assert expected == {_key([(s >> i) & 1 for i in range(n)]): 1.0}, "adder() inputs drifted"
    data = [f"a.{i}" for i in range(n)] + [f"b.{i}" for i in range(n)]
    idx = a_val | (s << n)
    order = [nm for i in range(n) for nm in (f"a.{i}", f"b.{i}")]
    return Spec("ADD", f"n={n}", n, lambda: build_add(n, a_val, b_val), data,
                [f"b.{i}" for i in range(n)], None, expected,
                mps_order=order, basis_index=idx)


# --------------------------------------------------------------------------
# Metrics and verification
# --------------------------------------------------------------------------
def _tvd(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def qubit_names(qc: QuantumCircuit) -> list[str]:
    names = []
    for q in qc.qubits:
        reg, i = qc.find_bit(q).registers[0]
        names.append(reg.name if reg.size == 1 else f"{reg.name}.{i}")
    return names


def measure_like_pipeline(qq: QuantumCircuit) -> tuple[QuantumCircuit, dict]:
    t0 = time.perf_counter()
    tq = transpile(qq, basis_gates=["cx", "u", "measure", "reset"], optimization_level=1,
                   seed_transpiler=11)
    tt = time.perf_counter() - t0
    ops = tq.count_ops()
    used = {tq.find_bit(q).index for ins in tq.data for q in ins.qubits}
    return tq, {
        "qubits": tq.num_qubits,
        "active_qubits": len(used),
        "cx": int(ops.get("cx", 0)),
        "u": int(ops.get("u", 0)),
        "gates": int(ops.get("cx", 0) + ops.get("u", 0)),
        "depth": int(tq.depth()),
        "measure": int(ops.get("measure", 0)),
        "reset": int(ops.get("reset", 0)),
        "qiskit_ms": 1e3 * tt,
    }


def verify(tq: QuantumCircuit, spec: Spec) -> tuple[bool, str]:
    names = qubit_names(tq)
    pos = {nm: i for i, nm in enumerate(names)}
    meas = {}
    for ins in tq.data:
        if ins.operation.name == "measure":
            meas[names[tq.find_bit(ins.qubits[0]).index]] = tq.find_bit(ins.clbits[0]).index
    if set(meas) != set(spec.measured):
        return False, f"measured qubits {sorted(meas)} != {sorted(spec.measured)}"
    bare = tq.remove_final_measurements(inplace=False)
    if bare.count_ops().get("measure", 0):
        return False, "non-final measurement present"
    w = bare.num_qubits
    didx = [pos[nm] for nm in spec.data]
    anc = [i for i in range(w) if i not in set(didx)]

    if w <= SV_MAX:
        c = bare.copy()
        c.save_statevector()
        sv = np.asarray(AerSimulator(method="statevector").run(c).result().get_statevector())
        d = len(didx)
        ks = np.arange(1 << d)
        full = np.zeros(1 << d, dtype=np.int64)
        for j, qi in enumerate(didx):
            full |= ((ks >> j) & 1) << qi
        psi = sv[full]                                # data amplitudes with all ancillas = 0
        p_anc0 = float(np.vdot(psi, psi).real)
        if spec.ref_state is None:                    # basis-state reference
            fid = float(abs(psi[spec.basis_index]) ** 2)
        else:
            fid = float(abs(np.vdot(spec.ref_state, psi)) ** 2)
        mj = [spec.data.index(nm) for nm in spec.measured]
        dist: dict = {}
        for k, pk in enumerate(np.abs(psi) ** 2):
            if pk > 1e-15:
                key = _key([(k >> j) & 1 for j in mj])
                dist[key] = dist.get(key, 0.0) + float(pk)
        tvd = _tvd(dist, spec.expected)
        ok = p_anc0 > 1 - TOL and fid > 1 - TOL and tvd < 1e-7
        return ok, (f"exact statevector ({w}q): P(ancillas=0)={p_anc0:.12f}, "
                    f"fidelity vs reference={fid:.12f}, TVD vs expected()={tvd:.1e}")

    if spec.basis_index is None:
        return False, f"not verified: width {w} > {SV_MAX} and reference is not a basis state"
    # MPS fallback: relabel qubits so that interacting data pairs are adjacent
    order = [pos[nm] for nm in spec.mps_order] + anc
    newpos = [0] * w
    for new, old in enumerate(order):
        newpos[old] = new
    c = QuantumCircuit(w)
    c.compose(bare, qubits=newpos, inplace=True)
    want = [0] * w                                     # expected bit per (new) qubit position
    for j, qi in enumerate(didx):
        want[newpos[qi]] = (spec.basis_index >> j) & 1
    # Aer's MPS save_amplitudes returned 0 for basis states that its statevector method
    # gives amplitude 1 (checked on HS n=8), so use single-qubit marginals instead:
    # P(full basis state) >= 1 - sum_q P(q != expected bit)   (union bound)
    for q in range(w):
        c.save_probabilities([q], label=f"p{q}")
    data = AerSimulator(method="matrix_product_state").run(c).result().data(0)
    miss = sum(1.0 - float(data[f"p{q}"][want[q]]) for q in range(w))
    ok = miss < 1e-7
    return ok, (f"Aer MPS ({w}q, qubits relabelled for locality), single-qubit marginals vs "
                f"the expected basis state (data = reference, ancillas = 0): "
                f"P(basis state) >= {1 - miss:.12f}")


def run_row(spec: Spec, cfg: str, ws: int, reps: int) -> dict:
    t0 = time.perf_counter()
    qs = spec.build()
    build_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    qs.compile(workspace=ws)                          # untimed warm-up (one-off caches)
    cold = time.perf_counter() - t0
    builds, comps, sig = [], [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        qs = spec.build()
        builds.append(time.perf_counter() - t0)
        logical = len(qs.qubits)
        t0 = time.perf_counter()
        qc = qs.compile(workspace=ws)
        comps.append(time.perf_counter() - t0)
        s = (qc.num_qubits(), tuple(sorted(qc.count_ops().items())))
        assert sig is None or s == sig, "Qrisp compile is not deterministic across builds"
        sig = s
    qq = qc.to_qiskit()
    tq, m = measure_like_pipeline(qq)
    ok, how = verify(tq, spec)
    return {
        "bench": spec.bench, "params": spec.params, "size": spec.size, "config": cfg,
        "workspace": ws, **m, "logical_qubits": logical,
        "compile_ms": round(1e3 * statistics.median(comps), 3),
        "compile_cold_ms": round(1e3 * cold, 3),
        "build_ms": round(1e3 * statistics.median(builds), 3),
        "build_cold_ms": round(1e3 * build_cold, 3),
        "qiskit_ms": round(m["qiskit_ms"], 3),
        "correct": ok, "check": how,
        "qrisp_ops": " ".join(f"{k}:{v}" for k, v in sorted(qc.count_ops().items())),
    }


FIELDS = ["bench", "params", "size", "config", "workspace", "qubits", "active_qubits",
          "logical_qubits", "cx", "u", "gates", "depth", "measure", "reset", "compile_ms",
          "compile_cold_ms", "build_ms", "build_cold_ms", "qiskit_ms", "correct", "check",
          "qrisp_ops", "qrisp_version"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "qrisp.csv"))
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    qv = md.version("qrisp")
    print(f"Python {platform.python_version()}, qrisp {qv}, qiskit {md.version('qiskit')}, "
          f"qiskit-aer {md.version('qiskit-aer')}, numpy {np.__version__}", flush=True)
    specs = ([spec_hs(n) for n in (4, 8, 16)] + [spec_qaoa(n) for n in (4, 8, 16)]
             + [spec_grv(n) for n in (4, 6)] + [spec_add(n) for n in (4, 8)])
    rows = []
    for spec in specs:
        for cfg, ws in CONFIGS:
            r = run_row(spec, cfg, ws, args.reps)
            r["qrisp_version"] = qv
            rows.append(r)
            print(f"{r['bench']:5s} {r['params']:10s} {cfg:10s} q={r['qubits']:3d} "
                  f"cx={r['cx']:5d} depth={r['depth']:5d} compile={r['compile_ms']:8.2f}ms "
                  f"(cold {r['compile_cold_ms']:8.1f}) correct={r['correct']}  {r['check']}",
                  flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    if not all(r["correct"] for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
