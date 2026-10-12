"""Benchmark programs, written in the high-level style the paper targets:
programmers allocate temporaries freely and never uncompute or free them.

Each builder returns a fresh ``Program`` plus an ``expected`` function that
maps the program to its ideal output distribution, used for verification.

* BV(n)      Bernstein-Vazirani with an n-bit secret      (measured data qubits)
* HS(n)      hidden shift for the inner-product bent function on 2n bits,
             oracle written with one temporary per product (flat temporaries)
* RC(d, r)   distance-d repetition-code memory, r syndrome rounds, one
             fresh syndrome qubit per parity check            (mid-circuit measurement)
* ADD(n)     ripple-carry adder: carries computed into ancillas, sum written
             top-down; carry-out unused                       (dependency chain)
* GRV(n, m)  Grover search over a planted 3-SAT instance with n variables and
             m clauses, one ancilla per clause                 (iterated oracle)
* QAOA(n)    one QAOA layer for MaxCut on an n-cycle; each edge phase is
             computed through a temporary t = x_i AND x_(i+1), so consecutive
             temporaries share an input                         (shared inputs)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from .ir import Program


@dataclass
class Bench:
    name: str
    params: str
    build: Callable[[], Program]
    expected: Callable[[], dict]


def _bits_key(bits: list[int]) -> str:
    return "".join(str(b) for b in bits)


# --------------------------------------------------------------------------
def bv(n: int, seed: int = 1) -> Bench:
    rng = random.Random(seed * 7919 + n)
    s = [rng.randint(0, 1) for _ in range(n)]
    s[0] = 1

    def build() -> Program:
        p = Program(f"bv{n}")
        x = p.qreg("x", n)
        t = p.qubit("tgt")
        p.x(t)
        p.h(t)
        for q in x:
            p.h(q)
        for i, q in enumerate(x):
            if s[i]:
                p.cx(q, t)
        for q in x:
            p.h(q)
        for q in x:
            p.measure(q)
        return p

    return Bench("BV", f"n={n}", build, lambda: {_bits_key(s): 1.0})


# --------------------------------------------------------------------------
def hidden_shift(n: int, seed: int = 2) -> Bench:
    rng = random.Random(seed * 104729 + n)
    shift = [rng.randint(0, 1) for _ in range(2 * n)]

    def oracle(p: Program, x, y) -> None:
        # phase (-1)^{x.y}, computed through one temporary per product
        with p.scope():
            for i in range(n):
                t = p.ancilla("t")
                p.ccx(x[i], y[i], t)
                p.z(t)

    def build() -> Program:
        p = Program(f"hs{n}")
        x = p.qreg("x", n)
        y = p.qreg("y", n)
        z = x + y
        for q in z:
            p.h(q)
        for q, b in zip(z, shift):
            if b:
                p.x(q)
        oracle(p, x, y)
        for q, b in zip(z, shift):
            if b:
                p.x(q)
        for q in z:
            p.h(q)
        oracle(p, x, y)            # the inner product is self-dual
        for q in z:
            p.h(q)
        for q in z:
            p.measure(q)
        return p

    return Bench("HS", f"n={n}", build, lambda: {_bits_key(shift): 1.0})


# --------------------------------------------------------------------------
def repetition_code(d: int, r: int, err: tuple | None = None) -> Bench:
    err = err if err is not None else (r // 2, d // 2)

    def build() -> Program:
        p = Program(f"rc{d}x{r}")
        data = p.qreg("d", d)
        for rnd in range(r):
            if rnd == err[0]:
                p.x(data[err[1]])
            for j in range(d - 1):
                s = p.qubit("s")
                p.cx(data[j], s)
                p.cx(data[j + 1], s)
                p.measure(s)
        for q in data:
            p.measure(q)
        return p

    def expected() -> dict:
        bits = []
        for rnd in range(r):
            flipped = rnd >= err[0]
            for j in range(d - 1):
                bits.append(int(flipped and err[1] in (j, j + 1)))
        bits += [int(i == err[1]) for i in range(d)]
        return {_bits_key(bits): 1.0}

    return Bench("RC", f"d={d},r={r}", build, expected)


# --------------------------------------------------------------------------
def adder(n: int, a_val: int | None = None, b_val: int | None = None, seed: int = 3) -> Bench:
    rng = random.Random(seed * 31 + n)
    a_val = rng.randrange(1 << n) if a_val is None else a_val
    b_val = rng.randrange(1 << n) if b_val is None else b_val

    def build() -> Program:
        p = Program(f"add{n}")
        a = p.qreg("a", n)
        b = p.qreg("b", n)
        for i in range(n):
            if (a_val >> i) & 1:
                p.x(a[i])
            if (b_val >> i) & 1:
                p.x(b[i])
        with p.scope():
            c = [None] * (n + 1)
            for i in range(n):                       # carries: c[i+1] = MAJ(a_i, b_i, c_i)
                c[i + 1] = p.ancilla("c")
                p.ccx(a[i], b[i], c[i + 1])
                if i > 0:
                    p.ccx(a[i], c[i], c[i + 1])
                    p.ccx(b[i], c[i], c[i + 1])
            for i in reversed(range(n)):             # sums, most significant bit first
                p.cx(a[i], b[i])
                if i > 0:
                    p.cx(c[i], b[i])
        for q in b:
            p.measure(q)
        return p

    def expected() -> dict:
        s = (a_val + b_val) % (1 << n)
        return {_bits_key([(s >> i) & 1 for i in range(n)]): 1.0}

    return Bench("ADD", f"n={n}", build, expected)


# --------------------------------------------------------------------------
def planted_3sat(n: int, m: int, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    sol = [rng.randint(0, 1) for _ in range(n)]
    clauses = []
    while len(clauses) < m:
        vs = rng.sample(range(n), 3)
        lits = [(v, rng.randint(0, 1)) for v in vs]         # (var, positive?)
        if any((sol[v] == 1) == bool(pos) for v, pos in lits):
            clauses.append(lits)
    return sol, clauses


def _sat(assign: list, clauses: list) -> bool:
    return all(any((assign[v] == 1) == bool(pos) for v, pos in c) for c in clauses)


def grover_sat(n: int, m: int, seed: int = 5, iterations: int | None = None) -> Bench:
    _, clauses = planted_3sat(n, m, seed * 1009 + n * 13 + m)
    N = 1 << n
    sols = [k for k in range(N) if _sat([(k >> i) & 1 for i in range(n)], clauses)]
    M = len(sols)
    theta = math.asin(math.sqrt(M / N))
    k_it = iterations if iterations is not None else max(1, int(math.floor(math.pi / (4 * theta))))

    def oracle(p: Program, x) -> None:
        with p.scope():
            cs = []
            for lits in clauses:
                c = p.ancilla("c")                     # c = l1 or l2 or l3
                p.x(c)
                p.mcx([x[v] for v, _ in lits], c, neg=[bool(pos) for _, pos in lits])
                cs.append(c)
            p.mcz(cs)

    def diffusion(p: Program, x) -> None:
        for q in x:
            p.h(q)
        for q in x:
            p.x(q)
        p.mcz(x)
        for q in x:
            p.x(q)
        for q in x:
            p.h(q)

    def build() -> Program:
        p = Program(f"grover{n}x{m}")
        x = p.qreg("x", n)
        for q in x:
            p.h(q)
        for _ in range(k_it):
            oracle(p, x)
            diffusion(p, x)
        for q in x:
            p.measure(q)
        return p

    def expected() -> dict:
        good = math.sin((2 * k_it + 1) * theta) ** 2
        dist = {}
        for k in range(N):
            key = _bits_key([(k >> i) & 1 for i in range(n)])
            dist[key] = good / M if k in sols else (1 - good) / (N - M) if N > M else 0.0
        return {k: v for k, v in dist.items() if v > 1e-12}

    return Bench("GRV", f"n={n},m={m}", build, expected)


def qaoa_ring(n: int, layers: int = 1, gamma: float = 0.7, beta: float = 0.4) -> Bench:
    edges = [(i, (i + 1) % n) for i in range(n)]

    def build() -> Program:
        p = Program(f"qaoa{n}")
        x = p.qreg("x", n)
        for q in x:
            p.h(q)
        for _ in range(layers):
            with p.scope():                      # phase separator e^{i gamma x_i x_j}
                for i, j in edges:
                    t = p.ancilla("t")
                    p.ccx(x[i], x[j], t)
                    p.p(gamma, t)
            for q in x:                          # mixer
                p.rx(2 * beta, q)
        for q in x:
            p.measure(q)
        return p

    def expected() -> dict:
        # Independent reference: the same circuit written with CP gates and no
        # ancillas, evaluated exactly by Qiskit's statevector simulator.
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for _ in range(layers):
            for i, j in edges:
                qc.cp(gamma, i, j)
            for q in range(n):
                qc.rx(2 * beta, q)
        probs = Statevector(qc).probabilities_dict()
        return {k[::-1]: v for k, v in probs.items() if v > 1e-14}

    return Bench("QAOA", f"n={n}", build, expected)


SUITES = {
    "BV": [bv(n) for n in (8, 16, 32, 64, 128)],
    "HS": [hidden_shift(n) for n in (4, 8, 16, 32, 64)],
    "RC": [repetition_code(d, d) for d in (3, 5, 7, 9, 11)],
    "ADD": [adder(n) for n in (4, 8, 16, 32, 64)],
    "GRV": [grover_sat(n, 2 * n) for n in (4, 6, 8, 10)],
    "QAOA": [qaoa_ring(n) for n in (4, 8, 16, 32, 64)],
}

VERIFY = [bv(8), bv(12), hidden_shift(3), hidden_shift(4), repetition_code(3, 3),
          repetition_code(5, 3), adder(4), adder(5), grover_sat(4, 4), grover_sat(5, 6),
          qaoa_ring(4), qaoa_ring(6)]
