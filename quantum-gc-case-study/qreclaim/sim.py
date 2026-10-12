"""Exact branching state-vector simulator with run-time GC assertions.

Mid-circuit measurements split the simulation into weighted branches, so the
returned distribution over classical bit-strings is exact (no sampling).
Every ``free`` asserts that the reclaimed qubit is exactly |0>, and every
``alloc`` asserts that the physical qubit handed out is |0>; together they
validate dynamically what the reclamation pass guarantees statically.
"""

from __future__ import annotations

import numpy as np

from .ir import Instr

EPS = 1e-10
_S2 = 1 / np.sqrt(2)


class GCAssertionError(AssertionError):
    pass


def _matrix(ins: Instr) -> np.ndarray:
    n = ins.name
    if n == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if n == "h":
        return np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex)
    if n == "z":
        return np.diag([1, -1]).astype(complex)
    if n in ("s", "sdg", "t", "tdg"):
        ang = {"s": np.pi / 2, "sdg": -np.pi / 2, "t": np.pi / 4, "tdg": -np.pi / 4}[n]
        return np.diag([1, np.exp(1j * ang)])
    th = ins.params[0] if ins.params else 0.0
    if n == "p":
        return np.diag([1, np.exp(1j * th)])
    if n == "rz":
        return np.diag([np.exp(-0.5j * th), np.exp(0.5j * th)])
    if n == "rx":
        c, s = np.cos(th / 2), np.sin(th / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])
    if n == "ry":
        c, s = np.cos(th / 2), np.sin(th / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    raise NotImplementedError(n)


class _Sim:
    def __init__(self, nq: int):
        self.idx = np.arange(1 << nq, dtype=np.int64)

    def bit(self, p: int) -> np.ndarray:
        return (self.idx >> p) & 1

    def apply(self, psi, U, t, ctrls, negs) -> None:
        mask = self.bit(t) == 0
        for c, ng in zip(ctrls, negs):
            mask &= self.bit(c) == (0 if ng else 1)
        i0 = self.idx[mask]
        i1 = i0 | (1 << t)
        a0, a1 = psi[i0], psi[i1]
        psi[i0] = U[0, 0] * a0 + U[0, 1] * a1
        psi[i1] = U[1, 0] * a0 + U[1, 1] * a1

    def p1(self, psi, p) -> float:
        return float(np.sum(np.abs(psi[self.bit(p) == 1]) ** 2))


def simulate(instrs: list[Instr], phys: dict, nqubits: int, ncbits: int,
             max_branches: int = 4096) -> dict:
    sim = _Sim(nqubits)
    psi0 = np.zeros(1 << nqubits, dtype=complex)
    psi0[0] = 1.0
    branches = [(1.0, psi0, (None,) * ncbits)]
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    for k, ins in enumerate(instrs):
        if ins.op == "gate":
            U = _matrix(ins)
            for _, psi, _ in branches:
                if ins.name == "swap":
                    a, b = phys[ins.tgts[0]], phys[ins.tgts[1]]
                    for c, t in ((a, b), (b, a), (a, b)):
                        sim.apply(psi, X, t, [c], [False])
                else:
                    sim.apply(psi, U, phys[ins.tgts[0]], [phys[c] for c in ins.ctrls], ins.neg)
        elif ins.op in ("alloc", "free"):
            p = phys[ins.tgts[0]]
            for w, psi, _ in branches:
                pr = sim.p1(psi, p)
                if pr > 1e-8:
                    raise GCAssertionError(
                        f"[{k}] {ins.op} {ins.tgts[0]} (physical {p}): qubit is not |0>, P(1)={pr:.3g}")
        elif ins.op in ("measure", "reset"):
            p = phys[ins.tgts[0]]
            sel1 = sim.bit(p) == 1
            nxt = []
            for w, psi, cb in branches:
                pr1 = sim.p1(psi, p)
                for outcome, pr in ((0, 1 - pr1), (1, pr1)):
                    if pr < EPS:
                        continue
                    phi = psi.copy()
                    phi[sel1 if outcome == 0 else ~sel1] = 0
                    phi /= np.sqrt(pr)
                    if ins.op == "measure":
                        cb2 = cb[:ins.cbit] + (outcome,) + cb[ins.cbit + 1:]
                    else:
                        cb2 = cb
                        if outcome == 1:
                            sim.apply(phi, X, p, [], [])
                    nxt.append((w * pr, phi, cb2))
            branches = nxt
            if len(branches) > max_branches:
                raise RuntimeError("too many measurement branches to simulate exactly")
    dist: dict = {}
    for w, _, cb in branches:
        key = "".join("x" if b is None else str(b) for b in cb)
        dist[key] = dist.get(key, 0.0) + w
    return dist


def tvd(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
