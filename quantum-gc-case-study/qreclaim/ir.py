"""Core intermediate representation (QR-IR) and the embedded front-end DSL.

A QR-IR program is a list of instructions over *logical* qubits.  Every gate
is stored as a base operation (``x``, ``z``, ``h`` ...) applied to target
qubits, optionally guarded by a list of control qubits with a per-control
polarity.  ``cx``, ``ccx`` and ``mcx`` are therefore the same base operation
``x`` with one, two or many controls, which keeps the analyses uniform.

Gates are classified by how they act on computational-basis states:

* ``qfree``   - basis permutations (X family, SWAP): |b> -> |pi(b)>
* ``diag``    - basis-preserving phases (Z, S, T, Rz, P families)
* ``general`` - everything that can create superposition (H, Rx, Ry)

This classification is what the reclamation analysis needs: only qfree
computation can be undone by simple reversal, and only qfree/general gates
change the *basis value* of the qubits they target.
"""

from __future__ import annotations

import collections
import itertools
from contextlib import contextmanager
from dataclasses import dataclass

QFREE = frozenset({"x", "swap"})
DIAG = frozenset({"z", "s", "sdg", "t", "tdg", "rz", "p"})
GENERAL = frozenset({"h", "rx", "ry"})
ROTATIONS = frozenset({"rz", "p", "rx", "ry"})
_INVERSE_NAME = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}

# Qubit kinds.  DATA qubits belong to the programmer (inputs / outputs);
# ANC qubits are temporaries whose reclamation is the compiler's job;
# HELPER qubits are introduced by lowering multi-controlled gates.
DATA, ANC, HELPER = "data", "anc", "helper"


class CompileError(Exception):
    """A program rejected by one of QReclaim's static checks."""


@dataclass(eq=False)
class Qubit:
    uid: int
    name: str
    kind: str
    scope: int = 0

    def __repr__(self) -> str:
        return self.name


@dataclass(eq=False)
class Instr:
    op: str                     # alloc | gate | measure | reset | free | scope_end
    name: str = ""              # base gate name (gates only)
    tgts: tuple = ()
    ctrls: tuple = ()
    neg: tuple = ()             # per control: True means "control on |0>"
    params: tuple = ()
    cbit: int | None = None
    tag: str = "user"           # user | uncompute | lower | gc
    sid: int | None = None      # scope id (scope_end markers only)

    @property
    def qubits(self) -> tuple:
        return self.ctrls + self.tgts

    @property
    def category(self) -> str:
        if self.name in QFREE:
            return "qfree"
        if self.name in DIAG:
            return "diag"
        return "general"

    def writes(self) -> tuple:
        """Qubits whose computational-basis value this instruction may change."""
        if self.op == "gate":
            return () if self.category == "diag" else self.tgts
        if self.op == "reset":
            return self.tgts
        return ()

    def inverse(self, tag: str = "uncompute") -> "Instr":
        name = _INVERSE_NAME.get(self.name, self.name)
        params = tuple(-p for p in self.params) if self.name in ROTATIONS else self.params
        return Instr("gate", name, self.tgts, self.ctrls, self.neg, params, tag=tag)

    def __repr__(self) -> str:
        if self.op == "gate":
            k = len(self.ctrls)
            prefix = "c" * k if k <= 2 else "mc"
            par = "(" + ", ".join(f"{p:.3g}" for p in self.params) + ")" if self.params else ""
            ctl = ", ".join(("!" if n else "") + q.name for q, n in zip(self.ctrls, self.neg))
            tgt = ", ".join(q.name for q in self.tgts)
            return f"{prefix}{self.name}{par} {ctl + ' -> ' if ctl else ''}{tgt}"
        if self.op == "measure":
            return f"measure {self.tgts[0].name} -> c[{self.cbit}]"
        if self.op == "scope_end":
            return f"end_scope {self.sid}"
        return f"{self.op} {self.tgts[0].name}"


class Program:
    """Embedded high-level front end.

    Programmers allocate registers and *ancillas*, write only the forward
    computation, and never uncompute or free anything by hand: reclaiming
    ancillas (and measured qubits) is the compiler's responsibility.
    ``with prog.scope():`` delimits a lexical scope, which the scope-based
    baseline uses as its reclamation point.
    """

    def __init__(self, name: str):
        self.name = name
        self.instrs: list[Instr] = []
        self.qubits: list[Qubit] = []
        self.ncbits = 0
        self._uid = itertools.count()
        self._sid = itertools.count(1)
        self._scopes = [0]
        self._names: collections.Counter = collections.Counter()

    # -- allocation -------------------------------------------------------
    def new_qubit(self, name: str, kind: str, scope: int | None = None) -> Qubit:
        q = Qubit(next(self._uid), name, kind, self._scopes[-1] if scope is None else scope)
        self.qubits.append(q)
        return q

    def new_scope_id(self) -> int:
        return next(self._sid)

    def _alloc(self, name: str, kind: str) -> Qubit:
        q = self.new_qubit(name, kind)
        self.instrs.append(Instr("alloc", tgts=(q,)))
        return q

    def _fresh_name(self, base: str) -> str:
        k = self._names[base]
        self._names[base] += 1
        return f"{base}{k}"

    def qreg(self, name: str, n: int) -> list[Qubit]:
        return [self._alloc(f"{name}{i}", DATA) for i in range(n)]

    def qubit(self, name: str = "q") -> Qubit:
        return self._alloc(self._fresh_name(name), DATA)

    def ancilla(self, name: str = "t") -> Qubit:
        return self._alloc(self._fresh_name(name), ANC)

    @contextmanager
    def scope(self):
        sid = next(self._sid)
        self._scopes.append(sid)
        try:
            yield sid
        finally:
            self._scopes.pop()
            self.instrs.append(Instr("scope_end", sid=sid))

    # -- gates ------------------------------------------------------------
    def gate(self, name, tgts, ctrls=(), neg=None, params=()) -> None:
        ctrls = tuple(ctrls)
        neg = tuple(bool(n) for n in neg) if neg is not None else (False,) * len(ctrls)
        if len(neg) != len(ctrls):
            raise ValueError("one polarity flag per control is required")
        self.instrs.append(Instr("gate", name, tuple(tgts), ctrls, neg, tuple(params)))

    def x(self, q): self.gate("x", (q,))
    def h(self, q): self.gate("h", (q,))
    def z(self, q): self.gate("z", (q,))
    def s(self, q): self.gate("s", (q,))
    def sdg(self, q): self.gate("sdg", (q,))
    def t(self, q): self.gate("t", (q,))
    def tdg(self, q): self.gate("tdg", (q,))
    def rz(self, theta, q): self.gate("rz", (q,), params=(theta,))
    def rx(self, theta, q): self.gate("rx", (q,), params=(theta,))
    def ry(self, theta, q): self.gate("ry", (q,), params=(theta,))
    def p(self, theta, q): self.gate("p", (q,), params=(theta,))
    def swap(self, a, b): self.gate("swap", (a, b))
    def cx(self, c, t, neg=False): self.gate("x", (t,), (c,), (neg,))
    def ccx(self, c1, c2, t, neg=(False, False)): self.gate("x", (t,), (c1, c2), neg)
    def mcx(self, ctrls, t, neg=None): self.gate("x", (t,), ctrls, neg)
    def cz(self, a, b): self.gate("z", (b,), (a,))
    def cp(self, theta, c, t): self.gate("p", (t,), (c,), params=(theta,))

    def mcz(self, qs) -> None:
        qs = list(qs)
        self.gate("z", (qs[-1],), qs[:-1])

    def measure(self, q) -> int:
        c = self.ncbits
        self.ncbits += 1
        self.instrs.append(Instr("measure", tgts=(q,), cbit=c))
        return c

    def reset(self, q) -> None:
        self.instrs.append(Instr("reset", tgts=(q,)))


def operands(ins: Instr) -> tuple:
    """Every qubit an instruction touches."""
    return ins.qubits if ins.op == "gate" else ins.tgts


def strip_markers(instrs: list[Instr]) -> list[Instr]:
    return [i for i in instrs if i.op != "scope_end"]
