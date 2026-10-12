import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from qreclaim import ABLATION, NR, QR, SB, SCOPE, CompileError, Program, compile_program, with_budget
from qreclaim.analysis import BASIS, TOP, Analysis
from qreclaim.benchmarks import VERIFY, adder, bv, hidden_shift, qaoa_ring, repetition_code
from qreclaim.ir import Instr
from qreclaim.typecheck import check_frees_clean
from qreclaim.sim import simulate, tvd


@pytest.mark.parametrize("bench", VERIFY, ids=lambda b: f"{b.name}-{b.params}")
@pytest.mark.parametrize("cfg", [NR, SCOPE, SB] + ABLATION, ids=lambda c: c.name)
def test_semantics_preserved(bench, cfg):
    c = compile_program(bench.build(), cfg)
    if c.nqubits > 20:
        pytest.skip("too wide for exact simulation")
    dist = simulate(c.instrs, c.phys, c.nqubits, c.program.ncbits)   # GC assertions inside
    assert tvd(dist, bench.expected()) < 1e-9


@pytest.mark.parametrize("budget", [3, 5, 9])
def test_budget_mode_preserves_semantics(budget):
    b = bv(8)
    c = compile_program(b.build(), with_budget(QR, budget))
    assert c.nqubits == budget
    assert tvd(simulate(c.instrs, c.phys, c.nqubits, c.program.ncbits), b.expected()) < 1e-9


def test_qubit_savings():
    assert compile_program(bv(16).build(), QR).nqubits == 2
    assert compile_program(repetition_code(5, 5).build(), QR).nqubits == 6
    assert compile_program(hidden_shift(8).build(), QR).nqubits == 17
    assert compile_program(hidden_shift(8).build(), SCOPE).nqubits == 24
    assert compile_program(adder(8).build(), QR).nqubits == 2 * 8 + 7     # carry-out eliminated
    # shared-input temporaries: only eager placement reclaims them
    assert compile_program(qaoa_ring(8).build(), QR).nqubits == 9
    assert compile_program(qaoa_ring(8).build(), SB).nqubits == 16


def test_analysis_lattice_and_deadline():
    p = Program("t")
    x = p.qreg("x", 2)
    t = p.ancilla()
    u = p.ancilla("u")
    p.ccx(x[0], x[1], t)
    p.h(u)
    p.x(x[0])
    A = Analysis(p.instrs)
    assert A.info[t].state == BASIS and set(A.info[t].deps) == {x[0], x[1]}
    assert A.info[t].deadline == len(p.instrs) - 1
    assert A.info[u].state == TOP


def test_aliasing_rejected():
    p = Program("alias")
    x = p.qreg("x", 1)
    p.cx(x[0], x[0])
    with pytest.raises(CompileError, match="aliasing"):
        compile_program(p, QR)


def test_modified_control_rejected():
    p = Program("deadline")
    x = p.qreg("x", 2)
    o = p.qubit("o")
    t = p.ancilla()
    p.ccx(x[0], x[1], t)
    p.h(x[0])
    p.cx(t, o)
    with pytest.raises(CompileError, match="cannot uncompute"):
        compile_program(p, QR)


def test_entangled_garbage_rejected():
    p = Program("leak")
    o = p.qubit("o")
    t = p.ancilla()
    p.h(t)
    p.cx(t, o)
    with pytest.raises(CompileError, match="never be reclaimed"):
        compile_program(p, QR)


def test_uncomputation_extends_liveness_of_dependencies():
    # u is computed from t; t must stay alive until u has been uncomputed.
    p = Program("chain")
    x = p.qreg("x", 2)
    o = p.qubit("o")
    t = p.ancilla()
    u = p.ancilla("u")
    p.ccx(x[0], x[1], t)
    p.cx(t, u)
    p.cx(u, o)
    p.h(x[0])
    c = compile_program(p, QR)
    order = [(i.op, i.tgts[0]) for i in c.instrs if i.op == "free"]
    assert order.index(("free", u)) < order.index(("free", t))


def test_free_counts_as_write_when_ordering_blocks():
    # z is a clean ancilla used only as a (negative) control of a's computation,
    # so its free and U(a) share an insertion point; the free must come last.
    p = Program("order")
    x = p.qreg("x", 1)
    o = p.qubit("o")
    a = p.ancilla("a")
    z = p.ancilla("z")
    p.cx(z, a, neg=True)
    p.cx(a, o)
    p.cx(x[0], o)
    c = compile_program(p, QR)
    frees = [i.tgts[0] for i in c.instrs if i.op == "free"]
    assert frees.index(a) < frees.index(z)


def test_tfree_validator_rejects_dirty_free():
    p = Program("dirty")
    x = p.qreg("x", 2)
    t = p.ancilla()
    p.ccx(x[0], x[1], t)
    instrs = p.instrs + [Instr("free", tgts=(t,))]
    with pytest.raises(CompileError, match="T-Free"):
        check_frees_clean(instrs, ("anc",))
    p.ccx(x[0], x[1], t)
    assert check_frees_clean(p.instrs + [Instr("free", tgts=(t,))], ("anc",)) == 1


def test_tfree_validator_rejects_modified_control():
    p = Program("ctl")
    x = p.qreg("x", 2)
    t = p.ancilla()
    p.ccx(x[0], x[1], t)
    p.x(x[0])
    p.ccx(x[0], x[1], t)
    with pytest.raises(CompileError, match="control"):
        check_frees_clean(p.instrs + [Instr("free", tgts=(t,))], ("anc",))
