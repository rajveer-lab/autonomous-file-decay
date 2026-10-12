"""Code generation: physical QR-IR -> Qiskit QuantumCircuit / OpenQASM 2.

Negative controls are realised with X conjugation at this last stage, so
the analyses never see the extra X gates as writes to control qubits.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, qasm2

from .ir import Instr


def to_qiskit(instrs: list[Instr], phys: dict, nqubits: int, ncbits: int) -> QuantumCircuit:
    qc = QuantumCircuit(nqubits, ncbits)
    for ins in instrs:
        if ins.op == "gate":
            cs = [phys[c] for c in ins.ctrls]
            ts = [phys[t] for t in ins.tgts]
            flips = [phys[c] for c, n in zip(ins.ctrls, ins.neg) if n]
            for p in flips:
                qc.x(p)
            _emit(qc, ins, cs, ts)
            for p in flips:
                qc.x(p)
        elif ins.op == "measure":
            qc.measure(phys[ins.tgts[0]], ins.cbit)
        elif ins.op == "reset":
            qc.reset(phys[ins.tgts[0]])
    return qc


def _emit(qc: QuantumCircuit, ins: Instr, cs: list, ts: list) -> None:
    k, n, t = len(cs), ins.name, ts[0]
    if n == "x":
        [qc.x, qc.cx, qc.ccx][k](*cs, t)
    elif n == "z":
        [qc.z, qc.cz, qc.ccz][k](*cs, t)
    elif n == "p":
        [qc.p, qc.cp][k](ins.params[0], *cs, t)
    elif n == "swap" and k == 0:
        qc.swap(ts[0], ts[1])
    elif k == 0:
        getattr(qc, n)(*ins.params, t)
    else:
        raise NotImplementedError(f"cannot emit {ins}")


def to_qasm(qc: QuantumCircuit) -> str:
    return qasm2.dumps(qc)
