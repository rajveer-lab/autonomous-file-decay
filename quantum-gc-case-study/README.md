# Case study: Memory Management and Garbage Collection Strategies in High-Level Quantum Programming Languages

Core CSE case study (Title 1). The report is in IEEE conference format:

- **Report (PDF):** [`Memory_Management_GC_Quantum_Languages_IEEE.pdf`](Memory_Management_GC_Quantum_Languages_IEEE.pdf)
- **LaTeX source:** [`paper/main.tex`](paper/main.tex) and [`paper/refs.bib`](paper/refs.bib). It uses the `IEEEtran` class, so it compiles on Overleaf without changes.

The report describes **QReclaim**, a compile-time garbage collector for qubits, and evaluates a working prototype. Every number, table and figure in the report comes from the code in this folder.

## What is here

| Path | Contents |
|---|---|
| `qreclaim/ir.py` | QR-IR (the intermediate representation) and the embedded front-end DSL |
| `qreclaim/typecheck.py` | linear type check (the static form of no-cloning) and the physical no-aliasing check |
| `qreclaim/analysis.py` | basis-state abstract interpretation (forward) and liveness (backward) |
| `qreclaim/reclaim.py` | reclamation pass: uncomputation-aware liveness fixpoint, eager/lazy placement, early measurement + reset, dead-garbage elimination |
| `qreclaim/lower.py` | lowering of multi-controlled gates to V-chains with helper qubits |
| `qreclaim/schedule.py` | pressure-aware list scheduling (PALS) |
| `qreclaim/alloc.py` | earliest-ready-first linear-scan allocation with a qubit budget |
| `qreclaim/codegen.py` | Qiskit / OpenQASM 2 code generation |
| `qreclaim/sim.py` | exact branching simulator that asserts every freed qubit is \|0> |
| `qreclaim/pipeline.py` | the pipeline and the NR / SCOPE / QReclaim configurations |
| `qreclaim/benchmarks.py` | BV, HS, RC, ADD and GRV benchmark programs, with their expected outputs |
| `experiments/run_all.py` | regenerates `results/*.csv`, `paper/figures/*.pdf` and `paper/tables/*.tex` |
| `tests/` | pytest suite |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests                # 98 passed, 1 skipped (too wide to simulate)
python experiments/run_all.py            # ~5 min; writes results/, paper/figures/, paper/tables/
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Headline results (from `results/main.csv`)

| Benchmark | Qubits: no reuse | Lexical (SCOPE) | QReclaim-min | QReclaim depth-neutral |
|---|---|---|---|---|
| Bernstein–Vazirani, n=128 | 129 | 129 | **2** | 12 (same depth) |
| Hidden shift, n=64 | 256 | 192 | **129** | 192 (same depth) |
| Repetition code, d=r=11 | 121 | 121 | **12** | 67 (same depth) |
| Ripple-carry adder, n=64 | 192 | 192 | **191** | 191 |
| Grover 3-SAT, n=10 | 262 | 47 | **47** | 49 (same depth) |

Over all 24 instances, QReclaim-min uses 55.6% fewer qubits than no reuse and 42.3% fewer than lexical reclamation. The depth-neutral mode saves 36.2% with no increase in circuit depth.
