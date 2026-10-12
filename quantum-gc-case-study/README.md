# Case study: Memory Management and Garbage Collection Strategies in High-Level Quantum Programming Languages

Core CSE case study (Title 1). The report is in IEEE conference format:

- **Report (PDF):** [`Memory_Management_GC_Quantum_Languages_IEEE.pdf`](Memory_Management_GC_Quantum_Languages_IEEE.pdf)
- **LaTeX source:** [`paper/main.tex`](paper/main.tex) and [`paper/refs.bib`](paper/refs.bib). It uses the `IEEEtran` class, so it compiles on Overleaf without changes.

The report describes **QReclaim**, a compile-time garbage collector for qubits, and evaluates a working prototype. Every number, table and figure in the report comes from the code in this folder.

## What is here

| Path | Contents |
|---|---|
| `qreclaim/ir.py` | QR-IR (the intermediate representation) and the embedded front-end DSL |
| `qreclaim/typecheck.py` | linear type check (no aliasing, lifetimes), static T-Free validator (XOR cancellation), physical no-aliasing check |
| `qreclaim/analysis.py` | basis-state abstract interpretation (forward) and liveness (backward) |
| `qreclaim/reclaim.py` | reclamation pass: uncomputation-aware liveness fixpoint, eager/lazy placement, early measurement + reset, dead-garbage elimination |
| `qreclaim/lower.py` | lowering of multi-controlled gates to V-chains with helper qubits |
| `qreclaim/schedule.py` | pressure-aware list scheduling (PALS) |
| `qreclaim/alloc.py` | earliest-ready-first linear-scan allocation with a qubit budget |
| `qreclaim/codegen.py` | Qiskit / OpenQASM 2 code generation |
| `qreclaim/sim.py` | exact branching simulator that asserts every freed qubit is \|0> |
| `qreclaim/pipeline.py` | the pipeline; NR, SCOPE, SB (strong baseline) and QReclaim configurations; leave-one-out ablation; latency model |
| `qreclaim/benchmarks.py` | BV, HS, RC, ADD, GRV and QAOA-ring programs, with their expected outputs |
| `experiments/run_all.py` | regenerates `results/*.csv`, `results/summary.json`, `paper/figures/*.pdf`, `paper/tables/*.tex` (incl. `numbers.tex`, every number quoted in the paper) |
| `experiments/mutation_eval.py` | bug-injection study of the static checks (`results/mutation.csv`) |
| `experiments/qrisp_baseline.py` | external baseline: the same programs in Qrisp 0.9.9 (`results/qrisp.csv`; needs `pip install qrisp`) |
| `tests/` | pytest suite |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests                # 98 passed, 1 skipped (too wide to simulate)
python experiments/mutation_eval.py      # ~5 min; bug-injection study
python experiments/run_all.py            # ~10 min; writes results/, paper/figures/, paper/tables/
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Headline results

See `results/summary.json` and Table IV of the report. Highlights: Bernstein-Vazirani 129 -> 2 qubits,
repetition code 121 -> 12, Grover 262 -> 47, QAOA ring 128 -> 65 (only eager placement achieves this; the
strong baseline needs 128), with a depth-neutral mode that never increases depth. Qrisp, run on the same
programs, matches QReclaim's width on HS and QAOA and is narrower on GRV and ADD because of its helper-free
gate synthesis; the report discusses this honestly. The static checks rejected every faulty mutant
in the bug-injection study.
