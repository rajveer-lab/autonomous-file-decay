# Learned Page Replacement

Working code for the Review 1 project *A Lightweight Learned Page Replacement
Policy for Operating System Memory Management* (Rajveer Singh Chopra). It
contains every module from the block diagram in the Review 1 notes:

| Module | File | What it does |
|---|---|---|
| Trace loader | `mlpaging/traces.py` | Textbook string, five synthetic patterns, three program-derived traces (matrix multiply, merge sort, BFS), and a loader for MSR Cambridge block traces |
| Simulator core | `mlpaging/sim.py` | Replays a trace through a policy and counts faults; times each eviction decision |
| Policy library | `mlpaging/policies.py` | FIFO, LRU, LFU, CLOCK, ARC [8], OPT [1], LeCaR [14], SIEVE [24], S3-FIFO [23], and sampled LRU |
| Feature extractor | `mlpaging/learned.py` (`Meta`) | Per-page history: age, access count, last three gaps, mean gap, age ÷ mean gap |
| Model trainer | `mlpaging/learned.py` | Replays training traces under OPT, records sampled pages with their true distance to next use, trains logistic regression and depth-4 decision trees |
| Learned policy | `mlpaging/learned.py` (`LearnedPolicy`) | On a fault: score k sampled pages and evict the highest; fall back to LRU when unsure; a regret guard switches between model and LRU |
| Metrics and graphs | `mlpaging/experiment.py`, `mlpaging/report.py` | Fault rate, reduction vs LRU, gap closed, worst case, decision time, metadata and model size; tables and figures |

Only Python 3.10+, NumPy and Matplotlib are needed.

## Run it

```bash
cd ml-page-replacement
python3 -m mlpaging verify            # FIFO 15, LRU 12, OPT 9 on the textbook string
python3 -m mlpaging demo --trace loop # every baseline on one trace
python3 -m mlpaging run               # full pipeline (about 15-20 min on 4 cores)
python3 -m mlpaging run --quick       # smaller smoke test (about 2 min)
python3 -m pytest -q tests            # 28 tests
```

`run` writes `results/summary.md` (all tables), `results/test.csv`,
`results/validation.csv`, `results/ablation.csv`, `results/summary.json`, and
the figures `figures/fault_rate_vs_memory.png` and `figures/reduction_vs_lru.png`.

## Methodology (matches the seven phases in the notes)

1. **Baselines and simulator.** All policies share one interface. The
   textbook string with 3 frames gives FIFO 15, LRU 12, OPT 9 faults; `run`
   stops if it doesn't.
2. **Workloads.** Three disjoint splits:
   - *train*: loop, Zipf, phase change, scan + hot set, mixed, matrix multiply
   - *validation*: the same families with new seeds and parameters, plus merge sort
   - *test*: new seeds and parameters again, a larger matrix multiply, a larger
     merge sort, and BFS, a program the model never sees in training.
3. **Training data.** Each training trace is replayed under OPT at memory
   sizes of 1, 2, 5 and 10% of its distinct pages. At OPT's evictions, 16
   resident pages are sampled. Each gets a feature row built only from its
   past accesses, plus its true distance to next use, which only OPT knows.
   The classifier label is 1 if that distance is more than H accesses
   (H = horizon × memory size). OPT is used only here, offline.
4. **Model selection.** Seven candidates are trained: logistic regression and
   a depth-4 tree as classifiers for horizons 0.5, 1 and 2, plus a depth-4
   regression tree that predicts log(distance to next use). The one with the
   fewest simulated faults on the validation traces wins, not the one with
   the best accuracy.
5. **Policy integration.** Sample size k ∈ {8, 16, 32} and the confidence
   threshold (3 values per model type) are tuned on validation traces.
6. **Evaluation.** Every policy runs on every test trace at every memory
   size. Test traces are never used for any choice. The test also runs
   `Learned (no guard)`, the same model without the regret guard, to show
   what the safety net costs and saves.
7. **Feature ablation.** The chosen model is retrained without each feature
   group (recency, frequency, gap history) to show which signals matter.

Formulas, with F the number of faults on a trace:

- Fault reduction vs LRU = (F_LRU − F_policy) / F_LRU × 100
- Gap closed = (F_LRU − F_policy) / (F_LRU − F_OPT) × 100

## Design notes

- **Two safety nets (Objective 4).** First, if even the best sampled score
  is below the threshold, evict the least recently used page. Second, a
  regret guard borrowed from LeCaR [14]: pages evicted on the model's advice
  and pages evicted by LRU go into two histories of C entries. A fault on a
  page in the model's history means the model threw out something still
  needed, so LRU gains weight, and the other way round; each eviction follows
  one of the two by weight. During development, merge sort exposed why this
  matters. Without the guard, the model evicted pages that were in the middle
  of being read and faulted up to 14 times as often as LRU. With the guard,
  it stays within 1% of LRU there.
- **Per-page metadata is 24 bytes:** last access, count, three gaps and a gap
  sum. It is kept for every page seen, including evicted ones, the way Linux
  keeps shadow entries for evicted page-cache pages. Without it, a page that
  returns after eviction would look brand new, and loops would be
  invisible.
- **Features are scaled by the memory size,** so one model works at every
  size.
- **The cost per fault is constant:** k feature rows and k model evaluations
  (9 multiply-adds for logistic regression, at most 4 comparisons for the
  tree). The decision times in the results are from this Python simulator,
  so compare them across policies only.
- **`SampledLRU-16` is the fallback rule on its own.** Comparing it with
  `Learned` shows what the model adds beyond sampling.

## MSR Cambridge traces

The SNIA IOTTA site couldn't be reached from the machine that built this
project, so the results use only the synthetic and program-derived traces.
To add a real trace, download a volume of the MSR Cambridge set from
http://iotta.snia.org/traces/block-io/388 (free under SNIA's trace licence),
then run:

```bash
python3 -m mlpaging msr path/to/web_0.csv.gz --fraction 0.05
```

`load_msr` expands each request into the 4 KiB pages it covers. To use a
trace in the main experiment, add it to `workload_set("test")` in `traces.py`.

## Limitations

- All workloads are synthetic or program-derived until an MSR trace is added.
- Decision times are measured in Python, not in a kernel. A C version of
  `LearnedPolicy._evict` would give real overhead numbers; the notes leave
  this as an open item after phase 6.
- The model is trained once offline. Periodic retraining on recent history
  (the stretch goal) is not implemented.
