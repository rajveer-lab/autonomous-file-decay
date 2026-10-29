# Results

Test traces: loop-2000-s3, zipf-1.0-s3, phase-1300-s3, scanhot-1000-s3, mixed-0.6-s3, matmul-320, mergesort-8192, bfs-15000. Memory sizes: 1%, 2%, 5%, 10% of each trace's distinct pages. Every number below comes from `results/test.csv`.

## Summary over all test traces and memory sizes

| Policy | Mean fault rate | Fault reduction vs LRU (mean) | LRU-to-OPT gap closed (mean) | Worst case vs LRU | Decision time per eviction (Python) | Metadata per page |
|---|---|---|---|---|---|---|
| FIFO | 0.574 | -4.6% | -72.4% | 28.2% | 0.18 µs | 8 B |
| LRU | 0.554 | 0.0% | 0.0% | 0.0% | 0.23 µs | 16 B |
| LFU | 0.707 | -187.9% | -10528.8% | 1481.2% | 0.64 µs | 16 B |
| CLOCK | 0.559 | -1.0% | -5.1% | 3.7% | 0.39 µs | 1 B |
| ARC | 0.513 | 6.5% | 30.9% | 1.4% | 0.56 µs | 24 B |
| LeCaR | 0.512 | 6.7% | 24.8% | 0.3% | 1.65 µs | 32 B |
| SIEVE | 0.512 | 6.5% | 22.4% | 4.3% | 0.58 µs | 17 B |
| S3-FIFO | 0.515 | 4.4% | -224.9% | 57.4% | 0.67 µs | 9 B |
| SampledLRU-16 | 0.553 | 0.0% | 1.2% | 0.2% | 8.76 µs | 8 B |
| Learned (no guard) | 0.603 | -93.6% | -8590.5% | 1373.8% | 86.89 µs | 40 B |
| Learned | 0.514 | 6.5% | 24.7% | 0.2% | 60.28 µs | 40 B |
| OPT | 0.428 | 22.0% | 100.0% | -0.7% | 0.63 µs | 8 B |

Fault reduction = (F_LRU − F_policy) / F_LRU × 100. Gap closed = (F_LRU − F_policy) / (F_LRU − F_OPT) × 100, averaged over cells where LRU and OPT differ. Worst case = the largest extra-fault percentage over LRU in any single cell. Decision time is the median time to pick a victim in this Python simulator; compare policies against each other, not against a kernel.

## Fault rate per trace

### Memory = 1% of distinct pages

| Trace | FIFO | LRU | LFU | CLOCK | ARC | LeCaR | SIEVE | S3-FIFO | SampledLRU-16 | Learned (no guard) | Learned | OPT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loop-2000-s3 | 0.999 | 0.999 | **0.992** | 0.999 | 0.996 | 0.999 | 0.993 | 0.992 | 0.999 | 0.992 | 0.999 | 0.962 |
| zipf-1.0-s3 | 0.660 | 0.613 | 0.500 | 0.627 | 0.506 | 0.501 | 0.499 | **0.499** | 0.614 | 0.508 | 0.525 | 0.423 |
| phase-1300-s3 | 0.908 | 0.900 | 0.957 | 0.904 | 0.823 | 0.868 | 0.813 | **0.806** | 0.900 | 0.814 | 0.837 | 0.654 |
| scanhot-1000-s3 | 0.555 | 0.541 | 0.485 | 0.543 | 0.489 | 0.540 | **0.482** | 0.484 | 0.541 | 0.496 | 0.541 | 0.427 |
| mixed-0.6-s3 | 0.950 | 0.940 | 0.856 | 0.944 | 0.850 | 0.870 | 0.856 | **0.848** | 0.940 | 0.858 | 0.869 | 0.814 |
| matmul-320 | 0.431 | 0.336 | 0.333 | 0.336 | 0.333 | 0.336 | 0.333 | 0.333 | 0.336 | **0.331** | 0.336 | 0.330 |
| mergesort-8192 | **0.062** | **0.062** | 0.988 | **0.062** | 0.063 | 0.063 | 0.064 | 0.098 | **0.062** | 0.921 | 0.063 | 0.062 |
| bfs-15000 | 0.599 | 0.588 | 0.970 | 0.592 | 0.555 | **0.553** | 0.575 | 0.632 | 0.588 | 0.957 | 0.575 | 0.426 |

### Memory = 2% of distinct pages

| Trace | FIFO | LRU | LFU | CLOCK | ARC | LeCaR | SIEVE | S3-FIFO | SampledLRU-16 | Learned (no guard) | Learned | OPT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loop-2000-s3 | 0.998 | 0.998 | **0.983** | 0.998 | 0.992 | 0.998 | 0.985 | 0.984 | 0.998 | 0.984 | 0.998 | 0.943 |
| zipf-1.0-s3 | 0.580 | 0.532 | 0.437 | 0.544 | 0.438 | 0.434 | 0.431 | **0.431** | 0.532 | 0.441 | 0.453 | 0.353 |
| phase-1300-s3 | 0.841 | 0.824 | 0.936 | 0.832 | 0.752 | 0.778 | 0.740 | **0.733** | 0.825 | 0.739 | 0.760 | 0.542 |
| scanhot-1000-s3 | 0.513 | 0.513 | **0.394** | 0.513 | **0.394** | 0.407 | **0.394** | **0.394** | 0.513 | 0.416 | 0.433 | 0.390 |
| mixed-0.6-s3 | 0.928 | 0.913 | 0.831 | 0.919 | 0.822 | 0.827 | 0.831 | **0.820** | 0.914 | 0.833 | 0.850 | 0.776 |
| matmul-320 | 0.380 | 0.336 | 0.333 | 0.336 | 0.333 | 0.336 | 0.333 | 0.333 | 0.336 | **0.323** | 0.336 | 0.322 |
| mergesort-8192 | **0.062** | **0.062** | 0.977 | **0.062** | 0.063 | 0.063 | 0.063 | 0.065 | **0.062** | 0.912 | 0.063 | 0.061 |
| bfs-15000 | 0.558 | 0.533 | 0.967 | 0.553 | 0.533 | **0.524** | 0.542 | 0.613 | 0.533 | 0.950 | 0.529 | 0.383 |

### Memory = 5% of distinct pages

| Trace | FIFO | LRU | LFU | CLOCK | ARC | LeCaR | SIEVE | S3-FIFO | SampledLRU-16 | Learned (no guard) | Learned | OPT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loop-2000-s3 | 0.995 | 0.995 | **0.958** | 0.995 | 0.977 | 0.995 | 0.963 | 0.961 | 0.995 | 0.959 | 0.995 | 0.905 |
| zipf-1.0-s3 | 0.468 | 0.421 | 0.345 | 0.433 | 0.349 | **0.342** | 0.344 | 0.343 | 0.422 | 0.355 | 0.365 | 0.262 |
| phase-1300-s3 | 0.690 | 0.660 | 0.888 | 0.670 | 0.612 | 0.593 | 0.605 | **0.591** | 0.661 | 0.593 | 0.607 | 0.366 |
| scanhot-1000-s3 | 0.510 | 0.489 | 0.394 | 0.502 | 0.392 | 0.395 | 0.394 | 0.392 | 0.483 | **0.392** | 0.394 | 0.390 |
| mixed-0.6-s3 | 0.893 | 0.874 | 0.795 | 0.880 | 0.783 | **0.722** | 0.795 | 0.782 | 0.874 | 0.746 | 0.738 | 0.674 |
| matmul-320 | 0.352 | 0.336 | 0.333 | 0.336 | 0.333 | 0.336 | 0.333 | 0.333 | 0.336 | **0.301** | 0.336 | 0.297 |
| mergesort-8192 | **0.062** | **0.062** | 0.943 | **0.062** | 0.063 | 0.063 | 0.063 | **0.062** | **0.062** | 0.190 | 0.063 | 0.060 |
| bfs-15000 | 0.458 | 0.436 | 0.881 | 0.437 | 0.440 | 0.438 | 0.455 | 0.474 | 0.436 | 0.881 | **0.435** | 0.256 |

### Memory = 10% of distinct pages

| Trace | FIFO | LRU | LFU | CLOCK | ARC | LeCaR | SIEVE | S3-FIFO | SampledLRU-16 | Learned (no guard) | Learned | OPT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loop-2000-s3 | 0.990 | 0.990 | 0.917 | 0.990 | 0.947 | 0.990 | 0.917 | 0.921 | 0.990 | **0.916** | 0.990 | 0.854 |
| zipf-1.0-s3 | 0.379 | 0.336 | 0.281 | 0.346 | 0.283 | 0.278 | 0.280 | **0.276** | 0.336 | 0.290 | 0.298 | 0.198 |
| phase-1300-s3 | 0.501 | 0.464 | 0.825 | 0.474 | 0.451 | **0.417** | 0.460 | 0.434 | 0.465 | 0.432 | 0.432 | 0.216 |
| scanhot-1000-s3 | 0.439 | 0.407 | 0.390 | 0.420 | 0.390 | 0.391 | 0.390 | 0.390 | 0.401 | **0.390** | 0.391 | 0.390 |
| mixed-0.6-s3 | 0.863 | 0.842 | 0.764 | 0.849 | 0.752 | 0.645 | 0.764 | 0.751 | 0.843 | 0.613 | **0.613** | 0.506 |
| matmul-320 | 0.343 | 0.336 | 0.333 | 0.336 | 0.333 | 0.336 | 0.333 | 0.333 | 0.336 | **0.261** | 0.268 | 0.254 |
| mergesort-8192 | **0.062** | **0.062** | 0.888 | **0.062** | 0.063 | 0.063 | 0.063 | **0.062** | **0.062** | 0.087 | 0.063 | 0.057 |
| bfs-15000 | 0.335 | 0.308 | 0.737 | 0.314 | 0.312 | 0.294 | 0.303 | 0.293 | 0.308 | 0.428 | **0.292** | 0.145 |

Bold marks the fewest faults among the online policies (OPT needs the future).

## Learned policy

- Model: logreg, 9 parameters (36 bytes as float32), trained on 474673 OPT-labelled rows in 0.362 s.
- Target: logreg-H2.0; sample k = 32 pages per fault; evict LRU instead when the best score is below 0.500.
- Regret guard on: the model and LRU are weighted by how often their evicted pages come back (LeCaR mechanism). `Learned (no guard)` is the same model without it.
- Chosen on validation traces only; the test traces were not used for any choice.

## Feature ablation (test traces)

| Model | Fault reduction vs LRU (mean) | Change vs full model |
|---|---|---|
| all features | 6.5% | – |
| without recency | 6.2% | -0.3 points |
| without frequency | 6.5% | +0.0 points |
| without gap history | 3.1% | -3.4 points |

## How the run went

- textbook check (3 frames): {'FIFO': 15, 'LRU': 12, 'OPT': 9}
- traces: loop-1500-s1 (120000 refs, 1500 pages), zipf-0.9-s1 (150000 refs, 9764 pages), phase-900-s1 (150000 refs, 4112 pages), scanhot-600-s1 (150000 refs, 41100 pages), mixed-0.5-s1 (150000 refs, 6961 pages), matmul-256 (196608 refs, 516 pages), loop-1800-s2 (120000 refs, 1800 pages), zipf-0.8-s2 (150000 refs, 9936 pages), phase-1100-s2 (150000 refs, 4750 pages), scanhot-800-s2 (150000 refs, 41300 pages), mixed-0.4-s2 (150000 refs, 7100 pages), mergesort-4096 (98304 refs, 512 pages), loop-2000-s3 (120000 refs, 2000 pages), zipf-1.0-s3 (150000 refs, 10732 pages), phase-1300-s3 (150000 refs, 4700 pages), scanhot-1000-s3 (150000 refs, 58500 pages), mixed-0.6-s3 (150000 refs, 6833 pages), matmul-320 (307200 refs, 806 pages), mergesort-8192 (212992 refs, 1024 pages), bfs-15000 (215907 refs, 528 pages)
- training data: 474673 OPT-labelled rows from 6 traces in 3.0s
- validation, faults / LRU (lower is better): logreg-H0.5=0.939, logreg-H1.0=0.938, logreg-H2.0=0.935, tree-H0.5=0.968, tree-H1.0=0.951, tree-H2.0=0.953, tree-regression=0.951
- chosen: logreg-H2.0, k=32, threshold=0.500 (validation faults / LRU = 0.932)
