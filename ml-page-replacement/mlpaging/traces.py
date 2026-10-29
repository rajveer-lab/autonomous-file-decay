"""Page reference traces: the textbook string, synthetic patterns,
program-derived traces and a loader for the MSR Cambridge block traces.

A trace is a 1-D ``numpy.int64`` array of page numbers. Generators relabel
pages to dense ids ``0..U-1`` so policies can use arrays indexed by page.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass, field

import numpy as np

PAGE = 4096


@dataclass
class Trace:
    name: str
    family: str
    pages: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def distinct(self) -> int:
        return int(np.unique(self.pages).size)

    def __len__(self) -> int:
        return int(self.pages.size)


def dense(pages: np.ndarray) -> np.ndarray:
    """Relabel page numbers to 0..U-1 in order of first appearance."""
    _, first, inv = np.unique(pages, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inv].astype(np.int64)


def collapse(pages: np.ndarray) -> np.ndarray:
    """Drop immediate repeats (a reference to the page just referenced).

    Such references are hits under every policy, so page-reference strings
    are usually recorded without them.
    """
    if pages.size == 0:
        return pages
    keep = np.ones(pages.size, dtype=bool)
    keep[1:] = pages[1:] != pages[:-1]
    return pages[keep]


# ---------------------------------------------------------------- textbook

TEXTBOOK = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]


def textbook() -> Trace:
    return Trace("textbook", "textbook", np.array(TEXTBOOK, dtype=np.int64))


# ---------------------------------------------------------------- synthetic

def _zipf_ranks(rng, n, alpha, size):
    w = 1.0 / np.arange(1, n + 1) ** alpha
    w /= w.sum()
    return rng.choice(n, size=size, p=w)


def loop(n_pages=1500, length=120_000, noise=0.02, seed=0) -> Trace:
    """A loop over more pages than fit in memory (LRU's worst case)."""
    rng = np.random.default_rng(seed)
    pages = np.arange(length) % n_pages
    flip = rng.random(length) < noise
    pages[flip] = rng.integers(0, n_pages, flip.sum())
    return Trace(f"loop-{n_pages}-s{seed}", "loop", dense(pages),
                 dict(n_pages=n_pages, noise=noise, seed=seed))


def zipf(n_pages=10_000, length=150_000, alpha=0.9, seed=0) -> Trace:
    """Skewed popularity: a few pages take most accesses."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pages)
    pages = perm[_zipf_ranks(rng, n_pages, alpha, length)]
    return Trace(f"zipf-{alpha}-s{seed}", "zipf", dense(pages),
                 dict(n_pages=n_pages, alpha=alpha, seed=seed))


def phase_change(n_phases=6, ws=900, n_pages=8_000, length=150_000,
                 alpha=0.6, seed=0) -> Trace:
    """The working set moves to a new region every phase."""
    rng = np.random.default_rng(seed)
    per = length // n_phases
    out = []
    for _ in range(n_phases):
        region = rng.choice(n_pages, ws, replace=False)
        out.append(region[_zipf_ranks(rng, ws, alpha, per)])
    pages = np.concatenate(out)
    return Trace(f"phase-{ws}-s{seed}", "phase", dense(pages),
                 dict(n_phases=n_phases, ws=ws, n_pages=n_pages, seed=seed))


def scan_hot(hot=600, length=150_000, scan_every=4_000, scan_len=1_500,
             alpha=0.8, seed=0) -> Trace:
    """A hot set, interrupted by one-time sequential scans of fresh pages."""
    rng = np.random.default_rng(seed)
    out, next_cold, n = [], hot, 0
    while n < length:
        k = min(scan_every, length - n)
        out.append(_zipf_ranks(rng, hot, alpha, k))
        n += k
        if n >= length:
            break
        k = min(scan_len, length - n)
        out.append(np.arange(next_cold, next_cold + k))
        next_cold += k
        n += k
    pages = np.concatenate(out)
    return Trace(f"scanhot-{hot}-s{seed}", "scanhot", dense(pages),
                 dict(hot=hot, scan_every=scan_every, scan_len=scan_len, seed=seed))


def mixed(loop_pages=1200, zipf_pages=6_000, length=150_000, loop_share=0.5,
          alpha=0.9, seed=0) -> Trace:
    """Two programs interleaved: one loops, one has skewed popularity."""
    rng = np.random.default_rng(seed)
    is_loop = rng.random(length) < loop_share
    n_loop = int(is_loop.sum())
    pages = np.empty(length, dtype=np.int64)
    pages[is_loop] = np.arange(n_loop) % loop_pages
    perm = rng.permutation(zipf_pages)
    pages[~is_loop] = loop_pages + perm[_zipf_ranks(rng, zipf_pages, alpha, length - n_loop)]
    return Trace(f"mixed-{loop_share}-s{seed}", "mixed", dense(pages),
                 dict(loop_pages=loop_pages, zipf_pages=zipf_pages,
                      loop_share=loop_share, seed=seed))


# ---------------------------------------------------------- program-derived

def _pages(addr: np.ndarray) -> np.ndarray:
    return collapse(addr // PAGE)


def matmul(n=256, rows=1, elem=32) -> Trace:
    """Naive i-j-k matrix multiply C = A x B on row-major elements.

    Records A[i][k], B[k][j], C[i][j] for the first ``rows`` values of i.
    """
    size = n * n * elem
    base_a, base_b, base_c = 0, size, 2 * size
    i = np.repeat(np.arange(rows), n * n)
    j = np.tile(np.repeat(np.arange(n), n), rows)
    k = np.tile(np.arange(n), rows * n)
    a = base_a + (i * n + k) * elem
    b = base_b + (k * n + j) * elem
    c = base_c + (i * n + j) * elem
    addr = np.stack([a, b, c], axis=1).ravel()
    return Trace(f"matmul-{n}", "matmul", dense(_pages(addr)), dict(n=n, rows=rows, elem=elem))


def mergesort(n=1 << 12, elem=256, seed=0) -> Trace:
    """Bottom-up merge sort between an array and a temp buffer."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 1 << 30, n)
    src_base, dst_base = 0, n * elem
    addr = []
    width = 1
    src = data.copy()
    dst = np.empty_like(src)
    while width < n:
        for lo in range(0, n, 2 * width):
            mid, hi = min(lo + width, n), min(lo + 2 * width, n)
            left, right = src[lo:mid], src[mid:hi]
            # order in which the merge reads elements: positions from each run
            merged_from_right = np.argsort(np.concatenate([left, right]), kind="stable") >= left.size
            li = np.cumsum(~merged_from_right) - 1
            ri = np.cumsum(merged_from_right) - 1
            read = np.where(merged_from_right, mid + ri, lo + li)
            write = np.arange(lo, hi)
            addr.append(np.stack([src_base + read * elem, dst_base + write * elem], axis=1).ravel())
            dst[lo:hi] = np.sort(np.concatenate([left, right]), kind="stable")
        src, dst = dst, src
        src_base, dst_base = dst_base, src_base
        width *= 2
    return Trace(f"mergesort-{n}", "mergesort", dense(_pages(np.concatenate(addr))),
                 dict(n=n, elem=elem, seed=seed))


def bfs(n=15_000, degree=6, elem=16, seed=0) -> Trace:
    """Breadth-first search over a random graph stored in CSR form.

    Touches the offsets array, the edge array, the visited array and the
    queue, in the order a textbook BFS would.
    """
    rng = np.random.default_rng(seed)
    m = n * degree
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    offsets = np.searchsorted(src, np.arange(n + 1))
    base_off, base_edge = 0, (n + 1) * elem
    base_vis = base_edge + m * elem
    base_q = base_vis + n * elem
    visited = np.zeros(n, dtype=bool)
    addr = []
    head = tail = 0
    queue = np.empty(n, dtype=np.int64)
    for start in rng.permutation(n):
        if visited[start]:
            continue
        visited[start] = True
        queue[tail] = start
        tail += 1
        while head < tail:
            u = queue[head]
            addr.append(np.array([base_q + head * elem, base_off + u * elem,
                                  base_off + (u + 1) * elem]))
            head += 1
            nbrs = dst[offsets[u]:offsets[u + 1]]
            if nbrs.size == 0:
                continue
            e_addr = base_edge + np.arange(offsets[u], offsets[u + 1]) * elem
            v_addr = base_vis + nbrs * elem
            addr.append(np.stack([e_addr, v_addr], axis=1).ravel())
            new = nbrs[~visited[nbrs]]
            new = np.unique(new)
            visited[new] = True
            if new.size:
                addr.append(base_q + np.arange(tail, tail + new.size) * elem)
                queue[tail:tail + new.size] = new
                tail += new.size
    return Trace(f"bfs-{n}", "bfs", dense(_pages(np.concatenate(addr))),
                 dict(n=n, degree=degree, seed=seed))


# ------------------------------------------------------------ MSR Cambridge

def load_msr(path: str, page_size: int = PAGE, reads_only: bool = False,
             limit: int | None = None) -> Trace:
    """Load an MSR Cambridge block trace (CSV, optionally .gz).

    Rows are ``Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime``.
    Each request is expanded to every page it covers. Download the traces
    from the SNIA IOTTA repository (MSR Cambridge set) under its licence.
    """
    opener = gzip.open if path.endswith(".gz") else open
    pages = []
    with opener(path, "rt", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6 or not row[4].isdigit():
                continue
            if reads_only and row[3].strip().lower() != "read":
                continue
            off, size = int(row[4]), int(row[5])
            first, last = off // page_size, (off + max(size, 1) - 1) // page_size
            pages.extend(range(first, last + 1))
            if limit is not None and len(pages) >= limit:
                pages = pages[:limit]
                break
    name = path.rsplit("/", 1)[-1].split(".")[0]
    return Trace(f"msr-{name}", "msr", dense(np.array(pages, dtype=np.int64)),
                 dict(path=path, reads_only=reads_only))


# ------------------------------------------------------------------ splits

def workload_set(split: str) -> list[Trace]:
    """Train, validation and test traces.

    Synthetic families appear in every split with different seeds and
    parameters. Program-derived traces are split by program so the test set
    contains a program (BFS) the model never saw.
    """
    if split == "train":
        return [
            loop(n_pages=1500, seed=1), zipf(alpha=0.9, seed=1),
            phase_change(ws=900, seed=1), scan_hot(hot=600, seed=1),
            mixed(loop_share=0.5, seed=1), matmul(n=256, rows=1, elem=32),
        ]
    if split == "val":
        return [
            loop(n_pages=1800, seed=2), zipf(alpha=0.8, seed=2),
            phase_change(ws=1100, seed=2), scan_hot(hot=800, seed=2),
            mixed(loop_share=0.4, seed=2), mergesort(n=1 << 12, elem=256, seed=2),
        ]
    if split == "test":
        return [
            loop(n_pages=2000, noise=0.05, seed=3), zipf(alpha=1.0, n_pages=12_000, seed=3),
            phase_change(ws=1300, n_phases=5, seed=3), scan_hot(hot=1000, scan_len=2500, seed=3),
            mixed(loop_share=0.6, seed=3), matmul(n=320, rows=1, elem=32),
            mergesort(n=1 << 13, elem=256, seed=3), bfs(n=15_000, elem=16, seed=3),
        ]
    raise ValueError(split)
