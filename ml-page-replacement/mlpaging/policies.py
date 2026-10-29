"""Page replacement policies behind one interface.

Every policy implements ``reset(capacity, trace)`` and ``access(page, t)``,
which returns True on a hit. On a fault with memory full the policy picks a
victim in ``_evict``; the simulator times that call as the decision cost.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import OrderedDict, deque
from time import perf_counter_ns

import numpy as np


class Policy:
    name = "policy"
    # bytes of bookkeeping per resident page in a C implementation
    metadata_bytes = 0

    def reset(self, capacity: int, trace: np.ndarray) -> None:
        self.c = capacity
        self.decision_ns = 0
        self.decisions = 0
        self.last_victim = None     # page evicted by the latest access, if any

    def access(self, page: int, t: int) -> bool:  # pragma: no cover
        raise NotImplementedError

    def _timed_evict(self, t: int):
        t0 = perf_counter_ns()
        v = self._evict(t)
        self.decision_ns += perf_counter_ns() - t0
        self.decisions += 1
        self.last_victim = v
        return v


class FIFO(Policy):
    name = "FIFO"
    metadata_bytes = 8          # next pointer in the queue

    def reset(self, capacity, trace):
        super().reset(capacity, trace)
        self.q, self.res = deque(), set()

    def _evict(self, t):
        v = self.q.popleft()
        self.res.remove(v)
        return v

    def access(self, page, t):
        if page in self.res:
            return True
        if len(self.res) >= self.c:
            self._timed_evict(t)
        self.q.append(page)
        self.res.add(page)
        return False


class LRU(Policy):
    name = "LRU"
    metadata_bytes = 16         # prev/next pointers

    def reset(self, capacity, trace):
        super().reset(capacity, trace)
        self.od = OrderedDict()

    def _evict(self, t):
        return self.od.popitem(last=False)[0]

    def access(self, page, t):
        od = self.od
        if page in od:
            od.move_to_end(page)
            return True
        if len(od) >= self.c:
            self._timed_evict(t)
        od[page] = None
        return False


class LFU(Policy):
    """In-memory frequency; ties go to the least recently used page."""
    name = "LFU"
    metadata_bytes = 16         # count + heap slot

    def reset(self, capacity, trace):
        super().reset(capacity, trace)
        self.freq, self.last, self.heap = {}, {}, []

    def _evict(self, t):
        heap, freq, last = self.heap, self.freq, self.last
        while True:
            f, l, p = heapq.heappop(heap)
            if freq.get(p) == f and last[p] == l:
                del freq[p], last[p]
                return p

    def access(self, page, t):
        freq = self.freq
        if page in freq:
            freq[page] += 1
            self.last[page] = t
            heapq.heappush(self.heap, (freq[page], t, page))
            if len(self.heap) > 8 * self.c:
                self.heap = [(f, self.last[p], p) for p, f in freq.items()]
                heapq.heapify(self.heap)
            return True
        if len(freq) >= self.c:
            self._timed_evict(t)
        freq[page] = 1
        self.last[page] = t
        heapq.heappush(self.heap, (1, t, page))
        return False


class CLOCK(Policy):
    """Second chance: a reference bit per frame and a sweeping hand."""
    name = "CLOCK"
    metadata_bytes = 1          # one reference bit, rounded up

    def reset(self, capacity, trace):
        super().reset(capacity, trace)
        self.frames = [-1] * capacity
        self.ref = [0] * capacity
        self.where = {}
        self.hand = 0
        self.used = 0

    def _evict(self, t):
        frames, ref = self.frames, self.ref
        while ref[self.hand]:
            ref[self.hand] = 0
            self.hand = (self.hand + 1) % self.c
        v = frames[self.hand]
        del self.where[v]
        return v

    def access(self, page, t):
        idx = self.where.get(page)
        if idx is not None:
            self.ref[idx] = 1
            return True
        if self.used < self.c:
            idx = self.used
            self.used += 1
        else:
            self._timed_evict(t)
            idx = self.hand
            self.hand = (self.hand + 1) % self.c
        self.frames[idx] = page
        self.ref[idx] = 1
        self.where[page] = idx
        return False


class ARC(Policy):
    """Adaptive Replacement Cache (Megiddo and Modha, FAST 2003, Fig. 4)."""
    name = "ARC"
    metadata_bytes = 24         # list pointers + list id; ghosts cost the same

    def reset(self, capacity, trace):
        super().reset(capacity, trace)
        self.t1, self.t2, self.b1, self.b2 = (OrderedDict() for _ in range(4))
        self.p = 0.0

    def _replace(self, x):
        t1, t2 = self.t1, self.t2
        if t1 and ((x in self.b2 and len(t1) == self.p) or len(t1) > self.p):
            v = t1.popitem(last=False)[0]
            self.b1[v] = None
        else:
            v = t2.popitem(last=False)[0]
            self.b2[v] = None
        return v

    def _evict(self, t):
        return self._replace(self._x)

    def access(self, x, t):
        t1, t2, b1, b2, c = self.t1, self.t2, self.b1, self.b2, self.c
        if x in t1:
            del t1[x]
            t2[x] = None
            return True
        if x in t2:
            t2.move_to_end(x)
            return True
        self._x = x
        if x in b1:
            self.p = min(c, self.p + max(len(b2) / len(b1), 1))
            self._timed_evict(t)
            del b1[x]
            t2[x] = None
            return False
        if x in b2:
            self.p = max(0.0, self.p - max(len(b1) / len(b2), 1))
            self._timed_evict(t)
            del b2[x]
            t2[x] = None
            return False
        l1 = len(t1) + len(b1)
        if l1 == c:
            if len(t1) < c:
                b1.popitem(last=False)
                self._timed_evict(t)
            else:
                t0 = perf_counter_ns()
                self.last_victim = t1.popitem(last=False)[0]
                self.decision_ns += perf_counter_ns() - t0
                self.decisions += 1
        else:
            total = l1 + len(t2) + len(b2)
            if total >= c:
                if total == 2 * c:
                    b2.popitem(last=False)
                self._timed_evict(t)
        t1[x] = None
        return False


class OPT(Policy):
    """Belady's MIN: evict the page whose next use is furthest away."""
    name = "OPT"
    metadata_bytes = 8          # next-use time (offline only)

    def reset(self, capacity, trace):
        super().reset(capacity, trace)
        self.nxt = next_use(trace)
        self.cur = {}            # page -> its next use time
        self.heap = []

    def _evict(self, t):
        heap, cur = self.heap, self.cur
        while True:
            neg, p = heapq.heappop(heap)
            if cur.get(p) == -neg:
                del cur[p]
                return p

    def access(self, page, t):
        n = int(self.nxt[t])
        hit = page in self.cur
        if not hit and len(self.cur) >= self.c:
            self._timed_evict(t)
        self.cur[page] = n
        heapq.heappush(self.heap, (-n, page))
        if len(self.heap) > 8 * self.c:
            self.heap = [(-v, p) for p, v in self.cur.items()]
            heapq.heapify(self.heap)
        return hit


def next_use(trace: np.ndarray) -> np.ndarray:
    """nxt[t] = index of the next access to trace[t], or len(trace) + page."""
    n = trace.size
    nxt = np.empty(n, dtype=np.int64)
    seen = {}
    pages = trace.tolist()
    for t in range(n - 1, -1, -1):
        p = pages[t]
        # never used again: distinct far-future times keep ties deterministic
        nxt[t] = seen.get(p, n + p)
        seen[p] = t
    return nxt


class LeCaR(Policy):
    """LeCaR (Vietri et al., HotStorage 2018): regret-weighted LRU vs LFU.

    Evicted pages go to the history of the expert that chose them. A miss on
    a page in LRU's history means LRU was wrong, so LFU's weight grows, and
    the other way round. Rewards decay with time spent in the history.
    """
    name = "LeCaR"
    metadata_bytes = 32         # recency pointers + count + heap slot + history

    def __init__(self, learning_rate=0.45, seed=0):
        self.lr = learning_rate
        self.seed = seed

    def reset(self, capacity, trace):
        super().reset(capacity, trace)
        self.rng = random.Random(self.seed)
        self.rec = OrderedDict()
        self.freq, self.last, self.heap = {}, {}, []
        self.h_lru, self.h_lfu = OrderedDict(), OrderedDict()   # page -> (time, freq)
        self.w = [0.5, 0.5]
        self.discount = 0.005 ** (1 / capacity)

    def _lfu_victim(self):
        heap, freq, last = self.heap, self.freq, self.last
        while True:
            f, l, p = heap[0]
            if freq.get(p) == f and last[p] == l:
                return p
            heapq.heappop(heap)

    def _evict(self, t):
        use_lru = self.rng.random() < self.w[0]
        v = next(iter(self.rec)) if use_lru else self._lfu_victim()
        hist = self.h_lru if use_lru else self.h_lfu
        hist[v] = (t, self.freq[v])
        if len(hist) > self.c:
            hist.popitem(last=False)
        del self.rec[v], self.freq[v], self.last[v]
        return v

    def access(self, page, t):
        if page in self.freq:
            self.rec.move_to_end(page)
            self.freq[page] += 1
            self.last[page] = t
            heapq.heappush(self.heap, (self.freq[page], t, page))
            return True
        f0 = 1
        for hist, other in ((self.h_lru, 1), (self.h_lfu, 0)):
            if page in hist:
                te, f = hist.pop(page)
                reward = self.discount ** (t - te)
                self.w[other] *= math.exp(self.lr * reward)
                s = self.w[0] + self.w[1]
                self.w = [self.w[0] / s, self.w[1] / s]
                f0 = f + 1
                break
        if len(self.freq) >= self.c:
            self._timed_evict(t)
        self.rec[page] = None
        self.freq[page] = f0
        self.last[page] = t
        heapq.heappush(self.heap, (f0, t, page))
        if len(self.heap) > 8 * self.c:
            self.heap = [(f, self.last[p], p) for p, f in self.freq.items()]
            heapq.heapify(self.heap)
        return False


class SIEVE(Policy):
    """SIEVE (Zhang et al., NSDI 2024): FIFO order, visited bit, moving hand."""
    name = "SIEVE"
    metadata_bytes = 17         # two pointers + visited bit

    def reset(self, capacity, trace):
        super().reset(capacity, trace)
        self.newer, self.older, self.visited = {}, {}, {}
        self.head = self.tail = self.hand = None   # head = newest

    def _unlink(self, p):
        n, o = self.newer.pop(p), self.older.pop(p)
        if n is None:
            self.head = o
        else:
            self.older[n] = o
        if o is None:
            self.tail = n
        else:
            self.newer[o] = n
        del self.visited[p]

    def _evict(self, t):
        obj = self.hand if self.hand is not None else self.tail
        while self.visited[obj]:
            self.visited[obj] = False
            obj = self.newer[obj] if self.newer[obj] is not None else self.tail
        self.hand = self.newer[obj]
        self._unlink(obj)
        return obj

    def access(self, page, t):
        if page in self.visited:
            self.visited[page] = True
            return True
        if len(self.visited) >= self.c:
            self._timed_evict(t)
        self.visited[page] = False
        self.older[page], self.newer[page] = self.head, None
        if self.head is not None:
            self.newer[self.head] = page
        self.head = page
        if self.tail is None:
            self.tail = page
        return False


class S3FIFO(Policy):
    """S3-FIFO (Yang et al., SOSP 2023): small, main and ghost FIFO queues.

    New pages enter the small queue (10% of memory). A page re-accessed while
    in the small queue moves to the main queue; others are demoted quickly to
    the ghost queue, which remembers them so a quick return goes straight to
    main. The main queue gives pages with a non-zero counter another lap.
    """
    name = "S3-FIFO"
    metadata_bytes = 9          # queue pointer + 2-bit counter, rounded

    def reset(self, capacity, trace):
        super().reset(capacity, trace)
        self.s_cap = max(1, capacity // 10)
        self.m_cap = capacity - self.s_cap
        self.small, self.main, self.ghost = deque(), deque(), OrderedDict()
        self.freq, self.where = {}, {}

    def _evict_main(self):
        while True:
            p = self.main.popleft()
            if self.freq[p] > 0:
                self.freq[p] -= 1
                self.main.append(p)
            else:
                del self.freq[p], self.where[p]
                return p

    def _evict(self, t):
        if len(self.small) >= self.s_cap or not self.main:
            while self.small:
                p = self.small.popleft()
                if self.freq[p] >= 1:
                    self.freq[p] = 0
                    self.main.append(p)
                    self.where[p] = "M"
                    if len(self.main) > self.m_cap:
                        return self._evict_main()
                else:
                    del self.freq[p], self.where[p]
                    self.ghost[p] = None
                    if len(self.ghost) > self.m_cap:
                        self.ghost.popitem(last=False)
                    return p
        return self._evict_main()

    def access(self, page, t):
        if page in self.where:
            self.freq[page] = min(self.freq[page] + 1, 3)
            return True
        if len(self.where) >= self.c:
            self._timed_evict(t)
        self.freq[page] = 0
        if page in self.ghost:
            del self.ghost[page]
            self.main.append(page)
            self.where[page] = "M"
        else:
            self.small.append(page)
            self.where[page] = "S"
        return False


class SampledLRU(Policy):
    """Evict the least recently used of k randomly sampled resident pages.

    This is the learned policy's fallback on its own, so the comparison shows
    what the model adds beyond sampling.
    """
    metadata_bytes = 8          # last access time

    def __init__(self, k=16, seed=0):
        self.k = k
        self.seed = seed
        self.name = f"SampledLRU-{k}"

    def reset(self, capacity, trace):
        super().reset(capacity, trace)
        self.rng = random.Random(self.seed)
        self.resident, self.pos, self.last = [], {}, {}

    def _evict(self, t):
        res = self.resident
        k = min(self.k, len(res))
        idx = self.rng.sample(range(len(res)), k)
        v = min((res[i] for i in idx), key=self.last.__getitem__)
        i = self.pos.pop(v)
        tail = res.pop()
        if tail != v:
            res[i] = tail
            self.pos[tail] = i
        del self.last[v]
        return v

    def access(self, page, t):
        if page in self.pos:
            self.last[page] = t
            return True
        if len(self.resident) >= self.c:
            self._timed_evict(t)
        self.pos[page] = len(self.resident)
        self.resident.append(page)
        self.last[page] = t
        return False


def baselines(seed=0) -> list[Policy]:
    return [FIFO(), LRU(), LFU(), CLOCK(), ARC(), LeCaR(seed=seed), SIEVE(),
            S3FIFO(), SampledLRU(16, seed=seed), OPT()]
