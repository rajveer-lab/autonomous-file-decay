"""The learned policy: features, OPT-labelled training data, tiny models.

Offline, training traces are replayed under OPT. At each OPT eviction a few
resident pages are sampled; each gets a feature vector built only from its
past accesses and the label 1 if OPT knows it is not needed within the next
H accesses. A tiny classifier learns P(not needed soon | features).

Online, on each fault the policy samples k resident pages, scores them and
evicts the highest score. If even the best score is below a threshold the
model is unsure, and it evicts the least recently used page of the sample.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import OrderedDict

import numpy as np

from .policies import Policy, next_use

FEATURES = ["age", "count", "gap1", "gap2", "gap3", "mean_gap", "age_over_gap", "has_gap"]
GROUPS = {
    "recency": ["age"],
    "frequency": ["count"],
    "gap history": ["gap1", "gap2", "gap3", "mean_gap", "age_over_gap", "has_gap"],
}
# per tracked page: last access, count, three gaps, gap sum (4-byte ints)
METADATA_BYTES = 24


class Meta:
    """Per-page history, kept for every page seen (like Linux shadow entries)."""

    def __init__(self, n_pages: int):
        self.last = [0] * n_pages
        self.count = [0] * n_pages
        self.g1 = [0] * n_pages
        self.g2 = [0] * n_pages
        self.g3 = [0] * n_pages
        self.gsum = [0] * n_pages

    def touch(self, p: int, t: int) -> None:
        if self.count[p]:
            gap = t - self.last[p]
            self.g3[p], self.g2[p], self.g1[p] = self.g2[p], self.g1[p], gap
            self.gsum[p] += gap
        self.count[p] += 1
        self.last[p] = t

    def row(self, p: int, t: int, c: int) -> list:
        """One feature row in plain Python (the online path; same as features())."""
        age = t - self.last[p]
        f_age = math.log1p(age / c)
        n_gaps = self.count[p] - 1
        gaps = [math.log1p(g / c) if n_gaps > j else f_age
                for j, g in enumerate((self.g1[p], self.g2[p], self.g3[p]))]
        if n_gaps > 0:
            mean = self.gsum[p] / n_gaps
            return [f_age, math.log1p(self.count[p]), *gaps, math.log1p(mean / c),
                    math.log1p(age) - math.log1p(mean), 1.0]
        return [f_age, math.log1p(self.count[p]), *gaps, math.log1p(age / c), 0.0, 0.0]

    def features(self, pages, t: int, c: int) -> np.ndarray:
        """Feature rows for ``pages`` at time t, scaled by memory size c."""
        last = np.array([self.last[p] for p in pages], dtype=np.float64)
        cnt = np.array([self.count[p] for p in pages], dtype=np.float64)
        g = np.array([[self.g1[p], self.g2[p], self.g3[p]] for p in pages], dtype=np.float64)
        gsum = np.array([self.gsum[p] for p in pages], dtype=np.float64)
        age = t - last
        f_age = np.log1p(age / c)
        n_gaps = cnt - 1
        has = n_gaps > 0
        mean = np.where(has, gsum / np.maximum(n_gaps, 1), age)
        gaps = np.log1p(g / c)
        for j in range(3):                      # gaps not seen yet: use the age
            gaps[:, j] = np.where(n_gaps > j, gaps[:, j], f_age)
        return np.column_stack([
            f_age, np.log1p(cnt), gaps,
            np.log1p(mean / c),
            np.where(has, np.log1p(age) - np.log1p(mean), 0.0),
            has.astype(np.float64),
        ])


# ------------------------------------------------------------------ models

class LogReg:
    """Logistic regression, one weight per feature, fitted by Newton steps."""
    kind = "logreg"

    def __init__(self, l2: float = 1e-3):
        self.l2 = l2

    def fit(self, X, y, cols=None):
        self.cols = list(range(X.shape[1])) if cols is None else cols
        X = X[:, self.cols]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        w = np.zeros(Z.shape[1])
        for _ in range(30):
            p = 1 / (1 + np.exp(-Z @ w))
            grad = Z.T @ (p - y) / len(y) + self.l2 * np.r_[0, w[1:]]
            H = (Z * (p * (1 - p))[:, None]).T @ Z / len(y) + self.l2 * np.eye(len(w))
            step = np.linalg.solve(H, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        # fold the standardisation into the weights
        self.w = w[1:] / sd
        self.b = w[0] - (w[1:] * mu / sd).sum()
        return self

    @property
    def n_params(self):
        return len(self.w) + 1

    def predict(self, X):
        return 1 / (1 + np.exp(-(X[:, self.cols] @ self.w + self.b)))

    def score(self, row):
        z = self.b
        for c, w in zip(self.cols, self.w.tolist() if hasattr(self.w, "tolist") else self.w):
            z += w * row[c]
        return 1 / (1 + math.exp(-z)) if z > -700 else 0.0

    def compile(self):
        self.w = list(map(float, self.w))
        self.b = float(self.b)
        return self

    def to_dict(self):
        return {"kind": "logreg", "cols": list(self.cols), "weights": list(map(float, self.w)),
                "bias": float(self.b)}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.cols = d.get("cols", list(range(len(d["weights"]))))
        m.w, m.b = list(map(float, d["weights"])), float(d["bias"])
        return m


class Tree:
    """A shallow CART tree with variance-reduction splits.

    On 0/1 labels the splits are the same as Gini splits and a leaf holds
    P(label = 1). On real-valued targets it is a regression tree.
    """
    kind = "tree"

    def __init__(self, depth: int = 4, min_leaf: int = 200, bins: int = 32):
        self.depth, self.min_leaf, self.bins = depth, min_leaf, bins

    def fit(self, X, y, cols=None):
        self.cols = list(range(X.shape[1])) if cols is None else cols
        X = X[:, self.cols]
        self.nodes = []          # (feature, threshold, left, right) or (-1, prob)
        self._grow(X, y, 0)
        return self

    def _grow(self, X, y, d):
        idx = len(self.nodes)
        self.nodes.append(None)
        best = None
        if d < self.depth and len(y) >= 2 * self.min_leaf and y.var() > 0:
            n, sy, syy = len(y), y.sum(), (y * y).sum()
            parent = syy / n - (sy / n) ** 2
            for f in range(X.shape[1]):
                xs = X[:, f]
                cuts = np.unique(np.quantile(xs, np.linspace(0, 1, self.bins + 1)[1:-1]))
                for c in cuts:
                    left = xs <= c
                    nl = int(left.sum())
                    nr = n - nl
                    if nl < self.min_leaf or nr < self.min_leaf:
                        continue
                    yl = y[left]
                    sl, sll = yl.sum(), (yl * yl).sum()
                    vl = sll / nl - (sl / nl) ** 2
                    vr = (syy - sll) / nr - ((sy - sl) / nr) ** 2
                    gain = parent - (nl * vl + nr * vr) / n
                    if best is None or gain > best[0]:
                        best = (gain, f, c)
        if best is None or best[0] <= 1e-7:
            self.nodes[idx] = (-1, float(y.mean()) if len(y) else 0.5)
            return idx
        _, f, c = best
        left = X[:, f] <= c
        li = self._grow(X[left], y[left], d + 1)
        ri = self._grow(X[~left], y[~left], d + 1)
        self.nodes[idx] = (f, float(c), li, ri)
        return idx

    @property
    def n_params(self):
        # a split stores feature + threshold, a leaf one probability
        return sum(2 if n[0] >= 0 else 1 for n in self.nodes)

    def score(self, row):
        nodes, cols = self.nodes, self.cols
        n = nodes[0]
        while n[0] >= 0:
            n = nodes[n[2] if row[cols[n[0]]] <= n[1] else n[3]]
        return n[1]

    def compile(self):
        return self

    def to_dict(self):
        return {"kind": "tree", "cols": list(self.cols), "nodes": [list(n) for n in self.nodes]}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.cols = d["cols"]
        m.nodes = [tuple(n) for n in d["nodes"]]
        return m

    def predict(self, X):
        X = X[:, self.cols]
        out = np.empty(len(X))
        for i, row in enumerate(X):
            n = self.nodes[0]
            while n[0] >= 0:
                n = self.nodes[n[2] if row[n[0]] <= n[1] else n[3]]
            out[i] = n[1]
        return out


# ------------------------------------------------------------ training data

def opt_dataset(trace: np.ndarray, c: int, k: int = 16,
                max_rows: int = 20_000, seed: int = 0):
    """Replay ``trace`` under OPT with ``c`` frames and record sampled pages.

    Returns (X, D): feature rows built only from accesses before the
    eviction time, and D = (next use - now) / c, which only OPT knows.
    Labels are made from D: ``D > horizon`` for the classifier, or
    ``log1p(D)`` for the regression tree.
    """
    rng = random.Random(seed)
    nxt = next_use(trace)
    pages = trace.tolist()
    n = len(pages)
    meta = Meta(int(trace.max()) + 1)
    resident, pos, cur = [], {}, {}
    heap = []
    # sample roughly max_rows rows spread over the whole trace
    evictions_est = max(1, n // 2)
    stride = max(1, (evictions_est * min(k, c)) // max_rows)
    Xs, ys = [], []
    ev = 0
    for t, p in enumerate(pages):
        if p not in pos and len(resident) >= c:
            if ev % stride == 0:
                sample = rng.sample(resident, min(k, len(resident)))
                Xs.append(meta.features(sample, t, c))
                ys.append([min(cur[q] - t, n) / c for q in sample])
            ev += 1
            while True:
                neg, v = heapq.heappop(heap)
                if cur.get(v) == -neg:
                    break
            del cur[v]
            i = pos.pop(v)
            tail = resident.pop()
            if tail != v:
                resident[i] = tail
                pos[tail] = i
        if p not in pos:
            pos[p] = len(resident)
            resident.append(p)
        cur[p] = int(nxt[t])
        heapq.heappush(heap, (-cur[p], p))
        if len(heap) > 8 * c:
            heap = [(-v, q) for q, v in cur.items()]
            heapq.heapify(heap)
        meta.touch(p, t)
    if not Xs:
        return np.empty((0, len(FEATURES))), np.empty(0)
    return np.vstack(Xs), np.concatenate([np.array(y) for y in ys])


# ------------------------------------------------------------ online policy

class LearnedPolicy(Policy):
    """Sampled learned eviction with two safety nets.

    1. Confidence fallback: if the best score in the sample is below the
       threshold, evict the least recently used page instead.
    2. Regret guard (the LeCaR mechanism [14], with the model and LRU as the
       two experts): pages evicted by each expert go into that expert's
       history of size C. A fault on a page in the model's history means the
       model evicted something still needed, so LRU gains weight, and the
       other way round. Each eviction follows an expert chosen by weight.
       With ``guard=False`` every eviction follows the model.
    """
    metadata_bytes = METADATA_BYTES + 16    # features + recency-list pointers

    def __init__(self, model, k=16, threshold=0.5, seed=0, name="Learned",
                 guard=True, learning_rate=0.45):
        self.model, self.k, self.threshold, self.seed = model, k, threshold, seed
        self.name, self.guard, self.lr = name, guard, learning_rate

    def reset(self, capacity, trace):
        super().reset(capacity, trace)
        self.rng = random.Random(self.seed)
        self.meta = Meta(int(trace.max()) + 1)
        self.resident, self.pos = [], {}
        self.order = OrderedDict()               # resident pages, LRU first
        self.h_model, self.h_lru = OrderedDict(), OrderedDict()
        self.w_model = 1.0 if not self.guard else 0.5
        self.discount = 0.005 ** (1 / capacity)
        self.fallbacks = 0
        self.model_evictions = 0

    def _model_victim(self, t):
        res, k = self.resident, min(self.k, len(self.resident))
        sample = [res[i] for i in self.rng.sample(range(len(res)), k)]
        row, score, c = self.meta.row, self.model.score, self.c
        scores = [score(row(p, t, c)) for p in sample]
        j = max(range(k), key=scores.__getitem__)
        if scores[j] < self.threshold:
            self.fallbacks += 1
            return None
        return sample[j]

    def _evict(self, t):
        v = None
        if not self.guard or self.rng.random() < self.w_model:
            v = self._model_victim(t)
            hist = self.h_model
        if v is None:
            v = next(iter(self.order))
            hist = self.h_lru
        else:
            self.model_evictions += 1
        if self.guard:
            hist[v] = t
            if len(hist) > self.c:
                hist.popitem(last=False)
        del self.order[v]
        i = self.pos.pop(v)
        tail = self.resident.pop()
        if tail != v:
            self.resident[i] = tail
            self.pos[tail] = i
        return v

    def _regret(self, page, t):
        for hist, model_was_wrong in ((self.h_model, True), (self.h_lru, False)):
            if page in hist:
                reward = self.discount ** (t - hist.pop(page))
                w_m, w_l = self.w_model, 1 - self.w_model
                if model_was_wrong:
                    w_l *= math.exp(self.lr * reward)
                else:
                    w_m *= math.exp(self.lr * reward)
                self.w_model = w_m / (w_m + w_l)
                return

    def access(self, page, t):
        if page in self.pos:
            self.order.move_to_end(page)
            self.meta.touch(page, t)
            return True
        if self.guard:
            self._regret(page, t)
        if len(self.resident) >= self.c:
            self._timed_evict(t)
        self.pos[page] = len(self.resident)
        self.resident.append(page)
        self.order[page] = None
        self.meta.touch(page, t)
        return False


def model_bytes(model) -> int:
    return 4 * model.n_params      # float32 parameters



def model_from_dict(d):
    return (Tree if d["kind"] == "tree" else LogReg).from_dict(d)
