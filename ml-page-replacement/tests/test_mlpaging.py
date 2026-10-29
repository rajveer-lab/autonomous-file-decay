import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpaging import learned as L  # noqa: E402
from mlpaging import policies as P  # noqa: E402
from mlpaging import traces as T  # noqa: E402
from mlpaging.sim import simulate  # noqa: E402


def all_policies():
    rng = np.random.default_rng(0)
    X = rng.random((400, len(L.FEATURES)))
    y = (rng.random(400) < 0.5).astype(float)
    return P.baselines() + [P.SampledLRU(4), L.LearnedPolicy(L.LogReg().fit(X, y).compile(), k=4),
                            L.LearnedPolicy(L.Tree(min_leaf=20).fit(X, y).compile(), k=4),
                            L.LearnedPolicy(L.LogReg().fit(X, y).compile(), k=4, guard=False)]


def test_textbook_string():
    tb = T.textbook().pages
    assert simulate(P.FIFO(), tb, 3)["faults"] == 15
    assert simulate(P.LRU(), tb, 3)["faults"] == 12
    assert simulate(P.OPT(), tb, 3)["faults"] == 9


def brute_force_opt(trace, c):
    """Fewest faults over every possible sequence of eviction choices."""
    best = [len(trace)]

    def go(i, res, faults):
        if faults >= best[0]:
            return
        if i == len(trace):
            best[0] = faults
            return
        p = trace[i]
        if p in res:
            go(i + 1, res, faults)
        elif len(res) < c:
            go(i + 1, res | {p}, faults + 1)
        else:
            for v in res:
                go(i + 1, (res - {v}) | {p}, faults + 1)

    go(0, frozenset(), 0)
    return best[0]


@pytest.mark.parametrize("seed", range(12))
def test_opt_is_optimal_on_small_traces(seed):
    rng = np.random.default_rng(seed)
    tr = rng.integers(0, 5, 12)
    for c in (2, 3):
        assert simulate(P.OPT(), tr, c)["faults"] == brute_force_opt(tr.tolist(), c)


@pytest.mark.parametrize("seed", range(5))
def test_no_policy_beats_opt(seed):
    tr = T.zipf(n_pages=300, length=4000, seed=seed).pages
    for c in (10, 40):
        opt = simulate(P.OPT(), tr, c)["faults"]
        for pol in all_policies():
            assert simulate(pol, tr, c)["faults"] >= opt, pol.name


def test_compulsory_misses_only_when_everything_fits():
    tr = T.mixed(loop_pages=50, zipf_pages=100, length=3000, seed=1).pages
    distinct = len(set(tr.tolist()))
    for pol in all_policies():
        assert simulate(pol, tr, distinct)["faults"] == distinct, pol.name


def test_one_frame_faults_on_every_change():
    tr = np.array([1, 1, 2, 3, 3, 3, 1, 2, 2, 4])
    changes = 1 + int((tr[1:] != tr[:-1]).sum())
    for pol in all_policies():
        assert simulate(pol, tr, 1)["faults"] == changes, pol.name


def test_arc_invariants():
    arc = P.ARC()
    tr = T.phase_change(n_phases=3, ws=200, n_pages=900, length=6000, seed=2).pages
    c = 64
    arc.reset(c, tr)
    for t, p in enumerate(tr.tolist()):
        arc.access(p, t)
        t1, t2, b1, b2 = map(len, (arc.t1, arc.t2, arc.b1, arc.b2))
        assert t1 + t2 <= c
        assert t1 + b1 <= c
        assert t1 + t2 + b1 + b2 <= 2 * c
        assert 0 <= arc.p <= c


def test_features_use_only_the_past():
    a = np.array([1, 2, 3, 1, 2, 4, 1, 5, 6, 7], dtype=np.int64)
    b = a.copy()
    b[6:] = [9, 9, 9, 9]                 # same past, different future
    feats = []
    for tr in (a, b):
        meta = L.Meta(10)
        for t, p in enumerate(tr[:6].tolist()):
            meta.touch(p, t)
        feats.append(meta.features([1, 2, 3, 4], 6, 3))
    assert np.array_equal(feats[0], feats[1])


def test_row_matches_vectorised_features():
    tr = T.zipf(n_pages=500, length=5000, seed=4).pages
    meta = L.Meta(int(tr.max()) + 1)
    for t, p in enumerate(tr.tolist()):
        meta.touch(p, t)
    pages = sorted(set(tr.tolist()))[:100]
    A = meta.features(pages, len(tr), 50)
    B = np.array([meta.row(p, len(tr), 50) for p in pages])
    assert np.allclose(A, B)


def test_opt_dataset_labels_match_next_use():
    tr = T.zipf(n_pages=400, length=6000, seed=5).pages
    c = 30
    X, D = L.opt_dataset(tr, c, max_rows=2000)
    assert len(X) == len(D) > 0
    assert X.shape[1] == len(L.FEATURES)
    assert (D > 0).all()                       # next use is always in the future
    assert (D * c <= len(tr)).all()


def test_models_score_like_predict():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, len(L.FEATURES)))
    y = (X[:, 0] + 0.3 * rng.normal(size=2000) > 0).astype(float)
    for M in (L.LogReg, L.Tree):
        m = M().fit(X, y)
        pv = m.predict(X[:200])
        m.compile()
        assert np.allclose(pv, [m.score(list(r)) for r in X[:200]])


def test_learned_policy_is_deterministic():
    tr = T.scan_hot(hot=100, length=8000, scan_every=500, scan_len=200, seed=1).pages
    X, D = L.opt_dataset(tr, 40)
    m = L.LogReg().fit(X, (D > 1).astype(float)).compile()
    r1 = simulate(L.LearnedPolicy(m, seed=3), tr, 40)["faults"]
    r2 = simulate(L.LearnedPolicy(m, seed=3), tr, 40)["faults"]
    assert r1 == r2


def test_msr_loader(tmp_path):
    f = tmp_path / "web_0.csv"
    f.write_text("128166372003061629,web,0,Read,8192,4096,100\n"
                 "128166372003061630,web,0,Write,0,12288,90\n"
                 "128166372003061631,web,0,Read,8192,512,80\n")
    tr = T.load_msr(str(f))
    # page 2; pages 0,1,2; page 2  -> dense ids in first-seen order
    assert tr.pages.tolist() == [0, 1, 2, 0, 0]
    assert T.load_msr(str(f), reads_only=True).pages.tolist() == [0, 0]


def test_generators_are_reproducible():
    for gen in (T.loop, T.zipf, T.phase_change, T.scan_hot, T.mixed):
        assert np.array_equal(gen(seed=7).pages, gen(seed=7).pages)
    assert len(T.matmul(n=32, rows=2)) > 0
    assert len(T.mergesort(n=1 << 10)) > 0
    assert len(T.bfs(n=500)) > 0


def test_guard_keeps_learned_close_to_lru_on_merge_sort():
    train = T.zipf(n_pages=2000, length=20000, seed=1).pages
    X, D = L.opt_dataset(train, 60)
    m = L.Tree().fit(X, np.log1p(D)).compile()
    tr = T.mergesort(n=1 << 11, elem=256, seed=2).pages
    c = 20
    lru = simulate(P.LRU(), tr, c)["faults"]
    guarded = simulate(L.LearnedPolicy(m, threshold=0.0), tr, c)["faults"]
    assert guarded <= 1.05 * lru
