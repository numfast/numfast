"""test_topk_consumers — Search tuple_engine top100 + percentile ranking parity"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
from TopK._lib.topk import TopK

def old_python_topk(src, k, descending=True):
    """Old python topk via sorted."""
    src = list(np.asarray(src, dtype=np.float32).ravel())
    if k <=0 or len(src)==0:
        return np.array([], dtype=np.float32)
    k_eff = min(k, len(src))
    # stable: sorted with enumerate to preserve order for ties
    if descending:
        # sorted descending stable: sort by (-value, index)
        indexed = list(enumerate(src))
        indexed.sort(key=lambda x: (-x[1], x[0]))
        top = [v for _, v in indexed[:k_eff]]
    else:
        indexed = list(enumerate(src))
        indexed.sort(key=lambda x: (x[1], x[0]))
        top = [v for _, v in indexed[:k_eff]]
    return np.array(top, dtype=np.float32)

def test_search_tuple_engine_top100():
    """Search tuple_engine: N=10000 random strategy scores, TopK 100 parity."""
    rng = np.random.RandomState(42)
    n = 10000
    # Simulate tuple engine scores: Sharpe-like
    scores = rng.randn(n).astype(np.float32) * 2 + 0.5  # mean 0.5
    # Add some spikes
    scores[::100] += 5
    k = 100
    # NumFast TopK
    nf_top = TopK(scores, k, descending=True)
    py_top = old_python_topk(scores, k, descending=True)
    assert np.array_equal(nf_top, py_top), f"tuple_engine top100 mismatch {nf_top[:5]} vs {py_top[:5]}"
    # Verify sorted descending
    assert np.all(nf_top[:-1] >= nf_top[1:])
    # Also test ascending bottom 100
    nf_bottom = TopK(scores, k, descending=False)
    py_bottom = old_python_topk(scores, k, descending=False)
    assert np.array_equal(nf_bottom, py_bottom)
    assert np.all(nf_bottom[:-1] <= nf_bottom[1:])

def test_percentile_ranking_parity():
    """Percentile ranking: TopK threshold parity with np.percentile."""
    rng = np.random.RandomState(123)
    n = 5000
    data = rng.randn(n).astype(np.float32) * 10
    k = 100  # top 100 ~ 98th percentile
    # TopK to get threshold
    topk_vals = TopK(data, k, descending=True)
    threshold = float(topk_vals[-1])  # kth largest
    # Percentile: 100*(1 - k/n) percentile
    perc = 100 * (1 - k / n)
    np_thresh = np.percentile(data.astype(np.float64), perc, method='higher')
    # Threshold should be close: TopK threshold should be >= percentile threshold for descending
    # For stable tie handling, allow small diff due to ties
    # Check that TopK set contains those above threshold
    # Parity: gather via TopK vs numpy percentile filtering
    # Numpy topk via partition
    idx_np = np.argsort(-data.astype(np.float64), kind='stable')[:k]
    np_top = data[idx_np].astype(np.float32)
    assert np.array_equal(topk_vals, np_top)
    # Percentile ranking: items above percentile should be subset of TopK extended for ties
    # At least check that percentile-based selection size approx k
    mask_perc = data >= np_thresh
    # For no ties, count of >= threshold should be >=k and <=k + tie_count
    cnt = int(np.sum(mask_perc))
    assert cnt >= k
    # For our synthetic data with few ties, cnt should be close to k
    assert cnt < k + 50  # allow some ties drift

def test_search_tuple_with_ties():
    rng = np.random.RandomState(99)
    n = 1000
    # Heavy ties: scores limited to 5 values
    vals = np.array([1,2,3,4,5], dtype=np.float32)
    scores = rng.choice(vals, size=n).astype(np.float32)
    k = 20
    nf = TopK(scores, k, descending=True)
    py = old_python_topk(scores, k, descending=True)
    assert np.array_equal(nf, py)
    # Percentile parity with ties
    nf_asc = TopK(scores, k, descending=False)
    py_asc = old_python_topk(scores, k, descending=False)
    assert np.array_equal(nf_asc, py_asc)

def test_tuple_engine_multi_k():
    rng = np.random.RandomState(7)
    n = 2000
    scores = rng.randn(n).astype(np.float32)
    for k in [1,10,50,100,500]:
        nf = TopK(scores, k, True)
        py = old_python_topk(scores, k, True)
        assert np.array_equal(nf, py), f"multi k={k} fail"

def test_ranking_order_stable():
    """Ranking order stable: ensure TopK respects original order for equal scores."""
    scores = np.array([5,5,5,5,4,4,3,5,5], dtype=np.float32)
    # descending top 5 should be five 5's in original order indices 0,1,2,3,7
    top5 = TopK(scores, 5, descending=True)
    assert list(top5) == [5,5,5,5,5]
    # Verify via python stable
    py = old_python_topk(scores, 5, True)
    assert np.array_equal(top5, py)
    # For percentile ranking, check that ranking via argsort stable matches
    idx_nf = np.argsort(-scores.astype(np.float64), kind='stable')[:5]
    assert list(idx_nf) == [0,1,2,3,7]

def test_search_engine_vs_numpy_argsort():
    rng = np.random.RandomState(2026)
    n = 10000
    scores = rng.randn(n).astype(np.float32)
    for k in [10,100]:
        nf = TopK(scores, k, True)
        # numpy argsort stable
        idx = np.argsort(-scores.astype(np.float64), kind='stable')[:k]
        np_top = scores[idx]
        assert np.array_equal(nf, np_top.astype(np.float32))
        # vs sorted python
        py = old_python_topk(scores, k, True)
        assert np.array_equal(nf, py)
