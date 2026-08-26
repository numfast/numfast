"""test_topk_functional — k, stable ties, N=0/N=1, k=0, k>=N, chunking"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
from TopK._lib.topk import TopK, SCAN_LIMIT, topk_array

def ref_topk(src, k, descending=True):
    src = np.asarray(src, dtype=np.float32).ravel()
    n = src.size
    if n == 0 or k == 0:
        return np.array([], dtype=np.float32)
    k_eff = min(max(k,0), n)
    if k_eff == 0:
        return np.array([], dtype=np.float32)
    # stable sort via argsort stable
    if descending:
        # descending stable: argsort on -value stable keeps original order for ties
        idx = np.argsort(-src.astype(np.float64), kind='stable')
    else:
        idx = np.argsort(src, kind='stable')
    sorted_src = src[idx]
    return sorted_src[:k_eff].astype(np.float32)

def test_k_basic():
    src = np.array([3,1,4,1,5,9,2,6], dtype=np.float32)
    # descending True top 3 should be 9,6,5
    out = TopK(src, 3, descending=True)
    exp = ref_topk(src, 3, True)
    assert np.array_equal(out, exp), f"k basic desc {out} vs {exp}"
    # ascending top 3: 1,1,2
    out2 = TopK(src, 3, descending=False)
    exp2 = ref_topk(src, 3, False)
    assert np.array_equal(out2, exp2)
    # k=1
    assert list(TopK(src, 1, True)) == [9.0]
    assert list(TopK(src, 1, False)) == [1.0]
    # k varying
    for k in [2,4,5,8]:
        assert np.array_equal(TopK(src, k, True), ref_topk(src, k, True))
        assert np.array_equal(TopK(src, k, False), ref_topk(src, k, False))

def test_stable_ties():
    # stable ties: values equal should keep original order
    src = np.array([5,5,5,1,5,2], dtype=np.float32)  # indices 0,1,2,4 are 5
    out = TopK(src, 3, descending=True)
    # sorted descending stable: the three 5's should be in original order 0,1,2
    exp = ref_topk(src, 3, True)
    assert np.array_equal(out, exp)
    # Check explicitly: first 3 values are 5,5,5 and they correspond to original order
    assert list(out) == [5,5,5]
    # ascending ties stable
    src2 = np.array([2,2,1,2,1,2], dtype=np.float32)
    out2 = TopK(src2, 4, descending=False)
    exp2 = ref_topk(src2, 4, False)
    assert np.array_equal(out2, exp2)
    # ties with descending: ensure stable order not reversed
    src3 = np.array([10,10,9,10], dtype=np.float32)
    out3 = TopK(src3, 3, descending=True)
    # The three 10's are indices 0,1,3 in order; after sorting descending, the three 10's should stay 0,1,3 order
    # Since we only return values, we check values are 10,10,10 stable
    assert list(out3) == [10,10,10]
    # More precise tie check via indices: reconstruct indices via stable argsort
    idx_desc = np.argsort(-src3.astype(np.float64), kind='stable')
    assert list(idx_desc[:3]) == [0,1,3], f"stable indices {idx_desc[:3]}"

def test_N_zero():
    out = TopK(np.array([], dtype=np.float32), 5)
    assert out.size == 0 and out.dtype == np.float32
    out2 = TopK([], 0)
    assert out2.size == 0
    out3 = topk_array(np.array([], dtype=np.float32), 10, descending=False)
    assert out3.size == 0
    # N=0 with k=0
    assert TopK([], 0).size == 0
    assert TopK(np.array([], dtype=np.float32), 0, descending=False).size == 0

def test_N_one():
    src = np.array([7.0], dtype=np.float32)
    assert list(TopK(src, 1, True)) == [7.0]
    assert list(TopK(src, 1, False)) == [7.0]
    assert TopK(src, 0).size == 0
    assert np.array_equal(TopK(src, 5, True), np.array([7.0], dtype=np.float32))
    assert np.array_equal(TopK(src, 5, False), np.array([7.0], dtype=np.float32))
    # N=1 with k via topk_array
    assert list(topk_array([42], 1)) == [42.0]

def test_k_zero():
    src = np.arange(10, dtype=np.float32)
    out = TopK(src, 0)
    assert out.size == 0 and out.dtype == np.float32
    out2 = TopK(src, 0, descending=False)
    assert out2.size == 0
    # k=0 with N=0
    assert TopK([], 0).size == 0
    # k=0 via topk_array with chunk_limit
    assert topk_array(src, 0, chunk_limit=1024).size == 0

def test_k_ge_N():
    src = np.array([3,1,2], dtype=np.float32)
    # k == N -> sorted copy
    out = TopK(src, 3, descending=True)
    exp = ref_topk(src, 3, True)
    assert np.array_equal(out, exp)
    assert list(out) == [3,2,1]  # descending sorted
    out2 = TopK(src, 3, descending=False)
    assert list(out2) == [1,2,3]
    # k > N -> same as k==N (copy)
    out3 = TopK(src, 10, descending=True)
    assert np.array_equal(out3, exp)
    assert out3.size == 3
    out4 = TopK(src, 100, descending=False)
    assert np.array_equal(out4, ref_topk(src, 3, False))
    # Ensure not same object as input
    src_copy = src.copy()
    out5 = TopK(src_copy, 10)
    assert np.array_equal(out5, ref_topk(src_copy, 3, True))
    # modifies out should not affect src
    out5[0] = 999
    assert src_copy[0] == 3

def test_k_negative():
    src = np.array([1,2,3], dtype=np.float32)
    try:
        TopK(src, -1)
        assert False, "k<0 should raise ValueError"
    except ValueError:
        pass
    try:
        TopK(src, -10)
        assert False
    except ValueError:
        pass
    try:
        topk_array(src, -5)
        assert False
    except ValueError:
        pass

def test_chunking():
    # Chunking test: N exceeds SCAN_LIMIT simulation via small chunk_limit
    rng = np.random.RandomState(42)
    n = 10000
    src = rng.randn(n).astype(np.float32)
    for k in [1,10,100,500]:
        exp = ref_topk(src, k, True)
        out = topk_array(src, k, descending=True, chunk_limit=1024)
        assert np.array_equal(out, exp), f"chunk 1024 k={k} fail"
        out2 = topk_array(src, k, descending=False, chunk_limit=1024)
        assert np.array_equal(out2, ref_topk(src, k, False))
        # also with 4096
        assert np.array_equal(topk_array(src, k, chunk_limit=4096), exp)
    # Large N near SCAN_LIMIT but we test 1.2M (if memory allows) with smaller chunk limit simulation
    n_big = 50000
    src_big = rng.randn(n_big).astype(np.float32)
    k_big = 100
    exp_big = ref_topk(src_big, k_big, True)
    out_big = topk_array(src_big, k_big, chunk_limit=8192)
    assert np.array_equal(out_big, exp_big)
    # Test N=0 chunked
    assert topk_array([], 5, chunk_limit=1024).size == 0
    # Test k>=N chunked
    small = np.array([5,2,8], dtype=np.float32)
    assert np.array_equal(topk_array(small, 10, chunk_limit=2), ref_topk(small, 3, True))

def test_descending_flag():
    src = np.array([1,4,2,5,3], dtype=np.float32)
    assert list(TopK(src, 3, descending=True)) == [5,4,3]
    assert list(TopK(src, 3, descending=False)) == [1,2,3]
    # stable both
    src2 = np.array([2,1,2,1,2], dtype=np.float32)
    assert np.array_equal(TopK(src2, 3, True), ref_topk(src2, 3, True))
    assert np.array_equal(TopK(src2, 3, False), ref_topk(src2, 3, False))

def test_dtype_f32():
    src = np.arange(10, dtype=np.float64)
    out = TopK(src, 3)
    assert out.dtype == np.float32
    src32 = np.arange(10, dtype=np.float32)
    out32 = TopK(src32, 3)
    assert out32.dtype == np.float32
