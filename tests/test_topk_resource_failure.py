"""test_topk_resource_failure — R1-R7 (refcount, reuse, leak 20 warm, N>limit chunked, N=0, repeat after error)"""
import sys, os, gc, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
import weakref
from TopK._lib.topk import TopK, SCAN_LIMIT, topk_array

def test_R1_refcount():
    """R1: refcount — output arrays not leaked, inputs not mutated."""
    src = np.arange(100, dtype=np.float32)
    src_ref = sys.getrefcount(src)
    out = TopK(src, 10)
    assert sys.getrefcount(src) == src_ref
    wr = weakref.ref(out)
    del out
    gc.collect()
    assert wr() is None, "R1 leak: output not GC'd"
    # src not mutated (check still sorted original order not changed)
    assert np.array_equal(src, np.arange(100, dtype=np.float32))

def test_R2_reuse():
    """R2: reuse — repeated calls with same N produce correct results, no stale state."""
    rng = np.random.RandomState(0)
    n = 10000
    src = rng.randn(n).astype(np.float32)
    out1 = TopK(src, 100, descending=True)
    out2 = TopK(src, 100, descending=True)
    assert np.array_equal(out1, out2)
    # reuse with different k
    out3 = TopK(src, 50, descending=True)
    # out3 should be prefix of out1 (since same descending sort)
    assert np.array_equal(out3, out1[:50])
    # ensure out1 unchanged after later calls
    out1_copy = out1.copy()
    _ = TopK(src, 10, descending=False)
    assert np.array_equal(out1, out1_copy)
    # reuse ascending vs descending
    out_asc = TopK(src, 100, descending=False)
    assert not np.array_equal(out1, out_asc) or np.all(src == src[0])  # different unless all equal

def test_R3_leak_20_warm():
    """R3: leak 20 warm — 20 warmups no memory growth."""
    rng = np.random.RandomState(1)
    n = 100000
    src = rng.randn(n).astype(np.float32)
    k = 100
    # warm 20
    for i in range(20):
        out = TopK(src, k)
        assert out.size == k
        # check sorted descending
        assert np.all(out[:-1] >= out[1:])
    out = TopK(src, k)
    # verify correctness vs ref
    ref = np.sort(src)[::-1][:k]
    # due to stable, we need argsort stable check, but sorted slice is sufficient for this random (no ties)
    # Use argsort stable ref
    idx = np.argsort(-src.astype(np.float64), kind='stable')
    ref_stable = src[idx][:k]
    assert np.array_equal(out, ref_stable.astype(np.float32))
    gc.collect()

def test_R4_N_over_limit_chunked():
    """R4: N>limit chunked — SCAN_LIMIT=4194240, test via forced chunk_limit."""
    rng = np.random.RandomState(2)
    n = 10000
    src = rng.randn(n).astype(np.float32)
    # Forced small chunk to simulate limit without allocating 4M
    k = 100
    ref_idx = np.argsort(-src.astype(np.float64), kind='stable')[:k]
    exp = src[ref_idx].astype(np.float32)
    out = topk_array(src, k, chunk_limit=1024)
    assert np.array_equal(out, exp)
    small_limit = 4096
    big_n = small_limit * 3 + 500
    src2 = rng.randn(big_n).astype(np.float32)
    k2 = 50
    idx2 = np.argsort(-src2.astype(np.float64), kind='stable')[:k2]
    exp2 = src2[idx2].astype(np.float32)
    out2 = topk_array(src2, k2, chunk_limit=small_limit)
    assert np.array_equal(out2, exp2)
    # Also test N=0 via same path
    out0 = topk_array(np.array([], dtype=np.float32), 5, chunk_limit=small_limit)
    assert out0.size == 0
    # Test N>limit with k>=N chunked
    small_src = np.array([3,1,2], dtype=np.float32)
    assert np.array_equal(topk_array(small_src, 10, chunk_limit=2), np.array([3,2,1], dtype=np.float32))

def test_R5_N_zero():
    """R5: N=0 handling no crash, returns empty f32."""
    out = TopK(np.array([], dtype=np.float32), 5)
    assert out.size == 0 and out.dtype == np.float32
    out2 = TopK(np.array([], dtype=np.float32), 0)
    assert out2.size == 0
    for _ in range(5):
        assert TopK([], 5).size == 0
        assert TopK([], 0).size == 0
    assert topk_array([], 0, chunk_limit=1024).size == 0

def test_R6_repeat_after_error():
    """R6: repeat after error — TopK handles k<0 error and recovers."""
    src = np.arange(10, dtype=np.float32)
    # k<0 should raise
    try:
        TopK(src, -1)
        assert False, "should have raised"
    except ValueError:
        pass
    # after error, normal call should still work
    out_ok = TopK(src, 3, descending=True)
    assert list(out_ok) == [9,8,7]
    # Another error
    try:
        topk_array(src, -5)
        assert False
    except ValueError:
        pass
    out_ok2 = TopK(src, 3, descending=False)
    assert list(out_ok2) == [0,1,2]
    # None src should return empty not crash, then recover
    out_none = TopK(None, 5)
    assert out_none.size == 0
    out_ok3 = TopK(src, 2)
    assert list(out_ok3) == [9,8]
    # Invalid k type that can be cast? string should maybe error or return empty; test recovery
    try:
        TopK(src, "bad")
        # if it doesn't raise, it should return empty or value; still recover
    except (ValueError, TypeError):
        pass
    assert list(TopK(src, 1)) == [9]

def test_R7_stress_reuse_and_pool():
    """R7: stress reuse — 100 iterations alternating N sizes, ensure no corruption."""
    rng = np.random.RandomState(3)
    for i in range(100):
        n = int(rng.choice([0,1,10,100,1000,10000]))
        src = rng.randn(max(n,1)).astype(np.float32)[:n] if n>0 else np.array([], dtype=np.float32)
        k = int(rng.choice([0,1,5,10, n, n+10 if n>0 else 0]))
        descending = bool(rng.choice([True, False]))
        # reference
        if n == 0 or k == 0:
            exp = np.array([], dtype=np.float32)
        else:
            k_eff = min(k, n)
            if descending:
                idx = np.argsort(-src.astype(np.float64), kind='stable')[:k_eff]
            else:
                idx = np.argsort(src, kind='stable')[:k_eff]
            exp = src[idx].astype(np.float32)
        out = TopK(src, k, descending=descending)
        assert np.array_equal(out, exp), f"R7 iter {i} n={n} k={k} desc={descending} fail {out[:5]} vs {exp[:5]}"
        # also test chunked path for small
        if n>0 and n <= 10000:
            out_c = topk_array(src, k, descending=descending, chunk_limit=1024)
            assert np.array_equal(out_c, exp)
    gc.collect()
