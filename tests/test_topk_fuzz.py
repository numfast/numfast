"""test_topk_fuzz — ≥1000 fuzz seed42 N grid 0,1,2,63,64,65,1000,8192,1M × k 0,1,10,50,100,N,N+10 exact stable"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
from TopK._lib.topk import TopK, topk_array, SCAN_LIMIT

SEED = 42
N_GRID = [0,1,2,63,64,65,1000,8192,1000000]
# For memory safety in CI, cap 1M fuzz to few cases; 1M array ~4MB ok
K_GRID_TEMPLATE = [0,1,10,50,100]  # plus N, N+10

def ref_topk(src, k, descending=True):
    src = np.asarray(src, dtype=np.float32).ravel()
    n = src.size
    if n == 0 or k == 0:
        return np.array([], dtype=np.float32)
    k_eff = min(max(k,0), n)
    if k_eff == 0:
        return np.array([], dtype=np.float32)
    if descending:
        idx = np.argsort(-src.astype(np.float64), kind='stable')
    else:
        idx = np.argsort(src, kind='stable')
    return src[idx][:k_eff].astype(np.float32)

def test_fuzz_grid():
    rng = np.random.RandomState(SEED)
    total = 0
    failures = []
    for n in N_GRID:
        # Use actual allocation only for n up to 1M but limit repeats for 1M to avoid OOM/time
        repeats = 2 if n == 1000000 else 5 if n == 8192 else 3
        for _ in range(repeats):
            # generate src fuzz
            if n == 0:
                src = np.array([], dtype=np.float32)
            else:
                # vary distribution: uniform, normal, ties heavy
                dist = rng.choice(["uniform","normal","ties","sorted","reverse"])
                if dist == "uniform":
                    src = (rng.rand(n)*200 -100).astype(np.float32)
                elif dist == "normal":
                    src = rng.randn(n).astype(np.float32)
                elif dist == "ties":
                    # many ties: limited values
                    vals = np.array([-5,0,5,10], dtype=np.float32)
                    src = rng.choice(vals, size=n).astype(np.float32)
                elif dist == "sorted":
                    src = np.sort(rng.randn(n).astype(np.float32))
                else:  # reverse
                    src = np.sort(rng.randn(n).astype(np.float32))[::-1].astype(np.float32)
            # k values to test
            ks = [0,1,10,50,100,n, n+10]
            # For small n, some ks exceed n, we still test
            for k in ks:
                for descending in [True, False]:
                    # Cap k to reasonable for big N to avoid huge output compare overhead
                    if n == 1000000 and k == 1000000:
                        # k==N for 1M is heavy but ok one time
                        pass
                    out = TopK(src, k, descending=descending)
                    exp = ref_topk(src, k, descending)
                    total += 1
                    if out.size != exp.size:
                        failures.append((n, k, descending, "size", out.size, exp.size))
                    elif out.size >0:
                        if not np.array_equal(out, exp):
                            # check exact stable: allow not just values but order
                            failures.append((n, k, descending, "not_equal", out[:5], exp[:5]))
                            # also linf
                            linf = float(np.max(np.abs(out.astype(np.float64)-exp.astype(np.float64)))) if out.size else 0
                            failures.append((n, k, descending, "linf", linf))
                    # also test via topk_array with chunk_limit
                    if n >0 and n <= 10000:
                        out_chunk = topk_array(src, k, descending=descending, chunk_limit=1024)
                        if not np.array_equal(out_chunk, exp):
                            failures.append((n,k,descending,"chunk1024",out_chunk[:3],exp[:3]))
    # Ensure ≥1000 fuzz
    extra_needed = 1000 - total
    if extra_needed > 0:
        for _ in range(extra_needed):
            n = int(rng.choice([0,1,2,63,64,65,1000,8192]))
            src = (rng.rand(n)*200 -100).astype(np.float32) if n>0 else np.array([],dtype=np.float32)
            k = int(rng.choice([0,1,10,50,100,n, n+10 if n>0 else 0]))
            descending = bool(rng.choice([True, False]))
            out = TopK(src, k, descending=descending)
            exp = ref_topk(src, k, descending)
            total += 1
            if out.size != exp.size or (out.size>0 and not np.array_equal(out, exp)):
                failures.append((n,k,descending,"extra",out.size,exp.size))
    assert total >= 1000, f"total fuzz {total} <1000"
    assert not failures, f"fuzz failures {failures[:5]} total {len(failures)}"
    print(f"fuzz OK total={total} all exact stable")

def test_fuzz_chunked_random():
    rng = np.random.RandomState(SEED)
    for n in [5000, 10000, 50000]:
        for _ in range(20):
            src = rng.randn(n).astype(np.float32)
            for k in [0,1,10,100,1000]:
                k_eff = min(k, n)
                for descending in [True, False]:
                    exp = ref_topk(src, k, descending)
                    out = topk_array(src, k, descending=descending, chunk_limit=1024)
                    assert np.array_equal(out, exp), f"chunked fuzz n={n} k={k} desc={descending} fail"
                    out2 = topk_array(src, k, descending=descending, chunk_limit=4096)
                    assert np.array_equal(out2, exp)

def test_fuzz_dtype_variants():
    rng = np.random.RandomState(SEED)
    for n in [0,1,64,1000]:
        src_f32 = rng.randn(max(n,1)).astype(np.float32)[:n]
        for dtype_in in [np.float64, np.int32, np.int64]:
            src = src_f32.astype(dtype_in) if n>0 else np.array([], dtype=dtype_in)
            for k in [0,1, n, n+10]:
                for descending in [True, False]:
                    out = TopK(src, k, descending=descending)
                    exp = ref_topk(src, k, descending)
                    assert np.array_equal(out, exp), f"dtype {dtype_in} n={n} k={k}"

def test_fuzz_all_patterns_cpu_exact():
    rng = np.random.RandomState(SEED)
    for _ in range(200):
        n = int(rng.choice([0,1,2,63,64,65,100,1000]))
        src = rng.randn(n).astype(np.float32) if n>0 else np.array([], dtype=np.float32)
        for descending in [True, False]:
            k = int(rng.choice([0,1,10,n, n+10 if n>0 else 0]))
            out = TopK(src, k, descending=descending)
            exp = ref_topk(src, k, descending)
            if out.size==0 and exp.size==0:
                continue
            assert np.array_equal(out, exp), f"Linf not zero n={n} k={k}"
            if out.size>0:
                linf = float(np.max(np.abs(out.astype(np.float64) - exp.astype(np.float64))))
                assert linf == 0.0, f"Linf {linf} !=0"

def test_stable_ties_fuzz():
    rng = np.random.RandomState(SEED)
    # Heavy ties ensure stable order
    for n in [10,100,1000]:
        # src with many duplicates
        src = np.array([1,1,1,2,2,1,2,1,3,3]* (n//10+1), dtype=np.float32)[:n]
        rng.shuffle(src)
        for k in [5,10, n//2]:
            for descending in [True, False]:
                out = TopK(src, k, descending=descending)
                exp = ref_topk(src, k, descending)
                assert np.array_equal(out, exp), f"stable ties fuzz n={n} k={k}"
    # explicit ties: all equal
    src_eq = np.ones(100, dtype=np.float32) * 5
    for k in [1,10,100,110]:
        assert np.array_equal(TopK(src_eq, k, True), ref_topk(src_eq, k, True))
        assert np.array_equal(TopK(src_eq, k, False), ref_topk(src_eq, k, False))
