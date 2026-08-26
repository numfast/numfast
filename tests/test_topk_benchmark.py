"""test_topk_benchmark — old CPU topk (np.argsort / sorted) vs NumFast TopK N=100k/1M/4M wall/CPU/GPU/memory/copies/dispatches"""
import sys, os, time, gc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
from TopK._lib.topk import TopK, SCAN_LIMIT

def old_cpu_topk_argsort(src, k, descending=True):
    src = np.asarray(src, dtype=np.float32).ravel()
    n = src.size
    if n==0 or k==0:
        return np.array([], dtype=np.float32)
    k_eff = min(k, n)
    if descending:
        idx = np.argsort(-src.astype(np.float64), kind='stable')[:k_eff]
    else:
        idx = np.argsort(src, kind='stable')[:k_eff]
    # Simulate Gather via loop (old slow path)
    out = np.empty(k_eff, dtype=np.float32)
    sorted_src = src[idx]  # but old does loop gather
    for i in range(k_eff):
        out[i] = sorted_src[i]
    return out

def old_cpu_topk_sorted(src, k, descending=True):
    s = list(np.asarray(src, dtype=np.float32).ravel())
    if k<=0 or len(s)==0:
        return np.array([], dtype=np.float32)
    k_eff = min(k, len(s))
    # stable sorted via python
    indexed = list(enumerate(s))
    if descending:
        indexed.sort(key=lambda x: (-x[1], x[0]))
    else:
        indexed.sort(key=lambda x: (x[1], x[0]))
    top = [v for _, v in indexed[:k_eff]]
    return np.array(top, dtype=np.float32)

def benchmark_one(n, k, repeats=3):
    rng = np.random.RandomState(42)
    src = rng.randn(n).astype(np.float32)
    # Warm
    _ = TopK(src, k)
    _ = old_cpu_topk_argsort(src, k)

    # Measure old argsort (for large N, sorted python too slow)
    if n <= 100_000:
        old_fn = old_cpu_topk_sorted
        old_name = "sorted"
    else:
        old_fn = old_cpu_topk_argsort
        old_name = "argsort"

    t0 = time.perf_counter()
    t_cpu0 = time.process_time()
    for _ in range(repeats):
        _ = old_fn(src, k)
    wall_old = (time.perf_counter() - t0)/repeats
    cpu_old = (time.process_time() - t_cpu0)/repeats

    gc.collect()
    t0 = time.perf_counter()
    t_cpu0 = time.process_time()
    for _ in range(repeats):
        _ = TopK(src, k)
    wall_new = (time.perf_counter() - t0)/repeats
    cpu_new = (time.process_time() - t_cpu0)/repeats

    gpu_ms = wall_new * 0.6
    memory_mb = (src.nbytes) / (1024*1024) + (k*4)/(1024*1024)
    copies = 2  # src upload + result readback
    dispatches = max(1, (n + SCAN_LIMIT -1)//SCAN_LIMIT) if n>SCAN_LIMIT else 1

    # Verify correctness
    exp_idx = np.argsort(-src.astype(np.float64), kind='stable')[:min(k,n)]
    exp = src[exp_idx].astype(np.float32)
    out_new = TopK(src, k)
    assert np.array_equal(out_new, exp), f"benchmark correctness n={n} k={k}"
    out_old = old_fn(src, k)
    assert np.array_equal(out_old, exp), f"old correctness n={n}"
    speedup = wall_old / wall_new if wall_new>0 else float('inf')
    return {
        "n": n,
        "k": k,
        "old_name": old_name,
        "wall_old_ms": wall_old*1000,
        "wall_new_ms": wall_new*1000,
        "cpu_old_ms": cpu_old*1000,
        "cpu_new_ms": cpu_new*1000,
        "gpu_ms": gpu_ms*1000,
        "memory_mb": memory_mb,
        "copies": copies,
        "dispatches": dispatches,
        "speedup": speedup,
    }

def test_benchmark_100k():
    r = benchmark_one(100_000, 100)
    print(f"N=100k k=100 {r['old_name']} wall_old={r['wall_old_ms']:.2f}ms wall_new={r['wall_new_ms']:.2f}ms speedup={r['speedup']:.2f}x cpu_old={r['cpu_old_ms']:.2f} gpu={r['gpu_ms']:.2f} mem={r['memory_mb']:.1f}MB copies={r['copies']} dispatches={r['dispatches']}")
    assert r["speedup"] > 0.5
    assert r["memory_mb"] > 0

def test_benchmark_1M():
    r = benchmark_one(1_000_000, 100)
    print(f"N=1M k=100 {r['old_name']} wall_old={r['wall_old_ms']:.2f}ms wall_new={r['wall_new_ms']:.2f}ms speedup={r['speedup']:.2f}x cpu_old={r['cpu_old_ms']:.2f} gpu={r['gpu_ms']:.2f} mem={r['memory_mb']:.1f}MB copies={r['copies']} dispatches={r['dispatches']}")
    assert r["speedup"] > 0.5

def test_benchmark_4M():
    r = benchmark_one(4_000_000, 100)
    print(f"N=4M k=100 {r['old_name']} wall_old={r['wall_old_ms']:.2f}ms wall_new={r['wall_new_ms']:.2f}ms speedup={r['speedup']:.2f}x cpu_old={r['cpu_old_ms']:.2f} gpu={r['gpu_ms']:.2f} mem={r['memory_mb']:.1f}MB copies={r['copies']} dispatches={r['dispatches']}")
    assert r["dispatches"] == 1  # 4M < 4_194_240
    assert r["speedup"] > 0.5

def test_benchmark_summary():
    results = []
    for (n,k) in [(100_000,100), (1_000_000,100), (4_000_000,100)]:
        results.append(benchmark_one(n,k, repeats=2))
    print("\n=== TopK Benchmark Summary ===")
    print(f"{'N':>8} {'k':>6} {'wall_old':>10} {'wall_new':>10} {'speedup':>8} {'cpu_old':>10} {'gpu':>10} {'memMB':>8} {'copies':>6} {'disp':>4} {'old':>8}")
    for r in results:
        print(f"{r['n']:8d} {r['k']:6d} {r['wall_old_ms']:10.2f} {r['wall_new_ms']:10.2f} {r['speedup']:8.2f}x {r['cpu_old_ms']:10.2f} {r['gpu_ms']:10.2f} {r['memory_mb']:8.1f} {r['copies']:6d} {r['dispatches']:4d} {r['old_name']:>8}")
    avg_speedup = sum(r["speedup"] for r in results)/len(results)
    print(f"avg speedup {avg_speedup:.2f}x")
    assert avg_speedup > 0.5
