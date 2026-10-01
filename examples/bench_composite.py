#!/usr/bin/env python3
"""Benchmark composite_tuple_index: before/after optimization."""
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app-builder"))

from Drivers.CPU._lib.groupindex import composite_tuple_index

SEED = 42
N = 1_000_000

def make_data(n, num_cols, card, rng):
    """Create num_cols int32 columns with cardinality ~card."""
    cols = []
    for i in range(num_cols):
        cols.append(rng.integers(0, card, size=n, dtype=np.int32))
    return cols

def bench_one(cols, reps=3):
    """Best-of-reps timing with stage breakdown."""
    import tracemalloc
    # warmup
    composite_tuple_index(cols)
    
    tracemalloc.start()
    t0 = time.perf_counter()
    ukeys, inverse, counts = composite_tuple_index(cols)
    elapsed = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return elapsed, peak, ukeys, inverse, counts

def main():
    rng = np.random.default_rng(SEED)
    results = {}
    
    for D in [100, 10_000, N]:
        cols = make_data(N, 4, D, rng)
        elapsed, peak, ukeys, inverse, counts = bench_one(cols)
        
        # Verify correctness
        assert len(ukeys) == len(counts), f"ukeys/counts length mismatch: {len(ukeys)} vs {len(counts)}"
        assert int(counts.sum()) == N, f"counts sum {counts.sum()} != {N}"
        assert inverse.shape == (N,), f"inverse shape {inverse.shape} != ({N},)"
        assert inverse.min() >= 0, f"inverse min {inverse.min()} < 0"
        assert inverse.max() < len(ukeys), f"inverse max {inverse.max()} >= {len(ukeys)}"
        
        # Verify sorted ukeys
        for i in range(1, len(ukeys)):
            assert ukeys[i] >= ukeys[i-1], f"ukeys not sorted at {i}: {ukeys[i]} < {ukeys[i-1]}"
        
        # Verify inverse mapping
        for idx in range(0, N, max(1, N // 1000)):
            row = tuple(c[idx] for c in cols)
            assert ukeys[inverse[idx]] == row, f"inverse mismatch at {idx}"
        
        results[D] = elapsed, peak, len(ukeys)
        print(f"D={D:>8}: {elapsed:>8.2f} ms, peak={peak/1024/1024:.1f} MiB, groups={len(ukeys)}")
    
    return results

if __name__ == "__main__":
    main()
