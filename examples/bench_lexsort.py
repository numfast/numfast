#!/usr/bin/env python3
"""Micro-benchmark: pack 4xint32 → 2xint64 for faster lexsort."""
import time
import numpy as np

N = 1_000_000
NC = 4

rng = np.random.default_rng(42)
cols = [rng.integers(0, 10000, size=N, dtype=np.int32) for _ in range(NC)]

def original_lexsort():
    """4x int32 lexsort."""
    order = np.lexsort(tuple(c[::-1] for c in cols))
    return order

def packed_lexsort():
    """Pack 4xint32 → 2xint64, lexsort on 2 int64."""
    # cols[0], cols[1] → pack_i64
    lo = cols[0].astype(np.int64)
    hi = cols[1].astype(np.int64)
    pack0 = (hi << 32) | (lo & 0xFFFFFFFF)
    lo2 = cols[2].astype(np.int64)
    hi2 = cols[3].astype(np.int64)
    pack1 = (hi2 << 32) | (lo2 & 0xFFFFFFFF)
    order = np.lexsort((pack1, pack0))
    return order

for name, fn in [("4xint32_lexsort", original_lexsort), ("2xint64_packed", packed_lexsort)]:
    fn()  # warmup
    best = float('inf')
    for _ in range(5):
        t0 = time.perf_counter()
        r = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        best = min(best, elapsed)
    print(f"{name:>20}: {best:>8.2f}ms")
