#!/usr/bin/env python3
"""Micro-benchmark: np.unique on structured array vs lexsort+new_group."""
import time
import numpy as np

N = 1_000_000
NC = 4

rng = np.random.default_rng(42)
cols = [rng.integers(0, 10000, size=N, dtype=np.int32) for _ in range(NC)]

def lexsort_approach():
    """Current: lexsort + new_group + starts."""
    order = np.lexsort(tuple(c[::-1] for c in cols))
    new_group = np.zeros(N, dtype=bool)
    new_group[0] = True
    for c in cols:
        sc = c[order]
        new_group[1:] |= (sc[1:] != sc[:-1])
    starts = np.flatnonzero(new_group)
    ng = int(starts.size)
    idx = order[starts]
    parts = [c[idx].tolist() for c in cols]
    ukeys = [tuple(v) for v in zip(*parts)]
    gid_sorted = np.cumsum(new_group.astype(np.int64)) - np.int64(1)
    inverse = np.empty(N, dtype=np.int64)
    inverse[order] = gid_sorted
    counts = np.bincount(gid_sorted, minlength=ng).astype(np.int64)
    return ukeys, inverse, counts

def unique_structured_approach():
    """Alternative: np.unique on structured array."""
    dt = np.dtype([(f'k{i}', cols[i].dtype) for i in range(NC)])
    structured = np.empty(N, dtype=dt)
    for i, c in enumerate(cols):
        structured[f'k{i}'] = c
    unique_arr, inverse, counts = np.unique(structured, return_inverse=True, return_counts=True)
    ukeys = [tuple(row) for row in unique_arr]
    return ukeys, inverse.astype(np.int64), counts.astype(np.int64)

for name, fn in [("lexsort", lexsort_approach), ("unique_struct", unique_structured_approach)]:
    fn()  # warmup
    best = float('inf')
    for _ in range(3):
        t0 = time.perf_counter()
        r = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        best = min(best, elapsed)
    uk, inv, cnt = r
    print(f"{name:>20}: {best:>8.2f}ms  groups={len(uk)}  inv_sum={cnt.sum()}")
