#!/usr/bin/env python3
"""Micro-benchmark: stacked diffs np.any vs loop for new_group."""
import time
import numpy as np

N = 1_000_000
NC = 4

rng = np.random.default_rng(42)
cols = [rng.integers(0, 10000, size=N, dtype=np.int32) for _ in range(NC)]
order = np.lexsort(tuple(cols[::-1]))

def loop_approach():
    new_group = np.zeros(N, dtype=bool)
    new_group[0] = True
    for c in cols:
        sc = c[order]
        new_group[1:] |= (sc[1:] != sc[:-1])
    return new_group

def stacked_approach():
    diffs = np.empty((N - 1, NC), dtype=bool)
    for j, c in enumerate(cols):
        sc = c[order]
        diffs[:, j] = sc[1:] != sc[:-1]
    changed = np.any(diffs, axis=1)
    new_group = np.empty(N, dtype=bool)
    new_group[0] = True
    new_group[1:] = changed
    return new_group

for name, fn in [("loop", loop_approach), ("stacked_any", stacked_approach)]:
    fn()  # warmup
    best = float('inf')
    for _ in range(5):
        t0 = time.perf_counter()
        r = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        best = min(best, elapsed)
    assert np.array_equal(loop_approach(), stacked_approach())
    print(f"{name:>15}: {best:>6.2f}ms")
