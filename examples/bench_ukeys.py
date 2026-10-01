#!/usr/bin/env python3
"""Micro-benchmark different ukeys construction strategies."""
import time
import numpy as np

N = 1_000_000
NG = 1_000_000  # worst case: all unique
NC = 4

rng = np.random.default_rng(42)
cols = [rng.integers(0, 10000, size=N, dtype=np.int32) for _ in range(NC)]
order = np.lexsort(tuple(cols[::-1]))
new_group = np.ones(N, dtype=bool)  # all unique for max ng
starts = np.arange(NG)
idx = order[starts]

def strategy_parts_zip():
    """Original: per-col tolist + zip + tuple."""
    parts = [c[idx].tolist() for c in cols]
    return [tuple(v) for v in zip(*parts)]

def strategy_parts_zip_nolist():
    """No tolist: zip numpy arrays directly."""
    parts = [c[idx] for c in cols]
    return [tuple(v) for v in zip(*parts)]

def strategy_column_stack():
    """column_stack + row iteration."""
    mat = np.column_stack([c[idx] for c in cols])
    return [tuple(row) for row in mat]

def strategy_rec_fromarrays():
    """np.rec.fromarrays + iteration."""
    dt = np.dtype([(f'k{i}', cols[i].dtype) for i in range(NC)])
    rec = np.rec.fromarrays([c[idx] for c in cols], dtype=dt)
    return [tuple(row) for row in rec]

def strategy_manual_loop():
    """Manual Python loop."""
    ca = [c[idx] for c in cols]
    return [tuple(a[i] for a in ca) for i in range(NG)]

strategies = [
    ("parts_zip", strategy_parts_zip),
    ("parts_zip_nolist", strategy_parts_zip_nolist),
    ("column_stack", strategy_column_stack),
    ("rec_fromarrays", strategy_rec_fromarrays),
    ("manual_loop", strategy_manual_loop),
]

for name, fn in strategies:
    # warmup
    fn()
    best = float('inf')
    for _ in range(5):
        t0 = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        best = min(best, elapsed)
    # verify
    assert len(result) == NG
    print(f"{name:>25}: {best:>8.2f}ms")
