#!/usr/bin/env python3
"""Verify packed lexsort produces same order as original."""
import numpy as np

N = 100_000
rng = np.random.default_rng(42)
cols = [rng.integers(0, 10000, size=N, dtype=np.int32) for _ in range(4)]

# Original lexsort: lexsort((c3, c2, c1, c0)) => primary c0, secondary c1, ...
order_orig = np.lexsort(tuple(c[::-1] for c in cols))

# Packed: c0 in high bits of pack0, c1 in low bits; c2 high, c3 low
_pack0 = (cols[0].astype(np.int64) << 32) | (cols[1].astype(np.int64) & np.int64(0xFFFFFFFF))
_pack1 = (cols[2].astype(np.int64) << 32) | (cols[3].astype(np.int64) & np.int64(0xFFFFFFFF))

order_packed = np.lexsort((_pack1, _pack0))

print(f"orders match: {np.array_equal(order_orig, order_packed)}")

# Verify on actual groupby: sorted keys should match
def get_sorted_keys(order, cols, n=10):
    keys_orig = list(zip(*[c[order] for c in cols]))
    return keys_orig[:n]

print("Original first 10:", get_sorted_keys(order_orig, cols))
print("Packed   first 10:", get_sorted_keys(order_packed, cols))
