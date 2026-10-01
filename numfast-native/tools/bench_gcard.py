# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Group-cardinality sweep: dense native vs numpy at fixed n (seed 42).

Usage: python tools/bench_gcard.py [--n N] [--gs "256,65536,1048576"]
g=256: sums fit L1; g=64K: L2; g=1M: 8MB sums spill past L3 per-core share.
Shows where dense scatter stops being cache-resident. Integrity exact.
"""
import ctypes
import os
import sys
import time

import numpy as np

N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 5_000_000
GS = [int(x) for x in (sys.argv[sys.argv.index("--gs") + 1] if "--gs" in sys.argv else "256,65536,1048576").split(",")]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")

d = ctypes.CDLL(DLL)
fd = d.nf_group_sum_count
fd.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
fd.restype = ctypes.c_int32

rng = np.random.default_rng(42)
vals = rng.standard_normal(N)


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def bench(fn, reps=7):
    for _ in range(2):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return med(ts)


print("n=%d seed=42 (sums table: g=256->2KB L1, 64K->512KB L2, 1M->8MB ~L3)" % N)
print("%-10s %12s %12s %8s" % ("g", "numpy_ms", "rust_ms", "speedup"))
for g in GS:
    keys = rng.integers(0, g, size=N).astype(np.int32)
    s = np.zeros(g)
    c = np.zeros(g, dtype=np.int64)
    ref_s = np.bincount(keys, weights=vals, minlength=g)
    ref_c = np.bincount(keys, minlength=g)
    tn = bench(lambda: (s.__setitem__(slice(None), np.bincount(keys, weights=vals, minlength=g)),
                        c.__setitem__(slice(None), np.bincount(keys, minlength=g))))
    tr = bench(lambda: fd(keys.ctypes.data, vals.ctypes.data, N, s.ctypes.data, c.ctypes.data, g))
    assert np.max(np.abs(s - ref_s)) == 0.0 and np.sum(c != ref_c) == 0, "integrity g=%d" % g
    print("%-10d %12.3f %12.3f %8.2f" % (g, tn, tr, tn / tr))
