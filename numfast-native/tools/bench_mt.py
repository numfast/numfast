# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""MT-host ladder: chunked dense groupby over ThreadPool (seed 42).

Usage: python tools/bench_mt.py [--n N] [--t "1,2,4,8,16,32"]
Contiguous row partition; per-thread partial (sums,counts); single-thread
reduction over T*g lanes. Kernels reentrant (no statics); ctypes releases
the GIL in-call. Integrity exact vs single-thread. New file.
"""
import ctypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10_000_000
TS = [int(x) for x in (sys.argv[sys.argv.index("--t") + 1] if "--t" in sys.argv else "1,2,4,8,16,32").split(",")]
G = 256
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")

rng = np.random.default_rng(42)
keys = rng.integers(0, G, size=N).astype(np.int32)
vals = rng.standard_normal(N)
ref_s = np.bincount(keys, weights=vals, minlength=G)
ref_c = np.bincount(keys, minlength=G)

d = ctypes.CDLL(DLL)
fd = d.nf_group_sum_count
fd.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
fd.restype = ctypes.c_int32


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def run_mt(T, reps=5):
    bounds = [i * N // T for i in range(T + 1)]
    ps = np.zeros((T, G))
    pc = np.zeros((T, G), dtype=np.int64)

    def chunk(t):
        lo, hi = bounds[t], bounds[t + 1]
        k = keys[lo:hi]
        v = vals[lo:hi]
        rc = fd(k.ctypes.data, v.ctypes.data, hi - lo, ps[t].ctypes.data, pc[t].ctypes.data, G)
        assert rc == 0, rc

    def full():
        with ThreadPoolExecutor(max_workers=T) as ex:
            list(ex.map(chunk, range(T)))
        return ps.sum(axis=0), pc.sum(axis=0)

    s, c = full()
    ds = float(np.max(np.abs(s - ref_s)))
    dc = int(np.sum(c != ref_c))
    rel = ds / float(np.max(np.abs(ref_s)) + 1e-300)
    assert dc == 0, "counts must be bit-exact"
    run_mt.last_ds, run_mt.last_dc, run_mt.last_rel = ds, dc, rel
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=T) as ex:
            list(ex.map(chunk, range(T)))
        s = ps.sum(axis=0)
        c = pc.sum(axis=0)
        ts.append((time.perf_counter() - t) * 1e3)
    return med(ts), min(ts)


print("n=%d g=%d seed=42 (counts must be exact; sums: ULP-tolerance, see matrix)" % (N, G))
print("%-5s %10s %10s %8s %12s %10s" % ("T", "med_ms", "min_ms", "x_vs_1T", "max|dsums|", "count_bad"))
base = None
for T in TS:
    m, mn = run_mt(T)
    if base is None:
        base = m
    print("%-5d %10.3f %10.3f %8.2f %12.3g %10d" % (T, m, mn, base / m, run_mt.last_ds, run_mt.last_dc))
