# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Carry branch-cost probe: time vs present-fraction (seed 42).

Usage: python tools/carry_skew.py [dll] [--m M] [--reps R]
Varies density of nonzero counts (1/10/50/100%) at fixed m, reports med ms.
Decides whether a branchless carry variant is worth building. New file.
"""
import ctypes
import sys
import time

import numpy as np

DLL = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else \
    r"numfast-native/target/x86_64-pc-windows-gnu/release/numfast_native.dll"
m = int(sys.argv[sys.argv.index("--m") + 1]) if "--m" in sys.argv else 1_000_000
reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 11

d = ctypes.CDLL(DLL)
f = d.nf_carry_build_i64
f.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
f.restype = ctypes.c_int64

rng = np.random.default_rng(42)
mag = rng.integers(1, 5, size=m).astype(np.int64)
sums = (rng.standard_normal(m) * 100).astype(np.int64)
ou = np.zeros(m, dtype=np.int64)
oc = np.zeros(m, dtype=np.int64)
os_ = np.zeros(m, dtype=np.int64)


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


print("%-8s %10s %10s" % ("present%", "ng", "med_ms"))
for frac in (0.01, 0.10, 0.50, 1.00):
    cm = (rng.random(m) < frac).astype(np.int64) * mag
    sm = sums * np.sign(cm).astype(np.int64)
    ng = f(cm.ctypes.data, sm.ctypes.data, m, ou.ctypes.data, oc.ctypes.data, os_.ctypes.data)
    assert ng == int(np.sum(cm != 0)), (frac, ng)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f(cm.ctypes.data, sm.ctypes.data, m, ou.ctypes.data, oc.ctypes.data, os_.ctypes.data)
        ts.append((time.perf_counter() - t) * 1e3)
    print("%-8.0f %10d %10.3f" % (frac * 100, ng, med(ts)))
