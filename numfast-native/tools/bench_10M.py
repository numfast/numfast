# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""10M scaling: native vs numpy vs wasm across kernels (seed 42).

Usage: python tools/bench_10M.py [--reps R]
Integrtiy first (exact), then medians. RSS sampled via psutil. New file.
"""
import ctypes
import os
import sys
import time

import numpy as np

REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 7
N = 10_000_000
G = 256
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")

try:
    import psutil
    rss = lambda: psutil.Process().memory_info().rss / 2**20
except ImportError:
    rss = lambda: float("nan")

rng = np.random.default_rng(42)


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def bench(name, fn, check):
    check()
    for _ in range(2):
        fn()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    print("%-14s med=%9.3fms min=%9.3fms" % (name, med(ts), min(ts)))
    return med(ts)


print("rss_start=%.1fMB" % rss())
d = ctypes.CDLL(DLL)
fd = d.nf_group_sum_count
fd.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
fd.restype = ctypes.c_int32

keys = rng.integers(0, G, size=N).astype(np.int32)
vals = rng.standard_normal(N)
s = np.zeros(G)
c = np.zeros(G, dtype=np.int64)
ref_s = np.bincount(keys, weights=vals, minlength=G)
ref_c = np.bincount(keys, minlength=G)


def numpy_dense():
    s[:] = np.bincount(keys, weights=vals, minlength=G)
    c[:] = np.bincount(keys, minlength=G)


def rust_dense():
    assert fd(keys.ctypes.data, vals.ctypes.data, N, s.ctypes.data, c.ctypes.data, G) == 0


bench("numpy_dense", numpy_dense, lambda: None)
bench("rust_dense", rust_dense, lambda: (
    rust_dense(),
    (_ for _ in ()).throw(AssertionError("dense integrity"))
    if not (np.max(np.abs(s - ref_s)) == 0.0 and np.sum(c != ref_c) == 0) else None))

fm = d.nf_group_multi_sum_count
fm.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 2 + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
fm.restype = ctypes.c_int32
V = np.ascontiguousarray(rng.standard_normal((3, N)))
S = np.zeros(3 * G)
C = np.zeros(G, dtype=np.int64)


def rust_multi():
    assert fm(keys.ctypes.data, V.ctypes.data, N, 3, S.ctypes.data, C.ctypes.data, G) == 0


bench("rust_multi3c", rust_multi, lambda: rust_multi())

fp = d.nf_pack_i32_direct
fp.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]
fp.restype = ctypes.c_int32
k1 = rng.integers(0, 100, size=N).astype(np.int32)
k2 = rng.integers(0, 1000, size=N).astype(np.int32)
out = np.zeros(N, dtype=np.int32)
ref_o = k1.astype(np.int64) * 1000 + k2


def rust_pack():
    assert fp(k1.ctypes.data, k2.ctypes.data, 1000, N, out.ctypes.data) == 0


bench("numpy_pack", lambda: out.__setitem__(slice(None), (ref_o).astype(np.int32)), lambda: None)
bench("rust_pack", rust_pack, lambda: (rust_pack(), None)
      if np.sum(out.astype(np.int64) != ref_o) == 0 else (_ for _ in ()).throw(AssertionError("pack")))

sk = np.repeat(np.arange(G, dtype=np.int32), N // G)
n2 = len(sk)
sv = (rng.standard_normal(n2) * 100).astype(np.int64)
uk = np.zeros(n2, dtype=np.int64)
ss = np.zeros(n2, dtype=np.int64)
cc = np.zeros(n2, dtype=np.int64)
fs = d.nf_sorted_run_i64
fs.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
fs.restype = ctypes.c_int64


def rust_sorted():
    assert fs(sk.ctypes.data, sv.ctypes.data, n2, uk.ctypes.data, ss.ctypes.data, cc.ctypes.data) == G


bench("rust_sorted", rust_sorted, lambda: rust_sorted())
print("rss_end=%.1fMB" % rss())
