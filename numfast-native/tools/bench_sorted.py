# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Benchmark sorted_run 1M sorted rows (G=256, seed 42), int64 + float64.

NumPy (unique+reduceat) vs Rust-native vs WASM. 5 warm + 10 measured.
Writes results/bench_sorted.json.
"""
import ctypes
import json
import os
import platform
import subprocess
import tempfile
import time

import numpy as np

import vectors_meta

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VEC = os.path.join(ROOT, "vectors")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
NODE = r"C:\Program Files\nodejs\node.exe"
WASM = os.path.join(ROOT, "target", "wasm32-unknown-unknown", "release", "numfast_native.wasm")
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
MJS = os.path.join(ROOT, "tools", "wasm_sorted.mjs")

# Resolved against THIS checkout: gen.py writes basenames, and a
# meta.json carrying another machine's absolute path used to kill this
# script with a FileNotFoundError that read like a parity failure.
meta = vectors_meta.load("big")
n = meta["n"]
keys = np.sort(np.fromfile(meta["keys"], dtype=np.int32))
rng = np.random.default_rng(42)
vi = rng.integers(-1000, 1000, size=n).astype(np.int64)
vf = rng.normal(0.0, 100.0, size=n).astype(np.float64)

dll = ctypes.CDLL(DLL)
fi = dll.nf_sorted_run_i64
fi.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
fi.restype = ctypes.c_int64
ff = dll.nf_sorted_run_f64
ff.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
ff.restype = ctypes.c_int64


def numpy_i64():
    uk, starts = np.unique(keys, return_index=True)
    return uk, np.add.reduceat(vi, starts), np.diff(np.append(starts, n))


def rust_i64(uk, s, c):
    ng = int(fi(keys.ctypes.data, vi.ctypes.data, n, uk.ctypes.data, s.ctypes.data, c.ctypes.data))
    assert ng >= 0
    return ng


uk0, s0, c0 = numpy_i64()
uk1 = np.zeros(n, dtype=np.int64)
s1 = np.zeros(n, dtype=np.int64)
c1 = np.zeros(n, dtype=np.int64)
ng1 = rust_i64(uk1, s1, c1)
assert ng1 == uk0.size and (uk1[:ng1] == uk0).all() and (s1[:ng1] == s0).all()
print("integrity i64: EXACT (ng=%d)" % ng1)


def bench(call):
    uk = np.zeros(n, dtype=np.int64)
    s = np.zeros(n, dtype=np.int64)
    c = np.zeros(n, dtype=np.int64)
    for _ in range(5):
        call(uk, s, c)
    runs = []
    for _ in range(10):
        t = time.perf_counter()
        call(uk, s, c)
        runs.append((time.perf_counter() - t) * 1e3)
    runs.sort()
    return runs[5], runs


def np_call(uk, s, c):
    uk[:], s[:], c[:] = 0, 0, 0
    u, st = np.unique(keys, return_index=True)
    m = u.size
    uk[:m] = u
    s[:m] = np.add.reduceat(vi, st)
    c[:m] = np.diff(np.append(st, n))


np_med, np_runs = bench(np_call)
ru_med, ru_runs = bench(rust_i64)

with tempfile.TemporaryDirectory() as td:
    ki = os.path.join(td, "k.i32")
    piv = os.path.join(td, "v.i64")
    keys.tofile(ki)
    vi.tofile(piv)
    node_out = subprocess.run([NODE, MJS, WASM, ki, piv, "1", str(n), "bench"],
                              capture_output=True, text=True, check=True)
nj = json.loads(node_out.stdout.strip())

machine = "%s | %s | %s | cpu=%s" % (
    platform.system(), platform.release(), platform.machine(), (platform.processor() or "?"))
print("machine:", machine)
print("case: sorted n=%d ng=%d (5 warm + 10 measured, median)" % (n, ng1))
print("numpy (unique+reduceat): med=%.3fms" % np_med)
print("rust-native (run pass)  : med=%.3fms" % ru_med)
print("wasm-node (run pass)    : med=%.3fms" % nj["median_ms"])
print("speedup rust/numpy=%.2fx wasm/numpy=%.2fx" % (np_med / ru_med, np_med / nj["median_ms"]))
json.dump({"machine": machine, "n": n, "ng": int(ng1),
           "numpy_med_ms": np_med, "numpy_runs": np_runs,
           "rust_med_ms": ru_med, "rust_runs": ru_runs,
           "wasm_med_ms": nj["median_ms"], "wasm_runs": nj["runs"]},
          open(os.path.join(RES, "bench_sorted.json"), "w"), indent=1)
