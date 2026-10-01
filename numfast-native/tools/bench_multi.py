# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Benchmark multi-aggregate SoA 1M rows x 3 cols (G=256, seeds 42/43+).

NumPy (3x bincount sums + 1x counts) vs Rust-native fused vs WASM fused.
5 warm + 10 measured, median. Writes results/bench_multi.json.
"""
import ctypes
import json
import os
import platform
import subprocess
import tempfile
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VEC = os.path.join(ROOT, "vectors")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
NODE = r"C:\Program Files\nodejs\node.exe"
WASM = os.path.join(ROOT, "target", "wasm32-unknown-unknown", "release", "numfast_native.wasm")
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
MJS = os.path.join(ROOT, "tools", "wasm_multi.mjs")

NCOLS = 3
meta = json.load(open(os.path.join(VEC, "meta.json")))["big"]
n, g = meta["n"], meta["g"]
keys = np.fromfile(meta["keys"], dtype=np.int32)
cols = np.stack(
    [np.random.default_rng(43 + c).normal(0.0, 100.0, size=n) for c in range(NCOLS)]
).astype(np.float64)
vflat = np.ascontiguousarray(cols.reshape(-1))

dll = ctypes.CDLL(DLL)
fn = dll.nf_group_multi_sum_count
fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 2 + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
fn.restype = ctypes.c_int32


def numpy_call(s, c):
    for i in range(NCOLS):
        s[i * g:(i + 1) * g] = np.bincount(keys, weights=cols[i], minlength=g)
    c[:] = np.bincount(keys, minlength=g)


def rust_call(s, c):
    rc = fn(keys.ctypes.data, vflat.ctypes.data, n, NCOLS, s.ctypes.data, c.ctypes.data, g)
    assert rc == 0, rc


s0 = np.zeros(NCOLS * g)
c0 = np.zeros(g, dtype=np.int64)
numpy_call(s0, c0)
s1 = np.zeros(NCOLS * g)
c1 = np.zeros(g, dtype=np.int64)
rust_call(s1, c1)
md = float(np.max(np.abs(s1 - s0)))
cb = int(np.sum(c1 != c0))
print("integrity: rust_vs_numpy maxdiff=%.3g count_bad=%d" % (md, cb))
assert md == 0.0 and cb == 0, "integrity FAILED"


def bench(call):
    s = np.zeros(NCOLS * g)
    c = np.zeros(g, dtype=np.int64)
    for _ in range(5):
        call(s, c)
    runs = []
    for _ in range(10):
        t = time.perf_counter()
        call(s, c)
        runs.append((time.perf_counter() - t) * 1e3)
    runs.sort()
    return runs[5], runs


np_med, np_runs = bench(numpy_call)
ru_med, ru_runs = bench(rust_call)

with tempfile.TemporaryDirectory() as td:
    vp = os.path.join(td, "vflat.f64")
    vflat.tofile(vp)
    node_out = subprocess.run(
        [NODE, MJS, WASM, meta["keys"], vp, str(n), str(NCOLS), str(g), "bench"],
        capture_output=True, text=True, check=True,
    )
nj = json.loads(node_out.stdout.strip())

machine = "%s | %s | %s | cpu=%s" % (
    platform.system(), platform.release(), platform.machine(), (platform.processor() or "?"))
print("machine:", machine)
print("case: big n=%d g=%d ncols=%d (5 warm + 10 measured, median)" % (n, g, NCOLS))
print("numpy (3x sums + counts): med=%.3fms" % np_med)
print("rust-native (fused SoA) : med=%.3fms" % ru_med)
print("wasm-node (fused SoA)   : med=%.3fms" % nj["median_ms"])
print("speedup rust/numpy=%.2fx wasm/numpy=%.2fx" % (np_med / ru_med, np_med / nj["median_ms"]))
json.dump({"machine": machine, "n": n, "g": g, "ncols": NCOLS,
           "numpy_med_ms": np_med, "numpy_runs": np_runs,
           "rust_med_ms": ru_med, "rust_runs": ru_runs,
           "wasm_med_ms": nj["median_ms"], "wasm_runs": nj["runs"]},
          open(os.path.join(RES, "bench_multi.json"), "w"), indent=1)
