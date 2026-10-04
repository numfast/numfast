# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Benchmark 1M rows (G=256, seed 42): NumPy vs Rust-native vs WASM(Node).

Warm runs then measured medians. Machine fingerprinted via platform.
Writes results/bench.json. Correctness re-checked (maxdiff) before timing.
"""
import ctypes
import json
import os
import platform
import subprocess
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
MJS = os.path.join(ROOT, "tools", "wasm_run.mjs")

# Resolved against THIS checkout: gen.py writes basenames, and a
# meta.json carrying another machine's absolute path used to kill this
# script with a FileNotFoundError that read like a parity failure.
meta = vectors_meta.load("big")
n, g = meta["n"], meta["g"]
keys = np.fromfile(meta["keys"], dtype=np.int32)
vals = np.fromfile(meta["values"], dtype=np.float64)

dll = ctypes.CDLL(DLL)
fn = dll.nf_group_sum_count
fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
fn.restype = ctypes.c_int32


def numpy_call(s, c):
    s[:] = np.bincount(keys, weights=vals, minlength=g)
    c[:] = np.bincount(keys, minlength=g)


def rust_call(s, c):
    rc = fn(keys.ctypes.data, vals.ctypes.data, n, s.ctypes.data, c.ctypes.data, g)
    assert rc == 0, rc


# integrity first (loss >1% = STOP per core rules; here exact expected)
s0 = np.zeros(g)
c0 = np.zeros(g, dtype=np.int64)
numpy_call(s0, c0)
s1 = np.zeros(g)
c1 = np.zeros(g, dtype=np.int64)
rust_call(s1, c1)
md = float(np.max(np.abs(s1 - s0)))
cb = int(np.sum(c1 != c0))
print("integrity: rust_vs_numpy maxdiff=%.3g count_bad=%d" % (md, cb))
assert md == 0.0 and cb == 0, "integrity FAILED"


def bench(name, call):
    s = np.zeros(g)
    c = np.zeros(g, dtype=np.int64)
    for _ in range(5):
        call(s, c)  # warm
    runs = []
    for _ in range(10):
        t = time.perf_counter()
        call(s, c)
        runs.append((time.perf_counter() - t) * 1e3)
    runs.sort()
    return runs[5], runs


np_med, np_runs = bench("numpy", numpy_call)
ru_med, ru_runs = bench("rust", rust_call)

node_out = subprocess.run(
    [NODE, MJS, WASM, meta["keys"], meta["values"], str(n), str(g), "bench"],
    capture_output=True, text=True, check=True,
)
nj = json.loads(node_out.stdout.strip())

machine = "%s | %s | %s | cpu=%s" % (
    platform.system(), platform.release(), platform.machine(),
    (platform.processor() or "?"))
try:
    import psutil  # optional
    machine += " ram=%.1fGB" % (psutil.virtual_memory().total / 2**30)
except ImportError:
    pass

print("machine:", machine)
print("case: big n=%d g=%d seed=42 (5 warm + 10 measured, median)" % (n, g))
print("numpy (2x bincount) : med=%.3fms runs=%s" % (np_med, ["%.2f" % x for x in np_runs]))
print("rust-native (fused) : med=%.3fms runs=%s" % (ru_med, ["%.2f" % x for x in ru_runs]))
print("wasm-node (fused)   : med=%.3fms runs=%s" % (
    nj["median_ms"], ["%.2f" % x for x in nj["runs"]]))
print("speedup rust/numpy=%.2fx wasm/numpy=%.2fx" % (np_med / ru_med, np_med / nj["median_ms"]))
json.dump({"machine": machine, "n": n, "g": g, "seed": 42,
           "numpy_med_ms": np_med, "numpy_runs": np_runs,
           "rust_med_ms": ru_med, "rust_runs": ru_runs,
           "wasm_med_ms": nj["median_ms"], "wasm_runs": nj["runs"]},
          open(os.path.join(RES, "bench.json"), "w"), indent=1)
