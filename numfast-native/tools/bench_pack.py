# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Benchmark pack_i32_direct 1M rows (K1=100, K2=1000, seed 42).

NumPy (multiply+add with out=) vs Rust-native vs WASM.
5 warm + 10 measured, median. Writes results/bench_pack.json.
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
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
NODE = r"C:\Program Files\nodejs\node.exe"
WASM = os.path.join(ROOT, "target", "wasm32-unknown-unknown", "release", "numfast_native.wasm")
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
MJS = os.path.join(ROOT, "tools", "wasm_pack.mjs")

N, K1, K2 = 1_000_000, 100, 1000
M2 = K2
rng = np.random.default_rng(42)
k1 = rng.integers(0, K1, size=N).astype(np.int32)
k2 = rng.integers(0, K2, size=N).astype(np.int32)

dll = ctypes.CDLL(DLL)
fn = dll.nf_pack_i32_direct
fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]
fn.restype = ctypes.c_int32


def numpy_call(o):
    np.multiply(k1, np.int32(M2), out=o)
    np.add(o, k2, out=o)


def rust_call(o):
    rc = fn(k1.ctypes.data, k2.ctypes.data, M2, N, o.ctypes.data)
    assert rc == 0, rc


o0 = np.zeros(N, dtype=np.int32)
numpy_call(o0)
o1 = np.zeros(N, dtype=np.int32)
rust_call(o1)
cb = int(np.sum(o1 != o0))
print("integrity: rust_vs_numpy mismatch=%d" % cb)
assert cb == 0, "integrity FAILED"


def bench(call):
    o = np.zeros(N, dtype=np.int32)
    for _ in range(5):
        call(o)
    runs = []
    for _ in range(10):
        t = time.perf_counter()
        call(o)
        runs.append((time.perf_counter() - t) * 1e3)
    runs.sort()
    return runs[5], runs


np_med, np_runs = bench(numpy_call)
ru_med, ru_runs = bench(rust_call)

with tempfile.TemporaryDirectory() as td:
    p1 = os.path.join(td, "k1.i32")
    p2 = os.path.join(td, "k2.i32")
    k1.tofile(p1)
    k2.tofile(p2)
    node_out = subprocess.run(
        [NODE, MJS, WASM, p1, p2, str(M2), str(N), "bench"],
        capture_output=True, text=True, check=True,
    )
nj = json.loads(node_out.stdout.strip())

machine = "%s | %s | %s | cpu=%s" % (
    platform.system(), platform.release(), platform.machine(), (platform.processor() or "?"))
print("machine:", machine)
print("case: pack n=%d (5 warm + 10 measured, median)" % N)
print("numpy (mul+add) : med=%.3fms" % np_med)
print("rust-native     : med=%.3fms" % ru_med)
print("wasm-node       : med=%.3fms" % nj["median_ms"])
print("speedup rust/numpy=%.2fx wasm/numpy=%.2fx" % (np_med / ru_med, np_med / nj["median_ms"]))
json.dump({"machine": machine, "n": N, "numpy_med_ms": np_med, "numpy_runs": np_runs,
           "rust_med_ms": ru_med, "rust_runs": ru_runs,
           "wasm_med_ms": nj["median_ms"], "wasm_runs": nj["runs"]},
          open(os.path.join(RES, "bench_pack.json"), "w"), indent=1)
