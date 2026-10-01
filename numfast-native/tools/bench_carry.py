# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Benchmark carry_result_build 1M dense lanes, 10% present (seed 42).

NumPy (flatnonzero + 2x index) vs Rust-native vs WASM. 5+10, median.
Writes results/bench_carry.json.
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
MJS = os.path.join(ROOT, "tools", "wasm_carry.mjs")

M = 1_000_000
rng = np.random.default_rng(42)
present = rng.random(M) < 0.10
cnt_m = np.where(present, rng.integers(1, 1000, size=M), 0).astype(np.int64)
si_m = np.where(present, rng.integers(-10**6, 10**6, size=M), 0).astype(np.int64)

dll = ctypes.CDLL(DLL)
fi = dll.nf_carry_build_i64
fi.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
fi.restype = ctypes.c_int64


def numpy_call(uk, c, s):
    pos = np.flatnonzero(cnt_m)
    uk[:pos.size] = pos
    c[:pos.size] = cnt_m[pos]
    s[:pos.size] = si_m[pos]
    return pos.size


def rust_call(uk, c, s):
    ng = int(fi(cnt_m.ctypes.data, si_m.ctypes.data, M, uk.ctypes.data, c.ctypes.data, s.ctypes.data))
    assert ng >= 0
    return ng


uk0 = np.zeros(M, dtype=np.int64)
c0 = np.zeros(M, dtype=np.int64)
s0 = np.zeros(M, dtype=np.int64)
ng0 = numpy_call(uk0, c0, s0)
uk1 = np.zeros(M, dtype=np.int64)
c1 = np.zeros(M, dtype=np.int64)
s1 = np.zeros(M, dtype=np.int64)
ng1 = rust_call(uk1, c1, s1)
assert ng1 == ng0 and (uk1[:ng1] == uk0[:ng1]).all() and (s1[:ng1] == s0[:ng1]).all()
print("integrity: EXACT (ng=%d)" % ng1)


def bench(call):
    uk = np.zeros(M, dtype=np.int64)
    c = np.zeros(M, dtype=np.int64)
    s = np.zeros(M, dtype=np.int64)
    for _ in range(5):
        call(uk, c, s)
    runs = []
    for _ in range(10):
        t = time.perf_counter()
        call(uk, c, s)
        runs.append((time.perf_counter() - t) * 1e3)
    runs.sort()
    return runs[5], runs


np_med, np_runs = bench(numpy_call)
ru_med, ru_runs = bench(rust_call)

with tempfile.TemporaryDirectory() as td:
    pc = os.path.join(td, "c.i64")
    pi = os.path.join(td, "s.i64")
    cnt_m.tofile(pc)
    si_m.tofile(pi)
    node_out = subprocess.run([NODE, MJS, WASM, pc, pi, "1", str(M), "bench"],
                              capture_output=True, text=True, check=True)
nj = json.loads(node_out.stdout.strip())

machine = "%s | %s | %s | cpu=%s" % (
    platform.system(), platform.release(), platform.machine(), (platform.processor() or "?"))
print("machine:", machine)
print("case: carry m=%d ng=%d (5 warm + 10 measured, median)" % (M, ng1))
print("numpy (flatnonzero+index): med=%.3fms" % np_med)
print("rust-native (fwd pass)    : med=%.3fms" % ru_med)
print("wasm-node (fwd pass)      : med=%.3fms" % nj["median_ms"])
print("speedup rust/numpy=%.2fx wasm/numpy=%.2fx" % (np_med / ru_med, np_med / nj["median_ms"]))
json.dump({"machine": machine, "m": M, "ng": int(ng1),
           "numpy_med_ms": np_med, "numpy_runs": np_runs,
           "rust_med_ms": ru_med, "rust_runs": ru_runs,
           "wasm_med_ms": nj["median_ms"], "wasm_runs": nj["runs"]},
          open(os.path.join(RES, "bench_carry.json"), "w"), indent=1)
