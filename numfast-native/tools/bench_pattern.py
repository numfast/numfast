# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Benchmark pattern_encode 100k H2O-like rows ("id%03d" + 5% invalid).

Python-loop reference (best of 3) vs Rust-native vs WASM (5 warm + 10).
Writes results/bench_pattern.json.
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
MJS = os.path.join(ROOT, "tools", "wasm_pattern.mjs")

PREFIX = "id"
N = 100_000
rng = np.random.default_rng(42)
strs = [("id%03d" % int(rng.integers(0, 1000))) if rng.random() > 0.05
        else rng.choice(["xx1", "id12a", "id", ""]) for _ in range(N)]
enc = [s.encode() for s in strs]
offs = np.zeros(N + 1, dtype=np.int32)
offs[1:] = np.cumsum([len(b) for b in enc])
data = b"".join(enc)
pfx = b"id"


def py_loop():
    codes = np.zeros(N, dtype=np.int32)
    valid = np.zeros(N, dtype=np.uint8)
    for i, s in enumerate(strs):
        if not s.startswith("id"):
            continue
        body = s[2:]
        neg = body.startswith("-")
        core = body[1:] if neg else body
        if not core or any(not ("0" <= ch <= "9") for ch in core):
            continue
        mag = int(core)
        if mag > (2**31 if neg else 2**31 - 1):
            raise OverflowError(i)
        codes[i] = -mag if neg else mag
        valid[i] = 1
    return codes, valid


dll = ctypes.CDLL(DLL)
fn = dll.nf_pattern_encode
fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
               ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
               ctypes.c_void_p, ctypes.c_void_p]
fn.restype = ctypes.c_int32
dp = np.frombuffer(data, dtype=np.uint8)
pp = np.frombuffer(pfx, dtype=np.uint8)


def rust_call(c, v):
    w = np.zeros(1, dtype=np.int32)
    e = np.zeros(1, dtype=np.int32)
    rc = fn(dp.ctypes.data, len(data), offs.ctypes.data, N,
            pp.ctypes.data, len(pfx), c.ctypes.data, v.ctypes.data,
            w.ctypes.data, e.ctypes.data)
    assert rc == 0, rc


ref_c, ref_v = py_loop()
c1 = np.zeros(N, dtype=np.int32)
v1 = np.zeros(N, dtype=np.uint8)
rust_call(c1, v1)
assert int(np.sum(c1 != ref_c)) == 0 and int(np.sum(v1 != ref_v)) == 0, "integrity FAILED"
print("integrity: rust_vs_pyloop EXACT")

t0s = []
for _ in range(3):
    t = time.perf_counter()
    py_loop()
    t0s.append((time.perf_counter() - t) * 1e3)
py_med = min(t0s)

c = np.zeros(N, dtype=np.int32)
v = np.zeros(N, dtype=np.uint8)
for _ in range(5):
    rust_call(c, v)
runs = []
for _ in range(10):
    t = time.perf_counter()
    rust_call(c, v)
    runs.append((time.perf_counter() - t) * 1e3)
runs.sort()
ru_med = runs[5]

with tempfile.TemporaryDirectory() as td:
    dp_ = os.path.join(td, "data.bin")
    op_ = os.path.join(td, "offs.i32")
    open(dp_, "wb").write(data)
    offs.tofile(op_)
    node_out = subprocess.run(
        [NODE, MJS, WASM, dp_, op_, PREFIX, str(N), "bench"],
        capture_output=True, text=True, check=True,
    )
nj = json.loads(node_out.stdout.strip())

machine = "%s | %s | %s | cpu=%s" % (
    platform.system(), platform.release(), platform.machine(), (platform.processor() or "?"))
print("machine:", machine)
print("case: pattern n=%d (py best-of-3; rust/wasm 5 warm + 10 measured)" % N)
print("py-loop reference : %.3fms (%.0f rows/s)" % (py_med, N / py_med * 1e3))
print("rust-native       : %.3fms (%.0f rows/s)" % (ru_med, N / ru_med * 1e3))
print("wasm-node         : %.3fms (%.0f rows/s)" % (nj["median_ms"], N / nj["median_ms"] * 1e3))
print("speedup rust/py=%.1fx wasm/py=%.1fx" % (py_med / ru_med, py_med / nj["median_ms"]))
json.dump({"machine": machine, "n": N, "py_med_ms": py_med,
           "rust_med_ms": ru_med, "rust_runs": runs,
           "wasm_med_ms": nj["median_ms"], "wasm_runs": nj["runs"]},
          open(os.path.join(RES, "bench_pattern.json"), "w"), indent=1)
