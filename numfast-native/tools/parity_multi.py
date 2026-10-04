# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden parity: multi-aggregate SoA — NumPy ref vs Rust-native vs WASM.

Synthetic: small keys vector (seed 42) + 3 f64 columns (seed 43..45).
Pass: counts bit-exact, per-column sums max|diff| == 0.0 (same row order
as np.bincount). Writes results/parity_multi.json.
"""
import ctypes
import json
import os
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
import wasm_artifact
# The CURRENT build, validated: 86 exports. wasm_artifact.resolve()
# aborts loudly on a missing or stale .wasm rather than falling back --
# four parity scripts used to validate the tracked 56-function copy and
# report green, which is how a suite comes to prove the wrong bytes.
WASM = wasm_artifact.resolve()
print(wasm_artifact.banner(WASM))
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
MJS = os.path.join(ROOT, "tools", "wasm_multi.mjs")

NCOLS = 3
# Resolved against THIS checkout: gen.py writes basenames, and a
# meta.json carrying another machine's absolute path used to kill this
# script with a FileNotFoundError that read like a parity failure.
meta = vectors_meta.load("small")
n, g = meta["n"], meta["g"]
keys = np.fromfile(meta["keys"], dtype=np.int32)
cols = np.stack(
    [np.random.default_rng(43 + c).normal(0.0, 100.0, size=n) for c in range(NCOLS)]
).astype(np.float64)
vflat = np.ascontiguousarray(cols.reshape(-1))

t = time.perf_counter()
ref_c = np.bincount(keys, minlength=g)
t_c = (time.perf_counter() - t) * 1e3
t = time.perf_counter()
ref_s = [np.bincount(keys, weights=cols[c], minlength=g) for c in range(NCOLS)]
t_s = (time.perf_counter() - t) * 1e3

dll = ctypes.CDLL(DLL)
fn = dll.nf_group_multi_sum_count
fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 2 + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
fn.restype = ctypes.c_int32
nat_s = np.zeros(NCOLS * g)
nat_c = np.zeros(g, dtype=np.int64)
rc = fn(keys.ctypes.data, vflat.ctypes.data, n, NCOLS, nat_s.ctypes.data, nat_c.ctypes.data, g)
assert rc == 0, rc
# error contract: null -> -1, OOB key -> -2
assert fn(0, vflat.ctypes.data, n, NCOLS, nat_s.ctypes.data, nat_c.ctypes.data, g) == -1
bad = keys.copy()
bad[0] = np.int32(g + 5)
s2 = np.zeros(NCOLS * g)
c2 = np.zeros(g, dtype=np.int64)
assert fn(bad.ctypes.data, vflat.ctypes.data, n, NCOLS, s2.ctypes.data, c2.ctypes.data, g) == -2

with tempfile.TemporaryDirectory() as td:
    vp = os.path.join(td, "vflat.f64")
    vflat.tofile(vp)
    node_out = subprocess.run(
        [NODE, MJS, WASM, meta["keys"], vp, str(n), str(NCOLS), str(g), "parity", RES],
        capture_output=True, text=True, check=True,
    )
w_ms = json.loads(node_out.stdout.strip())["ms"]
w_s = np.fromfile(os.path.join(RES, "multi_sums.f64"), dtype=np.float64).reshape(NCOLS, g)
w_c = np.fromfile(os.path.join(RES, "multi_counts.i64"), dtype=np.int64)

ok = True
dc = int(np.sum(nat_c != ref_c))
ok &= dc == 0
print("case: small n=%d g=%d ncols=%d seeds=42/43+" % (n, g, NCOLS))
print("numpy ref stages: counts=%.3fms sums3=%.3fms" % (t_c, t_s))
for c in range(NCOLS):
    ds = float(np.max(np.abs(nat_s[c * g:(c + 1) * g] - ref_s[c])))
    bit = ds == 0.0
    ok &= bit
    print("rust-native col%d max|dsum|=%.3g %s" % (c, ds, "EXACT" if bit else "DIFF"))
    dw = float(np.max(np.abs(w_s[c] - ref_s[c])))
    bitw = dw == 0.0
    ok &= bitw
    print("wasm-node   col%d max|dsum|=%.3g %s" % (c, dw, "EXACT" if bitw else "DIFF"))
dcw = int(np.sum(w_c != ref_c))
ok &= dcw == 0
print("counts mismatch: rust=%d wasm=%d" % (dc, dcw))
print("wasm-node call ms (1 run, cold): %.3f" % w_ms)
print("err contract: null=-1 ok, oob=-2 ok")
json.dump(
    {"n": n, "g": g, "ncols": NCOLS, "pass": bool(ok),
     "rust_count_bad": dc, "wasm_count_bad": dcw, "wasm_cold_ms": w_ms},
    open(os.path.join(RES, "parity_multi.json"), "w"), indent=1,
)
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
