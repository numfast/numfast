# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden parity: NumPy reference vs Rust-native(ctypes) vs WASM(via Node).

Synthetic input shared via tools/gen.py vectors (seed 42).
Pass: counts bit-exact (int64), sums max|diff| == 0.0 (same accumulation
order as np.bincount). Prints table + writes results/parity.json.
"""
import ctypes
import json
import os
import subprocess

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VEC = os.path.join(ROOT, "vectors")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
NODE = r"C:\Program Files\nodejs\node.exe"
WASM = os.path.join(ROOT, "target", "wasm32-unknown-unknown", "release", "numfast_native.wasm")
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
MJS = os.path.join(ROOT, "tools", "wasm_run.mjs")

meta = json.load(open(os.path.join(VEC, "meta.json")))["small"]
n, g = meta["n"], meta["g"]
keys = np.fromfile(meta["keys"], dtype=np.int32)
vals = np.fromfile(meta["values"], dtype=np.float64)

# --- reference (NumPy): stage breakdown
import time
t = time.perf_counter()
ref_s = np.bincount(keys, weights=vals, minlength=g)
t_s = (time.perf_counter() - t) * 1e3
t = time.perf_counter()
ref_c = np.bincount(keys, minlength=g)
t_c = (time.perf_counter() - t) * 1e3

# --- Rust native via ctypes
dll = ctypes.CDLL(DLL)
fn = dll.nf_group_sum_count
fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
fn.restype = ctypes.c_int32
nat_s = np.zeros(g)
nat_c = np.zeros(g, dtype=np.int64)
rc = fn(keys.ctypes.data, vals.ctypes.data, n, nat_s.ctypes.data, nat_c.ctypes.data, g)
assert rc == 0, rc

# --- WASM via Node (parity mode writes outputs)
node_out = subprocess.run(
    [NODE, MJS, WASM, meta["keys"], meta["values"], str(n), str(g), "parity", RES],
    capture_output=True, text=True, check=True,
)
w_ms = json.loads(node_out.stdout.strip())["ms"]
w_s = np.fromfile(os.path.join(RES, "sums.f64"), dtype=np.float64)
w_c = np.fromfile(os.path.join(RES, "counts.i64"), dtype=np.int64)

rows = [
    ("rust-native", nat_s, nat_c),
    ("wasm-node", w_s, w_c),
]
print("case: small n=%d g=%d seed=42" % (n, g))
print("numpy ref stages: bincount_sums=%.3fms bincount_counts=%.3fms" % (t_s, t_c))
ok = True
for name, s, c in rows:
    ds = float(np.max(np.abs(s - ref_s)))
    dc = int(np.sum(c != ref_c))
    bit = ds == 0.0 and dc == 0
    ok &= bit
    print("%-12s max|dsum|=%.3g count_mismatch=%d %s" % (name, ds, dc, "EXACT" if bit else "DIFF"))
print("wasm-node call ms (1 run, cold): %.3f" % w_ms)
json.dump(
    {"n": n, "g": g, "seed": 42,
     "numpy_ms": {"sums": t_s, "counts": t_c},
     "rust_maxdiff": float(np.max(np.abs(nat_s - ref_s))),
     "rust_count_bad": int(np.sum(nat_c != ref_c)),
     "wasm_maxdiff": float(np.max(np.abs(w_s - ref_s))),
     "wasm_count_bad": int(np.sum(w_c != ref_c)),
     "wasm_cold_ms": w_ms, "pass": bool(ok)},
    open(os.path.join(RES, "parity.json"), "w"), indent=1,
)
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
