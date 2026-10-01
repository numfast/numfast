# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden parity: pack_i32_direct — NumPy ref vs Rust-native vs WASM.

Synthetic: k1 in [0,K1), k2 in [0,K2), seed 42; M2=K2 (mixed radix).
Pass: bit-exact int32 (mismatch count == 0). Writes results/parity_pack.json.
"""
import ctypes
import json
import os
import subprocess
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VEC = os.path.join(ROOT, "vectors")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
NODE = r"C:\Program Files\nodejs\node.exe"
WASM = os.path.join(ROOT, "target", "wasm32-unknown-unknown", "release", "numfast_native.wasm")
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
MJS = os.path.join(ROOT, "tools", "wasm_pack.mjs")

N, K1, K2 = 100_000, 100, 1000
M2 = K2
rng = np.random.default_rng(42)
k1 = rng.integers(0, K1, size=N).astype(np.int32)
k2 = rng.integers(0, K2, size=N).astype(np.int32)
assert int(K1 - 1) * M2 + int(K2 - 1) <= 2**31 - 1

ref = np.empty(N, dtype=np.int32)
np.multiply(k1, np.int32(M2), out=ref)
np.add(ref, k2, out=ref)

dll = ctypes.CDLL(DLL)
fn = dll.nf_pack_i32_direct
fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]
fn.restype = ctypes.c_int32
nat = np.zeros(N, dtype=np.int32)
rc = fn(k1.ctypes.data, k2.ctypes.data, M2, N, nat.ctypes.data)
assert rc == 0, rc
assert fn(0, k2.ctypes.data, M2, N, nat.ctypes.data) == -1
ov1 = np.array([2**30], dtype=np.int32)
ov2 = np.array([2**30], dtype=np.int32)
ovo = np.zeros(1, dtype=np.int32)
assert fn(ov1.ctypes.data, ov2.ctypes.data, 2, 1, ovo.ctypes.data) == -2
neg = np.array([-1], dtype=np.int32)
assert fn(neg.ctypes.data, ov2.ctypes.data, M2, 1, ovo.ctypes.data) == -2

with tempfile.TemporaryDirectory() as td:
    p1 = os.path.join(td, "k1.i32")
    p2 = os.path.join(td, "k2.i32")
    k1.tofile(p1)
    k2.tofile(p2)
    node_out = subprocess.run(
        [NODE, MJS, WASM, p1, p2, str(M2), str(N), "parity", RES],
        capture_output=True, text=True, check=True,
    )
w_ms = json.loads(node_out.stdout.strip())["ms"]
w = np.fromfile(os.path.join(RES, "pack_out.i32"), dtype=np.int32)

dn = int(np.sum(nat != ref))
dw = int(np.sum(w != ref))
ok = dn == 0 and dw == 0
print("case: pack n=%d k1=%d k2=%d m2=%d seed=42" % (N, K1, K2, M2))
print("rust-native mismatch=%d %s" % (dn, "EXACT" if dn == 0 else "DIFF"))
print("wasm-node   mismatch=%d %s" % (dw, "EXACT" if dw == 0 else "DIFF"))
print("wasm-node call ms (1 run, cold): %.3f" % w_ms)
print("err contract: null=-1 ok, overflow=-2 ok, negative=-2 ok")
json.dump({"n": N, "k1": K1, "k2": K2, "m2": M2, "pass": bool(ok),
           "rust_bad": dn, "wasm_bad": dw, "wasm_cold_ms": w_ms},
          open(os.path.join(RES, "parity_pack.json"), "w"), indent=1)
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
