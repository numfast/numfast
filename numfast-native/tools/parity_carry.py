# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden parity: carry_result_build — NumPy ref vs Rust-native vs WASM.

M=100k dense state, 10% present (seed 42); ref = flatnonzero + index.
Both dtypes; empty (m=0) -> 0; null -> -2. Pass: ng equal + exact.
Writes results/parity_carry.json.
"""
import ctypes
import json
import os
import subprocess
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
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
MJS = os.path.join(ROOT, "tools", "wasm_carry.mjs")

M = 100_000
rng = np.random.default_rng(42)
present = rng.random(M) < 0.10
cnt_m = np.where(present, rng.integers(1, 1000, size=M), 0).astype(np.int64)
si_m = np.where(present, rng.integers(-10**6, 10**6, size=M), 0).astype(np.int64)
sf_m = np.where(present, rng.normal(0.0, 100.0, size=M), 0.0).astype(np.float64)

pos = np.flatnonzero(cnt_m)
ref_uk = pos.astype(np.int64)
ref_c = cnt_m[pos]
ref_si = si_m[pos]
ref_sf = sf_m[pos]
ng_ref = pos.size

dll = ctypes.CDLL(DLL)
fi = dll.nf_carry_build_i64
fi.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
fi.restype = ctypes.c_int64
ff = dll.nf_carry_build_f64
ff.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
ff.restype = ctypes.c_int64


def call_i64():
    uk = np.zeros(M, dtype=np.int64)
    c = np.zeros(M, dtype=np.int64)
    s = np.zeros(M, dtype=np.int64)
    ng = int(fi(cnt_m.ctypes.data, si_m.ctypes.data, M, uk.ctypes.data, c.ctypes.data, s.ctypes.data))
    return ng, uk, c, s


def call_f64():
    uk = np.zeros(M, dtype=np.int64)
    c = np.zeros(M, dtype=np.int64)
    s = np.zeros(M, dtype=np.float64)
    ng = int(ff(cnt_m.ctypes.data, sf_m.ctypes.data, M, uk.ctypes.data, c.ctypes.data, s.ctypes.data))
    return ng, uk, c, s


ng_i, uk_i, c_i, s_i = call_i64()
ng_f, uk_f, c_f, s_f = call_f64()
assert int(fi(cnt_m.ctypes.data, si_m.ctypes.data, 0, uk_i.ctypes.data, c_i.ctypes.data, s_i.ctypes.data)) == 0
assert int(fi(0, si_m.ctypes.data, M, uk_i.ctypes.data, c_i.ctypes.data, s_i.ctypes.data)) == -2

with tempfile.TemporaryDirectory() as td:
    pc = os.path.join(td, "c.i64")
    pi = os.path.join(td, "s.i64")
    pf = os.path.join(td, "s.f64")
    cnt_m.tofile(pc)
    si_m.tofile(pi)
    sf_m.tofile(pf)
    o1 = subprocess.run([NODE, MJS, WASM, pc, pi, "1", str(M), "parity", RES],
                        capture_output=True, text=True, check=True)
    o2 = subprocess.run([NODE, MJS, WASM, pc, pf, "0", str(M), "parity", RES],
                        capture_output=True, text=True, check=True)
ng_wi = json.loads(o1.stdout.strip())["ng"]
ng_wf = json.loads(o2.stdout.strip())["ng"]
w_uk_i = np.fromfile(os.path.join(RES, "carry_uk_i64.i64"), dtype=np.int64)
w_c_i = np.fromfile(os.path.join(RES, "carry_counts_i64.i64"), dtype=np.int64)
w_s_i = np.fromfile(os.path.join(RES, "carry_sums_i64.i64"), dtype=np.int64)
w_uk_f = np.fromfile(os.path.join(RES, "carry_uk_f64.i64"), dtype=np.int64)
w_c_f = np.fromfile(os.path.join(RES, "carry_counts_f64.i64"), dtype=np.int64)
w_s_f = np.fromfile(os.path.join(RES, "carry_sums_f64.f64"), dtype=np.float64)

ok = True
print("case: carry m=%d ng_ref=%d" % (M, ng_ref))
for name, ng, uk, c, s, rs in [
        ("rust-i64", ng_i, uk_i[:ng_i], c_i[:ng_i], s_i[:ng_i], ref_si),
        ("wasm-i64", ng_wi, w_uk_i, w_c_i, w_s_i, ref_si),
        ("rust-f64", ng_f, uk_f[:ng_f], c_f[:ng_f], s_f[:ng_f], ref_sf),
        ("wasm-f64", ng_wf, w_uk_f, w_c_f, w_s_f, ref_sf)]:
    bit = ng == ng_ref and (uk == ref_uk).all() and (c == ref_c).all() and (s == rs).all()
    ok &= bit
    print("%-10s ng=%d %s" % (name, ng, "EXACT" if bit else "DIFF"))
print("err contract: empty=0 ok, null=-2 ok")
json.dump({"m": M, "ng": int(ng_ref), "pass": bool(ok)},
          open(os.path.join(RES, "parity_carry.json"), "w"), indent=1)
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
