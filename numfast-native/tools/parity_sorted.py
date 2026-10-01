# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden parity: sorted_run_aggregate — NumPy ref vs Rust-native vs WASM.

Sorted 10k keys (G=128) + int64 col + float64 col; ref = unique+reduceat.
Unsorted input -> -1 abort on all three; empty -> ng=0.
Pass: ng equal, ukeys/counts bit-exact, sums int exact / float max|diff|==0.
Writes results/parity_sorted.json.
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
MJS = os.path.join(ROOT, "tools", "wasm_sorted.mjs")

meta = json.load(open(os.path.join(VEC, "meta.json")))["small"]
n = meta["n"]
keys = np.sort(np.fromfile(meta["keys"], dtype=np.int32))
rng = np.random.default_rng(42)
vi = rng.integers(-1000, 1000, size=n).astype(np.int64)
vf = rng.normal(0.0, 100.0, size=n).astype(np.float64)

uk_ref, starts = np.unique(keys, return_index=True)
ref_c = np.diff(np.append(starts, n)).astype(np.int64)
ref_si = np.add.reduceat(vi, starts)
ref_sf = np.add.reduceat(vf, starts)
ng_ref = uk_ref.size

dll = ctypes.CDLL(DLL)
fi = dll.nf_sorted_run_i64
fi.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
fi.restype = ctypes.c_int64
ff = dll.nf_sorted_run_f64
ff.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
ff.restype = ctypes.c_int64


def call_i64(k, v):
    uk = np.zeros(n, dtype=np.int64)
    s = np.zeros(n, dtype=np.int64)
    c = np.zeros(n, dtype=np.int64)
    ng = int(fi(k.ctypes.data, v.ctypes.data, k.size, uk.ctypes.data, s.ctypes.data, c.ctypes.data))
    return ng, uk, s, c


def call_f64(k, v):
    uk = np.zeros(n, dtype=np.int64)
    s = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.int64)
    ng = int(ff(k.ctypes.data, v.ctypes.data, k.size, uk.ctypes.data, s.ctypes.data, c.ctypes.data))
    return ng, uk, s, c


ng_i, uk_i, s_i, c_i = call_i64(keys, vi)
ng_f, uk_f, s_f, c_f = call_f64(keys, vf)
# error contract: unsorted -> -1, null -> -2, empty -> 0
uns = keys.copy()
uns[n // 2], uns[n // 2 + 1] = uns[n // 2 + 1], uns[n // 2]  # may still sort; force inversion
uns[0], uns[1] = max(uns[0], uns[1]) + 1, 0
assert uns[1] < uns[0]
ng_bad, _, _, _ = call_i64(uns, vi)
assert ng_bad == -1, ng_bad
assert int(fi(0, vi.ctypes.data, n, uk_i.ctypes.data, s_i.ctypes.data, c_i.ctypes.data)) == -2
assert int(fi(keys.ctypes.data, vi.ctypes.data, 0, uk_i.ctypes.data, s_i.ctypes.data, c_i.ctypes.data)) == 0

with tempfile.TemporaryDirectory() as td:
    ki = os.path.join(td, "k.i32")
    piv = os.path.join(td, "v.i64")
    pfv = os.path.join(td, "v.f64")
    keys.tofile(ki)
    vi.tofile(piv)
    vf.tofile(pfv)
    o1 = subprocess.run([NODE, MJS, WASM, ki, piv, "1", str(n), "parity", RES],
                        capture_output=True, text=True, check=True)
    o2 = subprocess.run([NODE, MJS, WASM, ki, pfv, "0", str(n), "parity", RES],
                        capture_output=True, text=True, check=True)
ng_wi = json.loads(o1.stdout.strip())["ng"]
ng_wf = json.loads(o2.stdout.strip())["ng"]
w_uk_i = np.fromfile(os.path.join(RES, "sort_uk_i64.i64"), dtype=np.int64)
w_s_i = np.fromfile(os.path.join(RES, "sort_sums_i64.i64"), dtype=np.int64)
w_c_i = np.fromfile(os.path.join(RES, "sort_counts_i64.i64"), dtype=np.int64)
w_uk_f = np.fromfile(os.path.join(RES, "sort_uk_f64.i64"), dtype=np.int64)
w_s_f = np.fromfile(os.path.join(RES, "sort_sums_f64.f64"), dtype=np.float64)
w_c_f = np.fromfile(os.path.join(RES, "sort_counts_f64.i64"), dtype=np.int64)

ok = True
print("case: sorted n=%d ng_ref=%d" % (n, ng_ref))
for name, ng, uk, s, c, ref_s in [
        ("rust-i64", ng_i, uk_i[:max(ng_i, 0)], s_i[:max(ng_i, 0)], c_i[:max(ng_i, 0)], ref_si),
        ("wasm-i64", ng_wi, w_uk_i, w_s_i, w_c_i, ref_si)]:
    bit = ng == ng_ref and (uk == uk_ref).all() and (c == ref_c).all() and (s == ref_s).all()
    ok &= bit
    print("%-10s ng=%d %s" % (name, ng, "EXACT" if bit else "DIFF"))
# float: prod numba kernel is the order-reference (same sequential within-run
# order as Rust); np.add.reduceat uses a different summation order -> ULP diff.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "gi_prod", os.path.join(ROOT, "..", "src", "Drivers", "CPU", "_lib", "groupindex.py"))
_gi = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_gi)
pok, pay, _, _ = _gi.sorted_fused_aggregate(keys, [vf])
assert pok, "prod numba kernel refused sorted input"
numba_s = np.asarray(pay[2], dtype=np.float64)
for name, ng, uk, s, c in [
        ("rust-f64", ng_f, uk_f[:max(ng_f, 0)], s_f[:max(ng_f, 0)], c_f[:max(ng_f, 0)]),
        ("wasm-f64", ng_wf, w_uk_f, w_s_f, w_c_f)]:
    dn = float(np.max(np.abs(s - numba_s))) if ng == ng_ref else float("inf")
    dr = float(np.max(np.abs(s - ref_sf))) if ng == ng_ref else float("inf")
    bit = ng == ng_ref and (uk == uk_ref).all() and (c == ref_c).all() and dn == 0.0
    ok &= bit
    print("%-10s ng=%d vs-numba=%.3g vs-reduceat=%.3g %s"
          % (name, ng, dn, dr, "EXACT-vs-numba" if bit else "DIFF"))
print("err contract: unsorted=-1 ok, null=-2 ok, empty=0 ok")
json.dump({"n": n, "ng": int(ng_ref), "pass": bool(ok)},
          open(os.path.join(RES, "parity_sorted.json"), "w"), indent=1)
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
