# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden parity: sorted_run_aggregate — NumPy ref vs Rust-native vs WASM.

Sorted 10k keys (G=128) + int64 col + float64 col; ref = unique+reduceat.
Unsorted input -> -1 abort on all three; empty -> ng=0.
Writes results/parity_sorted.json.

=============================================================================
THE f64 CONTRACT, and why it is not bit-exactness  (revised 2026-10-05)
=============================================================================

OLD REQUIREMENT (deleted, deliberately):
    "float max|diff| == 0.0" -- the f64 lane was required to be BIT-EXACT
    against the production kernel. That requirement cannot be met and was
    never a property of the operator. `results/parity_sorted.json` moved
    pass:true -> false because the number stopped being zero.

THE MEASUREMENT THAT FORCED THE CHANGE (seed 42, this exact fixture):
    max|diff| vs the production kernel        1.36424e-12
    max|diff| vs np.add.reduceat of the same column
                                             1.36424e-12   <- IDENTICAL
    The two "references" are the same computation, which is the first fact
    this file's old comment got backwards. It claimed the production numba
    kernel was the order-reference and that `np.add.reduceat` used a
    different order. Since the numba lane was retired (NATIVE-ALL), the
    production kernel IS `np.add.reduceat`:
        src/Drivers/CPU/_lib/groupindex.py:298
            sums = np.add.reduceat(v.astype(np.float64), starts)
    There is no second reference. `ref_sf` above and the production payload
    are the same array.

WHERE THE DIFFERENCE COMES FROM (measured, not reasoned):
    f64 unit roundoff u64 = 2^-53 = 1.11e-16; machine epsilon 2^-52 =
    2.22e-16. 1.36e-12 is ~12000x u64, so it is an ACCUMULATED quantity, not
    a single rounding. Two accumulation orders were compared directly:
      * `nf_sorted_run_f64` (Rust) accumulates strictly left-to-right within
        a run: `acc += vals[i]`, numfast-native/src/groupby/sorted.rs:52.
      * `np.add.reduceat` sums a run by numpy's pairwise scheme.
    A pure-Python strict sequential accumulator was run over the same
    fixture. Result, on this fixture and on 26 further configurations
    (n 1e4..1e6, G 2..50000, normal / lognormal / exponential / +-1e8 /
    12-orders-of-magnitude / exactly-cancelling / 1e-300 / 1e-320):
        max|rust - sequential|      0.0   (bit-identical, 27/27)
        max|wasm - sequential|      0.0   (bit-identical)
        max|pairwise - sequential|  > 0    (24/27; equal only where the
                                            data makes the orders coincide)
    So the cause is ESTABLISHED: the kernels and the production kernel
    compute the same mathematical sum in two different valid orders. It is
    not a WASM divergence -- Rust and WASM are the same source and return
    the same bits, which is why the old script printed one number for both.
    It is also not a defect in either side: IEEE-754 does not fix the order
    of a sum, so no ordering is the uniquely correct one.

WHAT conformance-profile.toml ALREADY PERMITS
    specs/core/conformance-profile.toml, [tolerance.f64]:
        atol = 1e-12, rtol = 1e-12, max_ulp = 2
        rule (from numeric contract 08), evaluated PER GROUP:
            max_abs_diff <= max(atol, rtol*|ref|)  OR  <= max_ulp ULP
    On this fixture the profile's own budget already covers the number:
        |ref| spans 9.77832 .. 2566.42, so rtol*|ref| >= 9.78e-12
        max|diff| = 1.36e-12  ->  0.14x of the rtol branch, PASSES.
    Stated precisely, because the shorthand is misleading: 1.36e-12 is
    ABOVE atol (1e-12) and ABOVE 2 ULP at that magnitude (9.09e-13 at
    |ref|=2566). It passes only via `rtol*|ref|`, which is the profile's
    stated rule and not a number invented for this fixture.

WHY THE PROFILE'S THREE BRANCHES ALONE ARE NOT ENOUGH HERE
    `rtol*|ref|` is relative to the group's SUM. A group whose sum nearly
    cancels has |ref| -> 0, so that branch collapses and only atol=1e-12 is
    left -- while the actual order difference grows with run length. This
    is not hypothetical; measured, single run, sum driven to 0:
        L=1025    max|diff| 7.276e-12   profile FAILS (rtol*|ref| = 0)
        L=8193    max|diff| 1.444e-11   profile FAILS
        L=100001  max|diff| 2.756e-08   profile FAILS
    The operator's accepted domain is "sorted int32 keys, any f64 values",
    so those inputs are legal. A tolerance that a legal input violates is
    not a contract, it is a coin flip. The script therefore keeps the
    profile's branches and adds the one the error actually lives in.

THE ADDED BRANCH, AND WHY IT IS DERIVED AND NOT FITTED
    Branch (d), per group g:
        |diff_g|  <=  L_g * u64 * sum(|a|_g),      u64 = 2^-53
    L_g is the run length, u64 the machine unit roundoff, sum(|a|_g) the
    run's own absolute sum. This is the standard forward-error bound for
    floating-point summation,
        |computed - exact| <= (L-1) * u * sum|a_i|
    (Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.,
    sec 4.2, Thm 4.1) for the sequential order, whose pairwise counterpart
    is bounded by log2(L)*u*sum|a_i|. The gap between the two orders is
    bounded by the sum of the two, i.e. by L*u*sum|a_i| to within a small
    constant. No constant here was chosen to turn this fixture green; the
    measurement simply lands far inside the bound. Measured worst observed
    ratio |diff| / (L*u64*sum|a|) over 36 configurations: 0.744 (this
    fixture: 0.0236). On the three cases that defeat the profile it holds
    with 0.6% .. 6% margin.

THE CONTRACT AS IMPLEMENTED
    * int64 lane  -- BIT-EXACT. Integer addition is exact and
      order-independent, so `max|diff| == 0` is the right requirement and
      is kept unchanged (measured EXACT at n=1e4/G=128 and n=1e6/G=256).
    * f64 lane    -- NOT bit-exact. A group passes when ANY of
        (a) |diff| <= 1e-12                        [profile atol]
        (b) |diff| <= 1e-12 * |ref_g|              [profile rtol]
        (c) |diff| <= 2 * ulp(ref_g)               [profile max_ulp]
        (d) |diff| <= L_g * 2**-53 * sum(|a|_g)    [accumulation order]
    * f64 ORDER   -- still checked EXACTLY. `_sequential_sums` recomputes
      each run as a strict left-to-right f64 sum in Python and the kernel
      output must be BIT-IDENTICAL to it. This is exact today (0.0 on
      27/27 configurations, DLL and WASM), so it costs nothing and it is
      the assertion that actually pins the kernels: if the Rust or WASM
      accumulation order ever changes, this fails loudly instead of being
      absorbed by the tolerance above. A tolerance that can hide an order
      change is not a gate, so the exact check is retained on purpose.
    * ukeys, counts, ng -- BIT-EXACT on every lane, unchanged.

WHAT THIS DOES NOT CLAIM
    It does not claim the two f64 implementations round the same. They do
    not, and they should not: making them bit-identical would mean putting
    numpy's pairwise block structure inside `sorted.rs`, which is a change
    to production code and a different design decision. Recorded, not taken.

STALE COMMENT STILL IN PRODUCTION (found by this revision, NOT fixed here):
    numfast-native/src/groupby/sorted.rs:12-13 reads "Float accumulation
    order matches the numba kernel (sequential within run) -> bit-exact vs
    the prod kernel". Both halves are false at HEAD: the numba kernel was
    retired, and the prod kernel is `np.add.reduceat`, which is not
    sequential. The claim is what this file used to inherit.
"""
import ctypes
import json
import os
import subprocess
import tempfile

import numpy as np

import vectors_meta

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VEC = os.path.join(ROOT, "vectors")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
NODE = r"C:\Program Files\nodejs\node.exe"
import wasm_artifact
# The CURRENT build, validated: 86 functions + 1 memory. wasm_artifact.resolve()
# aborts loudly on a missing or stale .wasm rather than falling back --
# four parity scripts used to validate the tracked 56-function copy and
# report green, which is how a suite comes to prove the wrong bytes.
WASM = wasm_artifact.resolve()
print(wasm_artifact.banner(WASM))
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
MJS = os.path.join(ROOT, "tools", "wasm_sorted.mjs")

# --- f64 contract constants. Derivation and the measurement that forced them
# --- are in the module docstring; they are named so no reader can mistake one
# --- of them for a fitted number.
# specs/core/conformance-profile.toml, [tolerance.f64].
ATOL_F64 = 1e-12
RTOL_F64 = 1e-12
MAX_ULP_F64 = 2
# Machine unit roundoff of binary64: half the machine epsilon (2**-52).
U64 = float(np.finfo(np.float64).eps) / 2.0


def f64_ok(diff, ref, run_len, sum_abs):
    """The decided f64 contract, per group.

    Branches (a)(b)(c) are the project's published profile; (d) is the
    accumulation-order bound. Returns (pass, report) where `report` counts
    groups by WHICH set of branches admitted them, plus `ratio` = max over
    groups of |diff| / (L*u64*sum|a|), i.e. how much of branch (d) was
    actually consumed -- 1.0 would mean the bound was exactly saturated.
    """
    diff = np.asarray(diff, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    bound = np.asarray(run_len, dtype=np.float64) * U64 * np.asarray(sum_abs, dtype=np.float64)
    profile = ((diff <= ATOL_F64)
               | (diff <= RTOL_F64 * np.abs(ref))
               | (diff <= MAX_ULP_F64 * np.spacing(np.abs(ref))))
    accum = diff <= bound
    good = profile | accum
    ratio = float(np.max(diff / np.maximum(bound, np.float64(np.finfo(np.float64).tiny))))
    report = {
        "groups_admitted_by_profile_only": int((profile & ~accum).sum()),
        "groups_admitted_by_accum_only": int((accum & ~profile).sum()),
        "groups_admitted_by_both": int((profile & accum).sum()),
        "groups_rejected": int((~good).sum()),
        "profile_alone_would_pass": bool(profile.all()),
    }
    return bool(good.all()), report, ratio


def _sequential_sums(keys, v):
    """Strict left-to-right f64 accumulation, run by run, in Python floats.

    This is the accumulation order `sorted.rs` implements. Recomputing it
    here is what lets the f64 lane keep an EXACT assertion while the
    cross-implementation comparison moves to a tolerance: "different order,
    same quantity" stays distinguishable from "different quantity".
    """
    starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    ends = np.concatenate((starts[1:], [v.size]))
    vl = v.tolist()
    out = np.empty(starts.size, dtype=np.float64)
    for g, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
        acc = vl[a]
        for i in range(a + 1, b):
            acc = acc + vl[i]
        out[g] = acc
    return out


# Resolved against THIS checkout: gen.py writes basenames, and a
# meta.json carrying another machine's absolute path used to kill this
# script with a FileNotFoundError that read like a parity failure.
meta = vectors_meta.load("small")
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

# f64: the production kernel and `ref_sf` are the SAME array -- both are
# np.add.reduceat (groupindex.py:298). There is no numba order-reference
# left to compare against; the old comment claimed otherwise.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "gi_prod", os.path.join(ROOT, "..", "src", "Drivers", "CPU", "_lib", "groupindex.py"))
_gi = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_gi)
pok, pay, _, _ = _gi.sorted_fused_aggregate(keys, [vf])
assert pok, "prod numba kernel refused sorted input"
numba_s = np.asarray(pay[2], dtype=np.float64)
assert np.array_equal(numba_s, ref_sf), \
    "prod kernel and np.add.reduceat diverged; the f64 contract below assumes they are one computation"

run_len = np.diff(np.append(starts, n)).astype(np.float64)
sum_abs = np.add.reduceat(np.abs(vf), starts)
seq_ref = _sequential_sums(keys, vf)
lanes = {}
for name, ng, uk, s, c in [
        ("rust-f64", ng_f, uk_f[:max(ng_f, 0)], s_f[:max(ng_f, 0)], c_f[:max(ng_f, 0)]),
        ("wasm-f64", ng_wf, w_uk_f, w_s_f, w_c_f)]:
    shape = ng == ng_ref and (uk == uk_ref).all() and (c == ref_c).all()
    if not shape:
        ok = False
        print("%-10s ng=%d SHAPE-DIFF" % (name, ng))
        continue
    # EXACT: the kernel's accumulation ORDER is pinned, not just its magnitude.
    bit_order = bool(np.array_equal(s, seq_ref))
    # TOLERANT: the contract, per group.
    good, report, ratio = f64_ok(np.abs(s - numba_s), numba_s, run_len, sum_abs)
    dmax = float(np.max(np.abs(s - numba_s)))
    lanes[name] = {"max_abs_diff": dmax, "order_bit_identical": bit_order,
                   "accum_bound_ratio": ratio, "admission": report}
    bit = bit_order and good
    ok &= bit
    print("%-10s ng=%d max|dsum|=%.6g vs-prod  ratio(d)=%.4g  order=BIT-IDENTICAL:%s  %s"
          % (name, ng, dmax, ratio, bit_order, "OK" if bit else "DIFF"))
    print("           profile-only=%d accum-only=%d both=%d rejected=%d"
          % (report["groups_admitted_by_profile_only"],
             report["groups_admitted_by_accum_only"],
             report["groups_admitted_by_both"], report["groups_rejected"]))
print("err contract: unsorted=-1 ok, null=-2 ok, empty=0 ok")
json.dump({"n": n, "ng": int(ng_ref), "pass": bool(ok),
           "contract": {
               "i64": "bit-exact",
               "f64": ("not bit-exact; per group |diff| <= max(atol=1e-12, "
                       "rtol=1e-12*|ref|, 2 ulp, L*2**-53*sum|a|); the profile's "
                       "three branches (specs/core/conformance-profile.toml "
                       "[tolerance.f64]) plus the accumulation-order bound, "
                       "because rtol*|ref| collapses on a near-cancelling sum "
                       "and a legal input defeats it (measured 2.76e-08 at "
                       "L=100001). See the parity_sorted.py docstring."),
               "f64_order": "still bit-exact vs a strict sequential f64 sum",
               "u64": U64},
           "lanes": lanes},
          open(os.path.join(RES, "parity_sorted.json"), "w"), indent=1)
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
