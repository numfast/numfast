# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research bench: scaled-int (Task 1) + multi-variant DLL (Task 2).

Q3-10M shape, synthetic, seed 42. No H2O data, no GPU, no 1B rows.
One DLL handle, runtime variant select, no rebuild between variants.

Usage (Git Bash):
  python tests/research/scaled_mv/bench_scaled_mv.py [--reps R]
"""
import ctypes
import gc
import json
import os
import sys
import time

import numpy as np

REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3
N = 10_000_000
G = 100_000  # Q3-10M gold ngroups
M = 1_000_000  # v3 scale: ticks per unit
FORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
DLL = os.path.join(FORK, "numfast-native", "target", "x86_64-pc-windows-gnu",
                   "release", "numfast_native.dll")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_scaled_mv.json")
INT64_MAX = 2 ** 63 - 1

try:
    import psutil
    _PROC = psutil.Process()
    def rss_gb():
        return _PROC.memory_info().rss / 1e9
except ImportError:
    def rss_gb():
        return float("nan")

STAGES = {}


def stage(name, ms):
    STAGES[name] = ms
    print("  [stage] %-22s %10.1f ms" % (name, ms), flush=True)


def med(xs):
    return sorted(xs)[len(xs) // 2]


def best_of(fn, reps=REPS):
    fn()  # warmup
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return med(ts), min(ts)


def gbs(nbytes, ms):
    return (nbytes / 1e9) / (ms / 1e3)


print("== gen (seed 42, synthetic Q3-10M, G=%d) ==" % G, flush=True)
print("rss_start=%.2fGB" % rss_gb(), flush=True)
t0 = time.perf_counter()
rng = np.random.default_rng(42)
keys = np.ascontiguousarray(rng.integers(0, G, size=N, dtype=np.int64).astype(np.int32))
ticks = np.ascontiguousarray(np.rint(rng.normal(5000.0, 1500.0, N) * M).astype(np.int64))
vals = np.ascontiguousarray(ticks.astype(np.float64) / M)
stage("gen_keys_ticks_f64", (time.perf_counter() - t0) * 1e3)
print("rss_after_gen=%.2fGB neg_ticks=%d max|tick|=%d" %
      (rss_gb(), int((ticks < 0).sum()), int(np.abs(ticks).max())), flush=True)

# Range proof: total and per-group fits int64 (explicit, before any bench).
assert int(N) * int(np.abs(ticks).max()) < INT64_MAX, "int64 range proof FAILED"
print("range_proof: N*max|tick|=%e < INT64_MAX=%e OK" %
      (float(N) * float(np.abs(ticks).max()), float(INT64_MAX)), flush=True)

print("== reference (exact, int64 bincount + bigint spot) ==", flush=True)
t0 = time.perf_counter()
ref_sums = np.bincount(keys, weights=ticks, minlength=G)
ref_counts = np.bincount(keys, minlength=G).astype(np.int64)
stage("ref_bincount", (time.perf_counter() - t0) * 1e3)
t0 = time.perf_counter()
for gspot in (0, 1, 7, 4242, G - 1):
    m = keys == gspot
    assert int(ticks[m].astype(object).sum()) == int(ref_sums[gspot]), gspot
    assert int(m.sum()) == int(ref_counts[gspot]), gspot
stage("ref_bigint_spot5", (time.perf_counter() - t0) * 1e3)
assert ref_counts.sum() == N and (ref_counts > 0).all()

print("== load ONE dll ==", flush=True)
d = ctypes.CDLL(DLL)
f_f64 = d.nf_group_variant_f64
f_f64.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t, ctypes.c_uint32] + \
    [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
f_f64.restype = ctypes.c_int32
f_i64 = d.nf_group_variant_i64
f_i64.argtypes = f_f64.argtypes
f_i64.restype = ctypes.c_int32
f_i32 = d.nf_group_variant_i32
f_i32.argtypes = f_f64.argtypes
f_i32.restype = ctypes.c_int32
f_legacy = d.nf_group_sum_count  # frozen incumbent f64 path
f_legacy.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + \
    [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
f_legacy.restype = ctypes.c_int32

s_f64 = np.zeros(G)
c_f64 = np.zeros(G, dtype=np.int64)
s_i64 = np.zeros(G, dtype=np.int64)
c_i64 = np.zeros(G, dtype=np.int64)

print("== Task1: f64 path vs scaled-int path (correctness FIRST) ==", flush=True)


def run_legacy_f64():
    assert f_legacy(keys.ctypes.data, vals.ctypes.data, N,
                    s_f64.ctypes.data, c_f64.ctypes.data, G) == 0


def run_int_dense():
    assert f_i64(keys.ctypes.data, ticks.ctypes.data, N, 0,
                 s_i64.ctypes.data, c_i64.ctypes.data, G) == 0


run_legacy_f64()
run_int_dense()
assert (c_f64 == ref_counts).all() and (c_i64 == ref_counts).all(), "counts DIFF"
int_tick_diff = np.abs(s_i64.astype(np.int64) - ref_sums).max()
print("scaled_int_sum: max|diff| vs bigint-ref = %d ticks (exact=%s)" %
      (int(int_tick_diff), int_tick_diff == 0), flush=True)
assert int_tick_diff == 0, "scaled sum NOT exact"
f64_tick_err = np.abs(np.rint(s_f64 * M) - ref_sums).max()
print("f64_sum: max|round(sum*1e6)-exact| = %.1f ticks" % float(f64_tick_err), flush=True)
# mean at boundary: single rounding each; int path must match exact-mean rounding.
exact_mean = ref_sums / np.maximum(ref_counts, 1) / M
mean_int = s_i64 / np.maximum(c_i64, 1) / M
mean_f64 = s_f64 / np.maximum(c_f64, 1)
print("mean|maxdiff| int-path=%.3e f64-path=%.3e (vs single-rounded exact)" %
      (float(np.abs(mean_int - exact_mean).max()),
       float(np.abs(mean_f64 - exact_mean).max())), flush=True)

t_f64, t_f64min = best_of(run_legacy_f64)
t_int, t_intmin = best_of(run_int_dense)
in_bytes = keys.nbytes + vals.nbytes
stage("agg_f64_legacy_dense", t_f64)
stage("agg_i64_scaled_dense", t_int)
print("f64 : med=%.1fms min=%.1fms %.1fMrows/s %.2fGB/s" %
      (t_f64, t_f64min, N / t_f64 / 1e3, gbs(in_bytes, t_f64)), flush=True)
print("int : med=%.1fms min=%.1fms %.1fMrows/s %.2fGB/s" %
      (t_int, t_intmin, N / t_int / 1e3, gbs(in_bytes, t_int)), flush=True)
print("TIME_RATIO int/f64 = %.3f" % (t_int / t_f64), flush=True)
print("memory: keys=%.0fMB ticks=%.0fMB f64vals=%.0fMB sums+counts=%.1fMB (int path: SAME footprint as f64 path)" %
      (keys.nbytes / 2**20, ticks.nbytes / 2**20, vals.nbytes / 2**20,
       (s_i64.nbytes + c_i64.nbytes) / 2**20), flush=True)
print("rss_after_task1=%.2fGB" % rss_gb(), flush=True)

print("== Task2: multi-variant, ONE dll, no rebuild ==", flush=True)
t0 = time.perf_counter()
order = np.argsort(keys, kind="stable")
skeys = np.ascontiguousarray(keys[order])
sticks = np.ascontiguousarray(ticks[order])
svals = np.ascontiguousarray(vals[order])
del order
gc.collect()
stage("stable_sort_once", (time.perf_counter() - t0) * 1e3)

so = np.zeros(G)
co = np.zeros(G, dtype=np.int64)
si = np.zeros(G, dtype=np.int64)
ci = np.zeros(G, dtype=np.int64)


def call_f64(v, kk, vv):
    assert f_f64(kk.ctypes.data, vv.ctypes.data, N, v,
                 so.ctypes.data, co.ctypes.data, G) == 0
    return so.copy(), co.copy()


def call_i64(v, kk, tt):
    assert f_i64(kk.ctypes.data, tt.ctypes.data, N, v,
                 si.ctypes.data, ci.ctypes.data, G) == 0
    return si.copy(), ci.copy()


# parity BEFORE timing (golden = ref)
f64_vars = {v: call_f64(v, keys, vals) for v in (0, 1)}
f64_vars[2] = call_f64(2, skeys, svals)
i64_vars = {v: call_i64(v, keys, ticks) for v in (0, 1)}
i64_vars[2] = call_i64(2, skeys, sticks)
for v, (s, c) in i64_vars.items():
    assert (s == ref_sums).all() and (c == ref_counts).all(), "i64 v=%d DIFF" % v
print("i64 variants 0/1/2: bit-EXACT vs ref (traversal order irrelevant)", flush=True)
for v, (s, c) in f64_vars.items():
    assert (c == ref_counts).all(), "f64 v=%d counts DIFF" % v
d01 = float(np.abs(f64_vars[0][0] - f64_vars[1][0]).max())
d02 = float(np.abs(f64_vars[0][0] - f64_vars[2][0]).max())
print("f64 v0==v1 max|diff|=%.1e (same order, expect 0); v0 vs v2(sorted) max|diff|=%.3e (ULP tolerance)" % (d01, d02), flush=True)
assert d01 == 0.0 and d02 < 1e-6 * float(np.abs(f64_vars[0][0]).max())

VAR_RES = {}
for v, kk, vv in ((0, keys, vals), (1, keys, vals)):
    t, tmin = best_of(lambda: call_f64(v, kk, vv))
    stage("variant_f64_v%d" % v, t)
    VAR_RES["f64_v%d" % v] = {"med_ms": t, "min_ms": tmin}
t, tmin = best_of(lambda: call_f64(2, skeys, svals))
stage("variant_f64_v2_sorted", t)
VAR_RES["f64_v2"] = {"med_ms": t, "min_ms": tmin}
for v, kk, tt in ((0, keys, ticks), (1, keys, ticks)):
    t, tmin = best_of(lambda: call_i64(v, kk, tt))
    stage("variant_i64_v%d" % v, t)
    VAR_RES["i64_v%d" % v] = {"med_ms": t, "min_ms": tmin}
t, tmin = best_of(lambda: call_i64(2, skeys, sticks))
stage("variant_i64_v2_sorted", t)
VAR_RES["i64_v2"] = {"med_ms": t, "min_ms": tmin}

print("== error paths (explicit codes, no panic) ==", flush=True)
so2 = np.zeros(G)
co2 = np.zeros(G, dtype=np.int64)
rc = f_f64(keys.ctypes.data, vals.ctypes.data, N, 99, so2.ctypes.data, co2.ctypes.data, G)
print("bad variant -> %d (expect -3)" % rc, flush=True)
assert rc == -3
rc = f_i64(keys.ctypes.data, ticks.ctypes.data, N, 2, si.ctypes.data, ci.ctypes.data, G)
print("V_SORTED on unsorted keys -> %d (expect -5)" % rc, flush=True)
assert rc == -5
kb = np.array([0, 1, G], dtype=np.int32)  # G = out of range
tb = np.array([1, 2, 3], dtype=np.int64)
sb = np.zeros(G, dtype=np.int64)
cb = np.zeros(G, dtype=np.int64)
rc = f_i64(kb.ctypes.data, tb.ctypes.data, 3, 1, sb.ctypes.data, cb.ctypes.data, G)
print("V_CHECKED out-of-range key -> %d (expect -2)" % rc, flush=True)
assert rc == -2
# int32 overflow demo on 10M: per-value fits int32 (ticks//10 <= ~1.3e9),
# per-group sums (~100 rows x ~5e8 = ~5e10) overflow int32 by design -> -4
t32 = np.ascontiguousarray((ticks // 10).astype(np.int32))
s32 = np.zeros(G, dtype=np.int32)
t0 = time.perf_counter()
rc = f_i32(keys.ctypes.data, t32.ctypes.data, N, 0, s32.ctypes.data, ci.ctypes.data, G)
ov_ms = (time.perf_counter() - t0) * 1e3
print("i32 V_DENSE on 10M -> %d (expect -4 OVERFLOW, %.1fms to abort)" % (rc, ov_ms), flush=True)
assert rc == -4
# int32 tiny exact demo: small ticks fit -> parity vs int64
rng2 = np.random.default_rng(7)
nk, ng = 1000, 16
kk = np.ascontiguousarray(rng2.integers(0, ng, size=nk).astype(np.int32))
tt = np.ascontiguousarray(rng2.integers(-1000, 1000, size=nk).astype(np.int32))
s32t = np.zeros(ng, dtype=np.int32)
ct = np.zeros(ng, dtype=np.int64)
assert f_i32(kk.ctypes.data, tt.ctypes.data, nk, 0, s32t.ctypes.data, ct.ctypes.data, ng) == 0
ref32 = np.bincount(kk, weights=tt.astype(np.int64), minlength=ng)
assert (s32t.astype(np.int64) == ref32).all(), "i32 tiny parity FAILED"
print("i32 tiny-1k: EXACT vs ref (fits) — contract proven both ways", flush=True)

res = {"n": N, "g": G, "scale": M, "seed": 42, "reps": REPS,
       "stages_ms": STAGES,
       "task1": {"f64_med_ms": STAGES["agg_f64_legacy_dense"],
                 "int_med_ms": STAGES["agg_i64_scaled_dense"],
                 "time_ratio_int_f64": STAGES["agg_i64_scaled_dense"] / STAGES["agg_f64_legacy_dense"],
                 "int_max_tick_diff": 0,
                 "f64_max_tick_err": float(f64_tick_err),
                 "mean_maxdiff_int": float(np.abs(mean_int - exact_mean).max()),
                 "mean_maxdiff_f64": float(np.abs(mean_f64 - exact_mean).max()),
                 "bytes": {"keys": int(keys.nbytes), "ticks": int(ticks.nbytes),
                           "f64vals": int(vals.nbytes)}},
       "task2_variants": VAR_RES,
       "task2_parity": {"i64_exact": True, "f64_v0_eq_v1": d01 == 0.0,
                        "f64_v0_vs_v2_sorted": d02},
       "error_paths": {"bad_variant": -3, "unsorted": -5, "oor": -2,
                       "i32_overflow_10M": -4, "i32_overflow_abort_ms": ov_ms,
                       "i32_tiny_exact": True},
       "rss_end_gb": rss_gb()}
json.dump(res, open(OUT, "w"), indent=1)
print("wrote %s rss_end=%.2fGB" % (OUT, rss_gb()), flush=True)
