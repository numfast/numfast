# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""T6: audit Python-элементов, переносимых в натив. Каждый кандидат измерен.
Seed 42. best-of-3. Research-only.
Git Bash: timeout 300 python tests/research/arch_leap/bench_t6_pyloops.py
"""
import ctypes, json, os, time
import numpy as np

REPS = 3
M = 1_000_000
HERE = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(HERE, "..", "..", "..", "numfast-native", "target",
                   "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
OUT = os.path.join(HERE, "results_t6.json")
try:
    import psutil; P = psutil.Process()
    def rss(): return P.memory_info().rss / 1e9
except ImportError:
    def rss(): return float("nan")
def med(xs): return sorted(xs)[len(xs)//2]
def best_of(fn, reps=REPS):
    fn(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1e3)
    return med(ts), min(ts)

d = ctypes.CDLL(DLL)
f_i64 = d.nf_group_variant_i64
f_i64.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_size_t, ctypes.c_uint32] + [ctypes.c_void_p]*2 + [ctypes.c_size_t]
f_i64.restype = ctypes.c_int32

N, G = 200_000, 10_000  # small so Python-loop candidates terminate
rng = np.random.default_rng(42)
keys = np.ascontiguousarray(rng.integers(0, G, size=N, dtype=np.int64).astype(np.int32))
lvals = rng.normal(5000.0, 1500.0, N)
ticks = np.ascontiguousarray(np.rint(lvals * M).astype(np.int64))
res = {"seed": 42, "n": N, "g": G}
print("rss_start=%.2fGB" % rss(), flush=True)

# C1: Python per-row loop vs native dense (extrapolate honestly, loop on 5k prefix)
K = 5_000
acc = np.zeros(G, dtype=np.int64)
def pyloop():
    acc[:] = 0
    for i in range(K):
        acc[keys[i]] += int(ticks[i])
t, tm = best_of(pyloop)
native_s = np.zeros(G, dtype=np.int64); native_c = np.zeros(G, dtype=np.int64)
kk = np.ascontiguousarray(keys[:K]); tt = np.ascontiguousarray(ticks[:K])
def nat(): assert f_i64(kk.ctypes.data, tt.ctypes.data, K, 0, native_s.ctypes.data, native_c.ctypes.data, G) == 0
tn, tnm = best_of(nat)
print("C1 pyloop-5k=%.1fms native-5k=%.3fms ratio=%.0fx" % (t, tn, t/tn), flush=True)
res["C1_pyloop_5k"] = {"py_ms": t, "native_ms": tn, "ratio": t/tn}

# C2: materialization copies (.copy/.astype) vs view
def copies(): return ticks.astype(np.float64) / M + keys.astype(np.int64)
t2, _ = best_of(copies)
def viewsum(): return ticks.sum() + keys.sum()
t2v, _ = best_of(viewsum)
print("C2 copies/convert=%.1fms view-reduce=%.1fms" % (t2, t2v), flush=True)
res["C2_materialize"] = {"copy_convert_ms": t2, "view_ms": t2v}

# C3: conversion float<->tick (rint+astype) — caller-side cost of scaled-int
def conv(): return np.rint(lvals * M).astype(np.int64)
t3, t3m = best_of(conv)
print("C3 float->tick conv=%.1fms (%.1f%% of 10M-scale agg ~90ms/10xN)" % (t3, t3), flush=True)
res["C3_conv"] = {"med_ms": t3, "min_ms": t3m}

# C4: dict/list groupby vs dense native (20k prefix)
K4 = 20_000
def dictgb():
    dd = {}
    for i in range(K4):
        k = int(keys[i]); dd[k] = dd.get(k, 0) + int(ticks[i])
    return dd
t4, _ = best_of(dictgb)
s4 = np.zeros(G, dtype=np.int64); c4 = np.zeros(G, dtype=np.int64)
k4 = np.ascontiguousarray(keys[:K4]); t4a = np.ascontiguousarray(ticks[:K4])
def nat4(): assert f_i64(k4.ctypes.data, t4a.ctypes.data, K4, 0, s4.ctypes.data, c4.ctypes.data, G) == 0
t4n, _ = best_of(nat4)
print("C4 dict-20k=%.1fms native-20k=%.2fms ratio=%.0fx" % (t4, t4n, t4/t4n), flush=True)
res["C4_dict"] = {"dict_ms": t4, "native_ms": t4n, "ratio": t4/t4n}

# C5: argsort prep cost vs kernel (the dominant Python-side element)
kk5 = np.ascontiguousarray(rng.integers(0, G, size=N, dtype=np.int64).astype(np.int32))
def srt(): return np.argsort(kk5, kind="stable")
t5, _ = best_of(srt)
s5 = np.zeros(G, dtype=np.int64); c5 = np.zeros(G, dtype=np.int64)
def nat5(): assert f_i64(kk5.ctypes.data, ticks.ctypes.data, N, 0, s5.ctypes.data, c5.ctypes.data, G) == 0
t5n, _ = best_of(nat5)
print("C5 argsort-200k=%.1fms dense-kernel=%.1fms sort/kernel=%.1fx" % (t5, t5n, t5/t5n), flush=True)
res["C5_argsort"] = {"sort_ms": t5, "kernel_ms": t5n, "ratio": t5/t5n}

# C6: list-of-runs materialization (np.diff+concatenate) vs in-native runs
def runs_py():
    chg = np.flatnonzero(np.diff(kk5) != 0)
    return chg.size
t6, _ = best_of(runs_py)
print("C6 diff-runs scan=%.2fms (pure numpy, no python loop)" % t6, flush=True)
res["C6_runs_scan"] = {"ms": t6}

res["rss_end_gb"] = rss()
json.dump(res, open(OUT, "w"), indent=1)
print("wrote %s rss=%.2fGB" % (OUT, rss()), flush=True)
