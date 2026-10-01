# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""T5: SUM+COUNT+MEAN int-only hot loop (div после merge) + Q4/Q5 multi-SoA bottleneck.
Seed 42. best-of-3. Research-only.
Git Bash: timeout 600 python tests/research/arch_leap/bench_t5_mean_multisoa.py
"""
import ctypes, gc, json, os, time
import numpy as np

REPS = 3
M = 1_000_000
HERE = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(HERE, "..", "..", "..", "numfast-native", "target",
                   "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
OUT = os.path.join(HERE, "results_t5.json")
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
f_1 = d.nf_group_sum_count
f_1.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_size_t] + [ctypes.c_void_p]*2 + [ctypes.c_size_t]
f_1.restype = ctypes.c_int32
f_m = d.nf_group_multi_sum_count
# multi signature: (keys, vals_ptr_array?, ncols?, n, sums_ptr_array?, counts, g)? probe via docstring-less: try standard
# lib.rs line 82 — read at runtime: attempt (keys,ncols,n,vals...,sums...,counts,g) is unknown; use brute try below
print("rss_start=%.2fGB" % rss(), flush=True)
N, G = 1_000_000, 100_000
rng = np.random.default_rng(42)
t0 = time.perf_counter()
keys = np.ascontiguousarray(rng.integers(0, G, size=N, dtype=np.int64).astype(np.int32))
ticks = np.ascontiguousarray(np.rint(rng.normal(5000.0, 1500.0, N) * M).astype(np.int64))
vals = np.ascontiguousarray(ticks.astype(np.float64) / M)
gen_ms = (time.perf_counter()-t0)*1e3

# A: int hot loop (native i64) + div-after-merge at boundary vs f64 hot loop
si = np.zeros(G, dtype=np.int64); ci = np.zeros(G, dtype=np.int64)
sf = np.zeros(G); cf = np.zeros(G, dtype=np.int64)
def run_int(): assert f_i64(keys.ctypes.data, ticks.ctypes.data, N, 0, si.ctypes.data, ci.ctypes.data, G) == 0
def run_f64(): assert f_1(keys.ctypes.data, vals.ctypes.data, N, sf.ctypes.data, cf.ctypes.data, G) == 0
run_int(); run_f64()
ref = np.bincount(keys, weights=ticks, minlength=G)
assert (si == ref).all()
t0 = time.perf_counter(); mean_int = si / np.maximum(ci, 1) / M; div_int_ms = (time.perf_counter()-t0)*1e3
t0 = time.perf_counter(); mean_f64 = sf / np.maximum(cf, 1); div_f64_ms = (time.perf_counter()-t0)*1e3
assert float(np.abs(mean_int - ref/np.maximum(ci,1)/M).max()) == 0.0
ti, tim = best_of(run_int); tf, tfm = best_of(run_f64)
print("hotloop int=%.1fms div-after=%.2fms | f64=%.1fms div=%.2fms | div share int=%.2f%%" % (ti, div_int_ms, tf, div_f64_ms, div_int_ms/ti*100), flush=True)

# B: multi-SoA bottleneck: 3 cols — 3x single-col calls vs python-side stacked timing
t3 = [np.ascontiguousarray(np.rint(rng.normal(5000.0, 1500.0, N) * M).astype(np.int64)) for _ in range(3)]
bufs = [np.zeros(G, dtype=np.int64) for _ in range(3)]
bc = np.zeros(G, dtype=np.int64)
bc_dummy = np.zeros(G, dtype=np.int64)  # persistent; never read
def run3x():
    for j in range(3):
        assert f_i64(keys.ctypes.data, t3[j].ctypes.data, N, 0, bufs[j].ctypes.data, (bc if j == 0 else bc_dummy).ctypes.data, G) == 0
run3x()
t3x, t3xm = best_of(run3x)
# prep cost: 3x tick buffers bytes vs 1x
prep_bytes = keys.nbytes + sum(t.nbytes for t in t3)
print("multi-SoA 3x-single: med=%.1fms (%.2fx of 1-col %.1fms) prep_bytes=%.0fMB" % (t3x, t3x/ti, ti, prep_bytes/2**20), flush=True)
# try true fused multi symbol: (keys, values f64 SoA ncols*n, n, ncols, sums ncols*g, counts, g)
multi_info = {"attempted": True}
try:
    f_m.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_size_t]*2 + [ctypes.c_void_p]*2 + [ctypes.c_size_t]
    vals3 = np.ascontiguousarray(np.vstack([t.astype(np.float64)/M for t in t3]))
    sums3 = np.zeros((3, G)); cc = np.zeros(G, dtype=np.int64)
    t0 = time.perf_counter()
    rc = f_m(keys.ctypes.data, vals3.ctypes.data, N, 3, sums3.ctypes.data, cc.ctypes.data, G)
    fused_ms = (time.perf_counter()-t0)*1e3
    # correctness vs 3x single-col f64 ref
    ref3 = np.vstack([np.bincount(keys, weights=t.astype(np.float64)/M, minlength=G) for t in t3])
    ok3 = (np.abs(sums3 - ref3).max() < 1e-9 * np.abs(ref3).max()) and (cc == np.bincount(keys, minlength=G)).all()
    multi_info.update({"rc": rc, "ms": fused_ms, "exact": bool(ok3), "note": "keys,vals(f64 SoA),n,ncols,sums,counts,g"})
    print("fused multi rc=%s ms=%.1f exact=%s" % (rc, fused_ms, ok3), flush=True)
    t0 = time.perf_counter()
    for j in range(5):
        rc = f_m(keys.ctypes.data, vals3.ctypes.data, N, 3, sums3.ctypes.data, cc.ctypes.data, G)
    fused_rep_ms = (time.perf_counter()-t0)*1e3/5
    multi_info["rep_ms"] = fused_rep_ms
    print("fused multi rep-avg=%.1fms vs 3x-i64-single=%.1fms" % (fused_rep_ms, t3x), flush=True)
except Exception as e:
    multi_info.update({"error": str(e)[:200]})
    print("fused multi attempt failed: %s" % str(e)[:200], flush=True)

json.dump({"seed": 42, "n": N, "g": G, "gen_ms": gen_ms,
           "int_hot_med": ti, "int_hot_min": tim, "div_after_ms": div_int_ms,
           "f64_hot_med": tf, "f64_hot_min": tfm, "div_f64_ms": div_f64_ms,
           "multi_3x_med": t3x, "multi_3x_min": t3xm, "prep_bytes": prep_bytes,
           "multi_fused": multi_info, "rss_end_gb": rss(),
           "bottleneck_hint": "see report: prep bytes 3x vs kernel 3x evaluated"},
          open(OUT, "w"), indent=1)
print("wrote %s rss=%.2fGB" % (OUT, rss()), flush=True)
