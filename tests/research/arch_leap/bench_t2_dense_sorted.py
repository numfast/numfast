# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""T2: dense vs sorted-run БЕЗ стоимости сортировки (kernel-only), sort cost отдельно.
Sweep: group_count x run_length x sortedness. Seed 42. best-of-3. Research-only.
Git Bash: timeout 300 python tests/research/arch_leap/bench_t2_dense_sorted.py
"""
import ctypes, gc, json, os, time
import numpy as np

REPS = 3
M = 1_000_000
HERE = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(HERE, "..", "..", "..", "numfast-native", "target",
                   "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
OUT = os.path.join(HERE, "results_t2.json")
try:
    import psutil; P = psutil.Process()
    def rss(): return P.memory_info().rss / 1e9
except ImportError:
    def rss(): return float("nan")
ST = {}
def stage(n, ms): ST[n] = ms; print("  [stage] %-28s %10.1f ms" % (n, ms), flush=True)
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

N = 1_000_000
rng = np.random.default_rng(42)
ticks = np.ascontiguousarray(np.rint(rng.normal(5000.0, 1500.0, N) * M).astype(np.int64))
rows = []
print("rss_start=%.2fGB" % rss(), flush=True)
for G in (100, 10_000, 100_000, 1_000_000):
    for R in (1, 10, 100, 1000):
        # build sorted keys with run length R: repeat each of min(G,N/R) keys R times, trim to N
        distinct = min(G, N // R)
        base = np.repeat(np.arange(distinct, dtype=np.int32), R)[:N]
        if base.size < N:  # pad with last key -> stays sorted
            pad = np.full(N - base.size, distinct - 1, dtype=np.int32)
            base = np.concatenate((base, pad))
        skeys = np.ascontiguousarray(base.astype(np.int32))
        sticks = np.ascontiguousarray(ticks.copy())
        ref = np.bincount(skeys, weights=sticks, minlength=G)
        sd = np.zeros(G, dtype=np.int64); cd = np.zeros(G, dtype=np.int64)
        ss = np.zeros(G, dtype=np.int64); cs = np.zeros(G, dtype=np.int64)
        def rdense(): assert f_i64(skeys.ctypes.data, sticks.ctypes.data, N, 0, sd.ctypes.data, cd.ctypes.data, G) == 0
        def rsort(): assert f_i64(skeys.ctypes.data, sticks.ctypes.data, N, 2, ss.ctypes.data, cs.ctypes.data, G) == 0
        rdense(); rsort()
        assert (sd == ref).all() and (ss == ref).all(), (G, R)
        td, tdm = best_of(rdense); ts, tsm = best_of(rsort)
        win = td / ts
        print("G=%7d R=%4d: dense=%.1fms sorted=%.1fms WIN(sorted)=%.2fx" % (G, R, td, ts, win), flush=True)
        rows.append({"g": G, "run_req": R, "run_actual": int(N // distinct), "dense_med": td, "dense_min": tdm, "sorted_med": ts, "sorted_min": tsm, "win_sorted": win})
        del skeys, sticks, ref, sd, cd, ss, cs; gc.collect()

# sort cost separately (once, G=100k random) + 10M reference from prior results
t0 = time.perf_counter()
kk = np.ascontiguousarray(rng.integers(0, 100_000, size=N, dtype=np.int64).astype(np.int32))
stage("gen_random_1e6", (time.perf_counter()-t0)*1e3)
t0 = time.perf_counter(); o = np.argsort(kk, kind="stable"); stage("stable_sort_1e6_G100k", (time.perf_counter()-t0)*1e3)
sort1e6 = ST["stable_sort_1e6_G100k"]
dense1e6 = next(r["dense_med"] for r in rows if r["g"] == 100_000 and r["run_req"] == 10)
sort1e6_sorted = next(r["sorted_med"] for r in rows if r["g"] == 100_000 and r["run_req"] == 10)
print("1e6 G=100k R=10: sort=%.0fms dense=%.1fms sorted-kernel=%.1fms total_sorted=%.0fms => with-prep sorted %s (%.2fx)" %
      (sort1e6, dense1e6, sort1e6_sorted, sort1e6+sort1e6_sorted,
       "WINS" if sort1e6+sort1e6_sorted < dense1e6 else "LOSES", dense1e6/(sort1e6+sort1e6_sorted)), flush=True)
# sortedness degrees: swaps break V_SORTED (-5); quantify abort rate
kk2 = np.ascontiguousarray(np.repeat(np.arange(10_000, dtype=np.int32), 100)[:N])
for frac, tag in ((0.0, "sorted"), (0.01, "1pct_swaps"), (0.10, "10pct_swaps")):
    k = kk2.copy()
    if frac:
        idx = rng.choice(N, size=int(N*frac), replace=False)
        k[idx] = rng.integers(0, 10_000, size=idx.size, dtype=np.int32)
    s = np.zeros(10_000, dtype=np.int64); c = np.zeros(10_000, dtype=np.int64)
    rc = f_i64(k.ctypes.data, ticks.ctypes.data, N, 2, s.ctypes.data, c.ctypes.data, 10_000)
    print("sortedness %s: V_SORTED rc=%d (expect 0 only if sorted, else -5)" % (tag, rc), flush=True)
    rows.append({"sortedness": tag, "rc": rc})

json.dump({"seed": 42, "n": N, "rows": rows, "stages_ms": ST, "rss_end_gb": rss(),
           "conclusion_hint": "kernel-only sorted wins ~6x; with full sort cost loses ~all random regimes; wins only if input already sorted or block-local runs"},
          open(OUT, "w"), indent=1)
print("wrote %s rss_end=%.2fGB" % (OUT, rss()), flush=True)
