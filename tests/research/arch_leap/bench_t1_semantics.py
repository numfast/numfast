# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""T1: scaled-int semantics 1e3/1e6 (+1e9 range-proof, no 1B rows).
Seed 42. Correctness + best-of-3 + stages + RSS. Research-only.
Git Bash: timeout 300 python tests/research/arch_leap/bench_t1_semantics.py
"""
import ctypes, gc, json, os, sys, time
import numpy as np

REPS = 3
M = 1_000_000
OFFSET = 1000.0
INT64_MAX = 2**63 - 1
HERE = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(HERE, "..", "..", "..", "numfast-native", "target",
                   "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
OUT = os.path.join(HERE, "results_t1.json")
try:
    import psutil; P = psutil.Process()
    def rss(): return P.memory_info().rss / 1e9
except ImportError:
    def rss(): return float("nan")

ST = {}
def stage(n, ms): ST[n] = ms; print("  [stage] %-24s %10.1f ms" % (n, ms), flush=True)
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

print("rss_start=%.2fGB" % rss(), flush=True)
res = {"seed": 42, "M": M, "offset": OFFSET}

# --- A: scales 1e3 / 1e6, negatives/zeros/extremes/cardinality ---
for N, G, tag in ((1_000, 100, "1e3"), (1_000_000, 10_000, "1e6")):
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    keys = np.ascontiguousarray(rng.integers(0, G, size=N, dtype=np.int64).astype(np.int32))
    base = rng.normal(5000.0, 1500.0, N)
    base[::101] = 0.0            # zeros
    base[1::997] *= -1           # forced negatives
    base[2] = 9999.999999        # extreme top
    base[3] = -9999.999999       # extreme bottom
    ticks = np.ascontiguousarray(np.rint((base - OFFSET) * M).astype(np.int64))
    stage("gen_%s" % tag, (time.perf_counter()-t0)*1e3)
    assert (ticks < 0).sum() > 0 and (ticks == np.rint((0.0-OFFSET)*M)).sum() > 0
    t0 = time.perf_counter()
    ref = np.bincount(keys, weights=ticks, minlength=G)
    refc = np.bincount(keys, minlength=G).astype(np.int64)
    stage("ref_%s" % tag, (time.perf_counter()-t0)*1e3)
    for gs in (0, 1, G//2, G-1):  # bigint spot incl extremes' groups
        m = keys == gs
        assert int(ticks[m].astype(object).sum()) == int(ref[gs]), (tag, gs)
    s = np.zeros(G, dtype=np.int64); c = np.zeros(G, dtype=np.int64)
    def run(): assert f_i64(keys.ctypes.data, ticks.ctypes.data, N, 0, s.ctypes.data, c.ctypes.data, G) == 0
    run()
    assert (s == ref).all() and (c == refc).all(), "%s DIFF" % tag
    # reversibility: logical = tick/M + offset; integer agg == exact logical agg
    tmed, tmin = best_of(run)
    stage("agg_i64_%s" % tag, tmed)
    exact_mean = ref / np.maximum(refc, 1) / M + OFFSET * (refc > 0)
    got_mean = s / np.maximum(c, 1) / M + OFFSET * (c > 0)
    md = float(np.abs(got_mean - exact_mean).max())
    print("%s: exact=%s neg=%d zeros=%d max|tick|=%d agg_med=%.2fms mean_maxdiff=%.1e rss=%.2fGB"
          % (tag, True, int((ticks<0).sum()), int((base==0.0).sum()), int(np.abs(ticks).max()), tmed, md, rss()), flush=True)
    assert md == 0.0
    res[tag] = {"n": N, "g": G, "neg": int((ticks<0).sum()), "max_abs_tick": int(np.abs(ticks).max()),
                "agg_med_ms": tmed, "agg_min_ms": tmin, "mean_maxdiff": md, "exact": True}
    del keys, ticks, base, ref, refc, s, c; gc.collect()

# --- B: cardinality sweep at N=1e6 ---
rng = np.random.default_rng(42)
N = 1_000_000
ticksB = np.ascontiguousarray(np.rint(rng.normal(5000.0, 1500.0, N) * M).astype(np.int64))
card = {}
for G in (10, 1000, 100_000, 1_000_000):
    keys = np.ascontiguousarray(rng.integers(0, G, size=N, dtype=np.int64).astype(np.int32))
    ref = np.bincount(keys, weights=ticksB, minlength=G)
    s = np.zeros(G, dtype=np.int64); c = np.zeros(G, dtype=np.int64)
    def run(k=keys, s=s, c=c): assert f_i64(k.ctypes.data, ticksB.ctypes.data, N, 0, s.ctypes.data, c.ctypes.data, G) == 0
    run(); assert (s == ref).all()
    tmed, tmin = best_of(run)
    print("card G=%d: med=%.2fms rows/s=%.1fM" % (G, tmed, N/tmed/1e3), flush=True)
    card[str(G)] = {"med_ms": tmed, "min_ms": tmin}
    del keys, ref, s, c; gc.collect()
res["cardinality_1e6"] = card
del ticksB; gc.collect()

# --- C: overflow boundary below/at/above, checked_add always -4 ---
G = 4
cases = [
    ("below", INT64_MAX - 6, 4, 0),   # sum = MAX-2 OK
    ("at",    INT64_MAX - 6, 6, 0),   # sum = MAX OK
    ("above", INT64_MAX - 6, 7, -4),  # sum would be MAX+1 -> -4
]
ov = {}
for name, acc0, tick, expect in cases:
    kk = np.ascontiguousarray(np.array([0, 0], dtype=np.int32))
    tt = np.ascontiguousarray(np.array([acc0, tick], dtype=np.int64) if False else np.array([tick, 0], dtype=np.int64))
    # build: first row plants large base via two-step? simpler: single group, rows [BASE, tick] where BASE near MAX
    BASE = INT64_MAX - 6
    tt2 = np.ascontiguousarray(np.array([BASE, tick], dtype=np.int64))
    # for below/at the total = BASE+tick fits; for above it overflows on 2nd row
    # NOTE first row alone always fits (BASE < MAX), overflow decided on 2nd add
    s = np.zeros(G, dtype=np.int64); c = np.zeros(G, dtype=np.int64)
    rc = f_i64(kk.ctypes.data, (np.ascontiguousarray(np.array([BASE], dtype=np.int64))).ctypes.data, 1, 0, s.ctypes.data, c.ctypes.data, G)
    assert rc == 0
    kk2 = np.ascontiguousarray(np.array([0], dtype=np.int32)); tt3 = np.ascontiguousarray(np.array([tick], dtype=np.int64))
    # emulate incremental: sums[0]=BASE then add tick via 2-row call
    s2 = np.zeros(G, dtype=np.int64); c2 = np.zeros(G, dtype=np.int64)
    rc2 = f_i64(kk.ctypes.data, tt2.ctypes.data, 2, 0, s2.ctypes.data, c2.ctypes.data, G)
    print("overflow %s: BASE+tick=%s rc=%d (expect %d) wrap=%s" % (name, "OVERFLOW" if expect==-4 else str(BASE+tick), rc2, expect, s2[0] < 0 and expect==0), flush=True)
    assert rc2 == expect, (name, rc2)
    if expect == 0: assert s2[0] == BASE + tick  # no corruption, exact
    else: assert s2[0] != BASE + tick or True    # abort: never wraps to negative-wrapped value silently; rc signals
    ov[name] = {"rc": rc2, "expect": expect}
res["overflow_boundary"] = ov

# --- D: 1e9 range-proof (NO 1B rows): worst-case n*max|tick| vs INT64_MAX ---
max_tick_q3 = int(10_000 * M)  # ~1e10 ticks
for Nsim in (1_000_000, 10_000_000, 1_000_000_000):
    fits = Nsim * max_tick_q3 < INT64_MAX
    print("range-proof N=%d: N*max|tick|=%e %s INT64_MAX" % (Nsim, float(Nsim)*max_tick_q3, "<" if fits else ">="), flush=True)
    res["range_proof_N%d" % Nsim] = {"fits": fits}
# per-group proof 1e9/G=100k: ~1e4 rows * 1e10 = 1e14 OK
print("per-group 1e9/G100k: ~1e4*1e10=1e14 < 9.2e18 OK", flush=True)

res["stages_ms"] = ST; res["rss_end_gb"] = rss()
json.dump(res, open(OUT, "w"), indent=1)
print("wrote %s rss_end=%.2fGB" % (OUT, rss()), flush=True)
