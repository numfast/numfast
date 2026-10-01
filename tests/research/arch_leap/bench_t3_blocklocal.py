# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""T3: block-local run aggregation + cheap merge, без глобальной сортировки.
Strategies per block: (a) dense native + dense-add merge; (b) block-sort + sorted-run + merge;
(c) adjacent-RLE (no sort) + merge. Seed 42. best-of-3. Research-only.
Git Bash: timeout 600 python tests/research/arch_leap/bench_t3_blocklocal.py
"""
import ctypes, gc, json, os, time
import numpy as np

REPS = 3
M = 1_000_000
HERE = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(HERE, "..", "..", "..", "numfast-native", "target",
                   "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
OUT = os.path.join(HERE, "results_t3.json")
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

def dense_block(keys_b, ticks_b, G):
    s = np.zeros(G, dtype=np.int64); c = np.zeros(G, dtype=np.int64)
    assert f_i64(keys_b.ctypes.data, ticks_b.ctypes.data, keys_b.size, 0, s.ctypes.data, c.ctypes.data, G) == 0
    return s, c

def run_cfg(N, G, BS, seed=42):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    keys = np.ascontiguousarray(rng.integers(0, G, size=N, dtype=np.int64).astype(np.int32))
    ticks = np.ascontiguousarray(np.rint(rng.normal(5000.0, 1500.0, N) * M).astype(np.int64))
    gen_ms = (time.perf_counter()-t0)*1e3
    t0 = time.perf_counter()
    ref = np.bincount(keys, weights=ticks, minlength=G); refc = np.bincount(keys, minlength=G).astype(np.int64)
    ref_ms = (time.perf_counter()-t0)*1e3
    nb = (N + BS - 1)//BS
    out = {"n": N, "g": G, "bs": BS, "nblocks": nb, "gen_ms": gen_ms, "ref_ms": ref_ms}
    # (a) dense per block + add-merge
    def strata():
        gsa = np.zeros(G, dtype=np.int64); gca = np.zeros(G, dtype=np.int64)
        for b in range(nb):
            kb = np.ascontiguousarray(keys[b*BS:(b+1)*BS]); tb = np.ascontiguousarray(ticks[b*BS:(b+1)*BS])
            s, c = dense_block(kb, tb, G); gsa += s; gca += c
        return gsa, gca
    gs0, gc0 = strata()
    assert (gs0 == ref).all() and (gc0 == refc).all(), "dense-block DIFF"
    ta, tam = best_of(strata)
    out["dense_block_med"] = ta; out["dense_block_min"] = tam
    # (b) block-sort + sorted-run + merge
    def strsort():
        gsb = np.zeros(G, dtype=np.int64); gcb = np.zeros(G, dtype=np.int64)
        s = np.zeros(G, dtype=np.int64); c = np.zeros(G, dtype=np.int64)
        for b in range(nb):
            kb = keys[b*BS:(b+1)*BS]; tb = ticks[b*BS:(b+1)*BS]
            o = np.argsort(kb, kind="stable")
            sk = np.ascontiguousarray(kb[o]); st = np.ascontiguousarray(tb[o])
            assert f_i64(sk.ctypes.data, st.ctypes.data, sk.size, 2, s.ctypes.data, c.ctypes.data, G) == 0
            gsb += s; gcb += c; s[:] = 0; c[:] = 0
        return gsb, gcb
    gs2, gc2 = strsort()
    assert (gs2 == ref).all() and (gc2 == refc).all(), "blocksort DIFF"
    tb, tbm = best_of(strsort)
    out["blocksort_med"] = tb; out["blocksort_min"] = tbm
    # (c) adjacent-RLE per block (no sort): runs of equal neighbours -> run sums, merge via bincount on run keys
    def strrle():
        rk_all = []; rs_all = []; rc_all = []
        for b in range(nb):
            kb = keys[b*BS:(b+1)*BS]; tb2 = ticks[b*BS:(b+1)*BS]
            chg = np.flatnonzero(np.diff(kb) != 0) + 1
            starts = np.concatenate(([0], chg)); ends = np.concatenate((chg, [kb.size]))
            rk = kb[starts]; rs = np.add.reduceat(tb2, starts); rct = np.diff(np.concatenate((starts, [kb.size]))) if False else (ends - starts)
            rk_all.append(rk); rs_all.append(rs); rc_all.append(rct)
        RK = np.concatenate(rk_all); RS = np.concatenate(rs_all); RC = np.concatenate(rc_all)
        gs3 = np.bincount(RK, weights=RS, minlength=G); gc3 = np.bincount(RK, weights=RC, minlength=G).astype(np.int64)
        return gs3, gc3, RK.size
    gs3, gc3, nruns = strrle()
    assert (gs3 == ref).all() and (gc3 == refc).all(), "rle DIFF"
    tr, trm = best_of(lambda: strrle()[:2])
    out["rle_med"] = tr; out["rle_min"] = trm; out["rle_runs"] = int(nruns)
    # global-sort reference cost (one stable sort, for comparison only)
    t0 = time.perf_counter(); o = np.argsort(keys, kind="stable"); sort_ms = (time.perf_counter()-t0)*1e3
    del o; gc.collect()
    out["global_sort_ms"] = sort_ms
    print("N=%d G=%d BS=%d nb=%d: dense-blk=%.1f rle=%.1f blksort=%.1f global_sort=%.0f runs=%d rss=%.2f" %
          (N, G, BS, nb, ta, tr, tb, sort_ms, nruns, rss()), flush=True)
    return out

print("rss_start=%.2fGB" % rss(), flush=True)
rows = [run_cfg(1_000_000, 100_000, bs) for bs in (16_384, 65_536, 262_144)]
rows.append(run_cfg(10_000_000, 100_000, 65_536))
json.dump({"seed": 42, "rows": rows, "rss_end_gb": rss()}, open(OUT, "w"), indent=1)
print("wrote %s rss_end=%.2fGB" % (OUT, rss()), flush=True)
