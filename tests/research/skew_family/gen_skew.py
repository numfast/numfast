# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research-only: synthetic skew family 10M (seed 42, deterministic).

Family (separate from official H2O data, official CSV untouched):
  uniform    : all keys uniform
  mild-skew  : zipf s=0.8 on id1/id2/id3/id4/id6 (id5 uniform, not queried)
  medium-skew: zipf s=1.2
  heavy-skew : zipf s=1.7

Cardinalities match G1_10M: id1=100 id2=100 id3=100000 id4=100 id5=100 id6=100000.
N=10M. v1/v2/v3 shared across family (seed 42 once): v1~U{1..5}, v2~U{1..15},
v3~U[0,100) float64. Keys per level: seed 42+level_idx (deterministic).
Probability law: p_i = (1/i^s)/H, i=1..K (rank1 hottest). Codes 0..K-1 map to
ranks in order (code0 hottest). Missing groups force-filled (1 row each) so
cardinality stays K; documented per dataset (n_forced).
Strings G1-compatible: id1/id2 "id%03d" (1..100), id3 "id%010d" (1..100000);
id4/id5 1..100 int32, id6 1..100000 int32 (official CSV stores these as ints).

Output (NOT in data repo): C:/App/competitions/H2O/session/skew_family/
  skew_{level}_10M.parquet  (+ {level}_stats_chk.json with distribution stats
  per grouping key Q1/Q2/Q3/Q4/Q5: N, cardinality, CV, Gini, p50/p95/p99,
  max/median, top1%/top10% share + reference chk Q1-Q5).

No prod change. No SPEC change. No push.
Usage (Git Bash):
  C:/App/competitions/H2O/python310/python.exe tests/research/skew_family/gen_skew.py
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

SEED_V = 42
LEVELS = [("uniform", None), ("mild-skew", 0.8), ("medium-skew", 1.2), ("heavy-skew", 1.7)]
N = 10_000_000
K1 = K2 = K4 = K5 = 100
K3 = K6 = 100_000
OUTDIR = Path("C:/App/competitions/H2O/session/skew_family")


def zipf_probs(k, s):
    ranks = np.arange(1, k + 1, dtype=np.float64)
    w = 1.0 / np.power(ranks, s)
    return (w / w.sum()).astype(np.float64)


def sample_codes(n, k, s, rng):
    if s is None:
        return rng.integers(0, k, size=n, dtype=np.int64)
    p = zipf_probs(k, s)
    return rng.choice(k, size=n, p=p).astype(np.int64)


def force_fill(codes, k):
    """Ensure every group 0..k-1 appears >=1 (overwrite head deterministically)."""
    present = np.zeros(k, dtype=bool)
    present[np.unique(codes)] = True
    missing = np.flatnonzero(~present)
    for j, g in enumerate(missing.tolist()):
        codes[j] = int(g)
    return len(missing)


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def dist_stats(sizes):
    s = np.sort(np.asarray(sizes, dtype=np.int64))
    n = s.size
    total = float(s.sum())
    mean = float(s.mean())
    std = float(s.std())
    p50, p95, p99 = (float(np.percentile(s, q)) for q in (50, 95, 99))
    median = float(np.median(s))
    mx = int(s.max())
    top1_n = max(1, n // 100)
    top10_n = max(1, n // 10)
    top1 = float(s[-top1_n:].sum() / total * 100)
    top10 = float(s[-top10_n:].sum() / total * 100)
    return {
        "ngroups": int(n), "mean": mean, "std": std,
        "cv": float(std / mean) if mean else 0.0, "gini": gini(s),
        "p50": p50, "p95": p95, "p99": p99, "max": mx, "median": median,
        "max_over_median": float(mx / median) if median else 0.0,
        "top1pct_share": top1, "top10pct_share": top10,
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rv = np.random.default_rng(SEED_V)
    v1 = rv.integers(1, 6, size=N).astype(np.int32)
    v2 = rv.integers(1, 16, size=N).astype(np.int32)
    v3 = (rv.random(size=N) * 100.0).astype(np.float64)
    print(f"values shared seed=42: v1sum={int(v1.sum())} v2sum={int(v2.sum())} "
          f"v3sum={float(v3.sum()):.2f} ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)
    del rv
    gc.collect()
    for li, (level, s) in enumerate(LEVELS):
        t = time.perf_counter()
        rng = np.random.default_rng(42 + li)
        c1 = sample_codes(N, K1, s, rng)
        c2 = sample_codes(N, K2, s, rng)
        c3 = sample_codes(N, K3, s, rng)
        c4 = sample_codes(N, K4, s, rng)
        c5 = rng.integers(0, K5, size=N, dtype=np.int64)
        c6 = sample_codes(N, K6, s, rng)
        f1 = force_fill(c1, K1); f2 = force_fill(c2, K2); f3 = force_fill(c3, K3)
        f4 = force_fill(c4, K4); f5 = force_fill(c5, K5); f6 = force_fill(c6, K6)
        print(f"{level}: sampled {(time.perf_counter()-t)*1000:.0f}ms "
              f"forced={f1},{f2},{f3},{f4},{f5},{f6}", flush=True)
        # group-size distributions per query key
        g_q1 = np.bincount(c1, minlength=K1)
        g_q4 = np.bincount(c4, minlength=K4)
        p2 = c1.astype(np.int64) * 100 + c2.astype(np.int64)
        g_q2 = np.bincount(p2, minlength=K1 * K2)
        g_q2nz = g_q2[g_q2 > 0]
        g_q3 = np.bincount(c3, minlength=K3)
        g_q5 = np.bincount(c6, minlength=K6)
        stats = {
            "N": N, "seed": 42, "skew_s": s,
            "cardinality": {"id1": int(np.unique(c1).size), "id2": int(np.unique(c2).size),
                            "id3": int(np.unique(c3).size), "id4": int(np.unique(c4).size),
                            "id5": int(np.unique(c5).size), "id6": int(np.unique(c6).size)},
            "forced": {"id1": f1, "id2": f2, "id3": f3, "id4": f4, "id5": f5, "id6": f6},
            "Q1_id1": dist_stats(g_q1), "Q2_id1id2": dist_stats(g_q2nz),
            "Q3_id3": dist_stats(g_q3), "Q4_id4": dist_stats(g_q4),
            "Q5_id6": dist_stats(g_q5),
            "Q2_joint_ngroups": int(g_q2nz.size),
        }
        # reference chk (own reference per synthetic dataset; official chk N/A)
        q1_tot = int(v1.sum())
        q2_tot = int(v1.sum())
        q3_v1 = int(v1.sum())
        # mean3 sum via bincount exact
        s3 = np.bincount(c3, weights=v3.astype(np.float64), minlength=K3)
        cnt3 = np.bincount(c3, minlength=K3).astype(np.float64)
        mean3_sum = float((s3 / np.maximum(cnt3, 1)).sum())
        s1m = np.bincount(c4, weights=v1.astype(np.float64), minlength=K4)
        s2m = np.bincount(c4, weights=v2.astype(np.float64), minlength=K4)
        s3m = np.bincount(c4, weights=v3, minlength=K4)
        cnt4 = np.bincount(c4, minlength=K4).astype(np.float64)
        q4 = {"m1": float((s1m / np.maximum(cnt4, 1)).sum()),
              "m2": float((s2m / np.maximum(cnt4, 1)).sum()),
              "m3": float((s3m / np.maximum(cnt4, 1)).sum())}
        q5 = {"s1": int(v1.sum()), "s2": int(v2.sum()), "s3": float(v3.sum()),
              "s3_scaled": int(round(float(v3.sum()) * 1e6))}
        stats["chk"] = {
            "Q1": {"ngroups": int(np.unique(c1).size), "total": q1_tot},
            "Q2": {"ngroups": int(g_q2nz.size), "total": q2_tot},
            "Q3": {"ngroups": int(np.unique(c3).size), "v1": q3_v1, "mean3_sum": mean3_sum},
            "Q4": {"ngroups": int(np.unique(c4).size), **q4},
            "Q5": {"ngroups": int(np.unique(c6).size), **q5},
        }
        # write parquet
        import pyarrow as pa
        import pyarrow.parquet as pq
        id1 = np.array(["id%03d" % (int(x) + 1) for x in c1.tolist()], dtype=object)
        del c1
        gc.collect()
        id2 = np.array(["id%03d" % (int(x) + 1) for x in c2.tolist()], dtype=object)
        del c2
        gc.collect()
        id3 = np.array(["id%010d" % (int(x) + 1) for x in c3.tolist()], dtype=object)
        del c3
        gc.collect()
        t_par = time.perf_counter()
        tbl = pa.table({
            "id1": pa.array(id1, type=pa.string()),
            "id2": pa.array(id2, type=pa.string()),
            "id3": pa.array(id3, type=pa.string()),
            "id4": pa.array((c4 + 1).astype(np.int32)),
            "id5": pa.array((c5 + 1).astype(np.int32)),
            "id6": pa.array((c6 + 1).astype(np.int32)),
            "v1": pa.array(v1), "v2": pa.array(v2), "v3": pa.array(v3),
        })
        del id1, id2, id3, c5
        gc.collect()
        out = OUTDIR / f"skew_{level}_10M.parquet"
        pq.write_table(tbl, out, compression="zstd")
        del tbl
        gc.collect()
        print(f"{level}: parquet {out.name} ({(time.perf_counter()-t_par)*1000:.0f}ms) "
              f"total {(time.perf_counter()-t)*1000:.0f}ms", flush=True)
        with open(OUTDIR / f"skew_{level}_10M_stats_chk.json", "w") as f:
            json.dump(stats, f, indent=1)
        # keep c4/c6/v alive via reload? c4,c6 needed for next iter? no, regen each iter
        del c4, c6
        gc.collect()
    print(f"DONE total {(time.perf_counter()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
