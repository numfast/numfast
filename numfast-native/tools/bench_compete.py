# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Competitors query-only: DuckDB + Polars vs native at 10M (seed 42).

Usage: python tools/bench_compete.py [--n N]
Preparation (table build/register) timed separately from query. Group-by
sum + count over int32 keys, g=256. New file; no existing bench touched.
"""
import ctypes
import os
import sys
import time

import numpy as np

N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10_000_000
G = 256
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")

rng = np.random.default_rng(42)
keys = rng.integers(0, G, size=N).astype(np.int32)
vals = rng.standard_normal(N)


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def bench(fn, reps=5):
    for _ in range(2):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return med(ts)


# native
d = ctypes.CDLL(DLL)
fd = d.nf_group_sum_count
fd.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
fd.restype = ctypes.c_int32
s = np.zeros(G)
c = np.zeros(G, dtype=np.int64)
t_nat = bench(lambda: fd(keys.ctypes.data, vals.ctypes.data, N, s.ctypes.data, c.ctypes.data, G))
ref_s = s.copy()

# polars
import polars as pl
t = time.perf_counter()
df = pl.DataFrame({"k": keys, "v": vals})
t_prep_pl = (time.perf_counter() - t) * 1e3


def pl_query():
    return df.group_by("k").agg(pl.col("v").sum().alias("s"), pl.len().alias("n")).sort("k")


t_pl = bench(pl_query)
out = pl_query()
pls = out["s"].to_numpy()

# duckdb (arrow table in, zero bulk-insert cost)
import duckdb
import pyarrow as pa
t = time.perf_counter()
tab = pa.table({"k": keys, "v": vals})
con = duckdb.connect()
con.register("t", tab)
t_prep_duck = (time.perf_counter() - t) * 1e3


def duck_query():
    return con.execute("SELECT k, SUM(v) s, COUNT(*) n FROM t GROUP BY k ORDER BY k").fetchall()


t_duck = bench(duck_query)
rows = duck_query()
ducks = np.array([r[1] for r in rows])

print("n=%d g=%d seed=42" % (N, G))
print("native (1T fused) : med=%.3fms" % t_nat)
print("polars prep=%.1fms query med=%.3fms (%.2fx vs native)" % (t_prep_pl, t_pl, t_pl / t_nat))
print("duckdb prep=%.1fms query med=%.3fms (%.2fx vs native)" % (t_prep_duck, t_duck, t_duck / t_nat))
print("polars max|dsum|=%.3g duckdb max|dsum|=%.3g (fp-order tolerance)" % (
    float(np.max(np.abs(pls - ref_s[:len(pls)]))), float(np.max(np.abs(ducks - ref_s[:len(ducks)])))))
