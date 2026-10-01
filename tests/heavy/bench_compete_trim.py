# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trimmed same-T competitors 10M (NEW file; existing benches untouched).

One process per T (argv[1]); POLARS_MAX_THREADS=T set BEFORE polars import,
duckdb PRAGMA threads=T. CSV load once per process (outside per-engine
timing); queries best-of-3, gold-checked (total 29998789). Appends to
tests/heavy/bench_compete_trim.jsonl. Run T=1,8,16 SEQUENTIALLY
(never parallel: memory-bus sensitive).

Usage (Git Bash, sequential):
  timeout 550 .../python tests/heavy/bench_compete_trim.py 1
  timeout 550 .../python tests/heavy/bench_compete_trim.py 8
  timeout 550 .../python tests/heavy/bench_compete_trim.py 16
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
os.environ["POLARS_MAX_THREADS"] = str(T)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

FORK = Path(__file__).resolve().parents[2]
CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000
OUT = FORK / "tests" / "heavy" / "bench_compete_trim.jsonl"


def best_of(fn, reps=3):
    best, out = 1e18, None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t) * 1000)
    return out, best


def main():
    import pandas as pd

    print(f"=== compete-trim 10M T={T} pid={os.getpid()} ===", flush=True)
    t = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6",
                                   "v1", "v2", "v3"])
    assert len(df) == N
    print(f"csv {((time.perf_counter() - t) * 1000):.0f}ms", flush=True)

    import duckdb

    con = duckdb.connect()
    con.execute("PRAGMA threads=%d" % T)
    con.register("t", df)
    DQ = {
        "Q1": "SELECT id1, SUM(v1) s FROM t GROUP BY id1",
        "Q2": "SELECT id1, id2, SUM(v1) s FROM t GROUP BY id1, id2",
        "Q3": "SELECT id3, SUM(v1) s, AVG(v3) m FROM t GROUP BY id3",
        "Q4": "SELECT id4, AVG(v1) a, AVG(v2) b, AVG(v3) c FROM t GROUP BY id4",
        "Q5": "SELECT id6, SUM(v1) a, SUM(v2) b, SUM(v3) c FROM t GROUP BY id6",
    }
    duck_ms = {}
    for q, sql in DQ.items():
        (r, ms) = best_of(lambda sql=sql: con.execute(sql).fetchall(), 3)
        duck_ms[q] = round(ms, 1)
    assert sum(r[1] for r in con.execute(DQ["Q1"]).fetchall()) == 29998789
    print(f"T={T} duckdb={duck_ms}", flush=True)
    del con
    gc.collect()

    import polars as pl

    pf = pl.from_pandas(df)
    del df
    gc.collect()
    PQ = {
        "Q1": lambda: pf.group_by("id1").agg(pl.col("v1").sum()),
        "Q2": lambda: pf.group_by(["id1", "id2"]).agg(pl.col("v1").sum()),
        "Q3": lambda: pf.group_by("id3").agg(pl.col("v1").sum(),
                                             pl.col("v3").mean()),
        "Q4": lambda: pf.group_by("id4").agg(pl.col("v1").mean(),
                                             pl.col("v2").mean(),
                                             pl.col("v3").mean()),
        "Q5": lambda: pf.group_by("id6").agg(pl.col("v1").sum(),
                                             pl.col("v2").sum(),
                                             pl.col("v3").sum()),
    }
    pol_ms = {}
    for q, fn in PQ.items():
        (r, ms) = best_of(lambda fn=fn: fn().to_series(-1).sum(), 3)
        pol_ms[q] = round(ms, 1)
    assert int(pf.group_by("id1").agg(
        pl.col("v1").sum()).to_series(-1).sum()) == 29998789
    print(f"T={T} polars={pol_ms}", flush=True)

    with open(str(OUT), "a") as f:
        f.write(json.dumps({"T": T, "n": N, "duckdb_ms": duck_ms,
                            "polars_ms": pol_ms, "gold": "EXACT/PASS"}) + "\n")
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
