# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Competitors 1T: DuckDB (threads=1) vs Polars (single thread) on H2O.

Q1-Q5 equivalents, best-of-3 query-only (load reported separately).
Datasets: G1_1e7_1e2_0_0 (10M) + G1_1e8_1e2_0_0 (100M). No 1B.
Goldens: group counts + Q1 total checked per engine.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

FORK = Path(__file__).resolve().parents[3]
OUT = FORK / "tests" / "research" / "singlepass" / "competitors_1T.json"

DATA = {
    "10M": "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv",
    "100M": "C:/App/competitions/H2O/data/G1_1e8_1e2_0_0.csv",
}
GROUPS = {"Q1": 100, "Q2": 10000}
GOLD_TOT = {"10M": 29998789, "100M": 299991302}


def best(fn, reps=3):
    ts = []
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return out, min(ts), ts


def duck_queries(con):
    def q1():
        return con.execute(
            "SELECT id1, SUM(v1) s FROM t GROUP BY id1").fetchall()

    def q2():
        return con.execute(
            "SELECT id1, id2, SUM(v1) s FROM t GROUP BY id1, id2").fetchall()

    def q3():
        return con.execute(
            "SELECT id3, SUM(v1) s, AVG(v3) m FROM t GROUP BY id3").fetchall()

    def q4():
        return con.execute(
            "SELECT id4, AVG(v1) a, AVG(v2) b, AVG(v3) c FROM t "
            "GROUP BY id4").fetchall()

    def q5():
        return con.execute(
            "SELECT id6, SUM(v1) a, SUM(v2) b, SUM(v3) c FROM t "
            "GROUP BY id6").fetchall()

    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5}


def pol_queries(df):
    import polars as pl

    def q1():
        return df.group_by("id1").agg(pl.col("v1").sum())

    def q2():
        return df.group_by(["id1", "id2"]).agg(pl.col("v1").sum())

    def q3():
        return df.group_by("id3").agg(
            pl.col("v1").sum(), pl.col("v3").mean())

    def q4():
        return df.group_by("id4").agg(
            pl.col("v1").mean(), pl.col("v2").mean(), pl.col("v3").mean())

    def q5():
        return df.group_by("id6").agg(
            pl.col("v1").sum(), pl.col("v2").sum(), pl.col("v3").sum())

    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5}


def main():
    import duckdb
    import polars as pl
    print(f"polars {pl.__version__} threads={pl.thread_pool_size()} "
          f"duckdb {duckdb.__version__}", flush=True)
    out = {}
    for tag, csv in DATA.items():
        print(f"=== {tag} {csv} ===", flush=True)
        res = {}
        # ---- DuckDB ----
        t = time.perf_counter()
        con = duckdb.connect()
        con.execute("SET threads TO 1")
        con.execute(
            "CREATE TABLE t AS SELECT id1,id2,id3,id4,id6,v1,v2,v3 "
            f"FROM read_csv('{csv}', header=true)")
        load_ms = (time.perf_counter() - t) * 1000
        nth = con.execute("SELECT current_setting('threads')").fetchall()
        print(f"duck load {load_ms:.0f}ms threads={nth}", flush=True)
        dq = duck_queries(con)
        d = {"load_ms": load_ms}
        for q, fn in dq.items():
            rows, b, ts = best(fn)
            if q == "Q1":
                assert len(rows) == 100, len(rows)
                tot = int(sum(r[1] for r in rows))
                assert tot == GOLD_TOT[tag], tot
            if q == "Q2":
                assert len(rows) == GROUPS["Q2"], len(rows)
            print(f"duck.{q}: {[f'{x:.0f}' for x in ts]} best={b:.0f}ms "
                  f"ng={len(rows)}", flush=True)
            d[q] = {"best_ms": b, "reps_ms": ts, "ngroups": len(rows)}
        res["duckdb_1T"] = d
        con.close()
        gc.collect()
        # ---- Polars ----
        t = time.perf_counter()
        df = pl.read_csv(csv, columns=["id1", "id2", "id3", "id4", "id6",
                                       "v1", "v2", "v3"])
        load_ms = (time.perf_counter() - t) * 1000
        print(f"polars load {load_ms:.0f}ms shape={df.shape}", flush=True)
        pq = pol_queries(df)
        p = {"load_ms": load_ms}
        for q, fn in pq.items():
            fr, b, ts = best(fn)
            if q == "Q1":
                assert fr.shape[0] == 100, fr.shape
                tot = int(fr["v1"].sum())
                assert tot == GOLD_TOT[tag], tot
            if q == "Q2":
                assert fr.shape[0] == GROUPS["Q2"], fr.shape
            print(f"polars.{q}: {[f'{x:.0f}' for x in ts]} best={b:.0f}ms "
                  f"ng={fr.shape[0]}", flush=True)
            p[q] = {"best_ms": b, "reps_ms": ts, "ngroups": fr.shape[0]}
        res["polars_1T"] = p
        del df
        gc.collect()
        out[tag] = res
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{OUT.name} written", flush=True)


if __name__ == "__main__":
    main()
