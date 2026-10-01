# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research: DuckDB same-thread ladder (threads=1,2,4,8,16) on H2O 10M.

Single process, CSV loaded once, PRAGMA threads set per (T,Q) cell.
Q1-Q5 equivalents, best-of-3 query-only. Research-only.
Usage: python tests/research/mtgroup/bench_duck_mt.py
"""
import gc
import json
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[3]
OUT = FORK / "tests" / "research" / "mtgroup" / "results_duck_mt.json"

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
LADDER = [1, 2, 4, 8, 16]
GOLD = {"Q1": (100, 29998789), "Q2": (10000, 29998789)}


def best(fn, reps=3):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return out, min(ts), ts


def main():
    import duckdb
    print("loading csv...", flush=True)
    t = time.perf_counter()
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS SELECT * FROM read_csv('{CSV}')")
    print(f"load={(time.perf_counter() - t) * 1000:.0f}ms", flush=True)
    n = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert n == 10_000_000, n
    tot_v1 = con.execute("SELECT SUM(v1)::BIGINT FROM t").fetchone()[0]
    print(f"types: {con.execute('DESCRIBE t').fetchall()}", flush=True)
    print(f"total v1={tot_v1}", flush=True)
    assert tot_v1 == 29998789, tot_v1
    qs = {
        "Q1": "SELECT id1, SUM(v1) s FROM t GROUP BY id1",
        "Q2": "SELECT id1, id2, SUM(v1) s FROM t GROUP BY id1, id2",
        "Q3": "SELECT id3, SUM(v1) s, AVG(v3) m FROM t GROUP BY id3",
        "Q4": "SELECT id4, AVG(v1) a, AVG(v2) b, AVG(v3) c FROM t GROUP BY id4",
        "Q5": "SELECT id6, SUM(v1) a, SUM(v2) b, SUM(v3) c FROM t GROUP BY id6",
    }
    out = {"N": n, "cells": {}}
    for T in LADDER:
        con.execute(f"PRAGMA threads={T}")
        for q, sql in qs.items():
            res, b, ts = best(lambda: con.execute(sql).fetchall())
            if q in GOLD:
                ng, tot = GOLD[q]
                assert len(res) == ng, (q, T, len(res))
                ints = [v for row in res for v in row if isinstance(v, int)]
                assert sum(ints) == tot, (q, T, sum(ints))
            out["cells"][f"{q}@{T}"] = {"best_ms": b, "reps_ms": ts}
            print(f"{q}@{T}: best={b:.0f} {[f'{x:.0f}' for x in ts]}",
                  flush=True)
            gc.collect()
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{OUT.name} written", flush=True)


if __name__ == "__main__":
    main()
