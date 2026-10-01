# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research: Polars same-thread ladder. ONE thread count per process
(POLARS_MAX_THREADS is pool-fixed at first use) -> run once per T:
  python tests/research/mtgroup/bench_polars_mt.py <T>
Q1-Q5 equivalents, best-of-3 query-only. Research-only.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
os.environ["POLARS_MAX_THREADS"] = str(T)
os.environ["OMP_NUM_THREADS"] = "1"

FORK = Path(__file__).resolve().parents[3]
OUT = FORK / "tests" / "research" / "mtgroup" / f"results_polars_mt_{T}.json"

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"


def best(fn, reps=3):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return out, min(ts), ts


def main():
    import polars as pl
    print(f"loading csv (threads={T})...", flush=True)
    t = time.perf_counter()
    df = pl.read_csv(CSV, columns=["id1", "id2", "id3", "id4", "id6",
                                   "v1", "v2", "v3"])
    print(f"load={(time.perf_counter() - t) * 1000:.0f}ms n={len(df)}",
          flush=True)
    assert len(df) == 10_000_000
    qs = {
        "Q1": lambda: df.group_by("id1").agg(pl.col("v1").sum()),
        "Q2": lambda: df.group_by(["id1", "id2"]).agg(pl.col("v1").sum()),
        "Q3": lambda: df.group_by("id3").agg(pl.col("v1").sum(),
                                             pl.col("v3").mean()),
        "Q4": lambda: df.group_by("id4").agg(pl.col("v1").mean(),
                                             pl.col("v2").mean(),
                                             pl.col("v3").mean()),
        "Q5": lambda: df.group_by("id6").agg(pl.col("v1").sum(),
                                             pl.col("v2").sum(),
                                             pl.col("v3").sum()),
    }
    out = {"threads": T, "N": len(df), "cells": {}}
    for q, fn in qs.items():
        res, b, ts = best(fn)
        if q == "Q1":
            assert res.height == 100, res.height
            assert int(res["v1"].sum()) == 29998789
        if q == "Q2":
            assert res.height == 10000, res.height
        out["cells"][q] = {"best_ms": b, "reps_ms": ts,
                           "height": res.height}
        print(f"{q}@{T}: best={b:.0f} {[f'{x:.0f}' for x in ts]}",
              flush=True)
        gc.collect()
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{OUT.name} written", flush=True)


if __name__ == "__main__":
    main()
