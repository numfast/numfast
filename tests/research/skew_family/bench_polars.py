# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research-only: Polars thread ladder. ONE thread count per process
(POLARS_MAX_THREADS pool-fixed at first use) -> run once per T:
  C:/App/competitions/H2O/python310/python.exe tests/research/skew_family/bench_polars.py <T> [level|all]
Q1-Q5 query-only run1/run2, chk separate vs own stats_chk.json reference.
Thread proof: pl.thread_pool_size() logged per cell (must equal T).
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

SESS = Path("C:/App/competitions/H2O/session/skew_family")
LEVELS_ALL = ["uniform", "mild-skew", "medium-skew", "heavy-skew"]


def main():
    import polars as pl
    arg = sys.argv[2] if len(sys.argv) > 2 else "all"
    levels = LEVELS_ALL if arg == "all" else [arg]
    pool = pl.thread_pool_size()
    print(f"polars {pl.__version__} POLARS_MAX_THREADS={T} pool={pool}", flush=True)
    assert pool == T, (pool, T)
    for level in levels:
        ref = json.load(open(SESS / f"skew_{level}_10M_stats_chk.json"))
        t = time.perf_counter()
        df = pl.read_parquet(SESS / f"skew_{level}_10M.parquet")
        load_ms = (time.perf_counter() - t) * 1000
        assert df.height == ref["N"], df.height
        print(f"=== {level} load={load_ms:.0f}ms n={df.height} ===", flush=True)
        qs = {
            "Q1": lambda: df.group_by("id1").agg(pl.col("v1").sum()),
            "Q2": lambda: df.group_by(["id1", "id2"]).agg(pl.col("v1").sum()),
            "Q3": lambda: df.group_by("id3").agg(pl.col("v1").sum(), pl.col("v3").mean()),
            "Q4": lambda: df.group_by("id4").agg(pl.col("v1").mean(), pl.col("v2").mean(),
                                                 pl.col("v3").mean()),
            "Q5": lambda: df.group_by("id6").agg(pl.col("v1").sum(), pl.col("v2").sum(),
                                                 pl.col("v3").sum()),
        }
        out = {"level": level, "engine": f"polars {pl.__version__}", "threads": T,
               "pool_proof": pool, "N": df.height, "load_ms": load_ms, "cells": {}}
        for q, fn in qs.items():
            reps = []
            res = None
            for _ in range(2):
                s = time.perf_counter()
                res = fn()
                reps.append((time.perf_counter() - s) * 1000)
            assert res.height == ref["chk"][q]["ngroups"], (level, q, res.height)
            if q in ("Q1", "Q2"):
                assert int(res["v1"].sum()) == ref["chk"][q]["total"], (level, q)
            out["cells"][q] = {"run1_ms": reps[0], "run2_ms": reps[1],
                               "best_ms": min(reps), "pool_proof": pool}
            print(f"{q}@pool={pool}: run1={reps[0]:.0f} run2={reps[1]:.0f} ng={res.height}",
                  flush=True)
            gc.collect()
        with open(SESS / f"results_polars_{level}_{T}T.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"results_polars_{level}_{T}T.json written", flush=True)
        del df
        gc.collect()


if __name__ == "__main__":
    main()
