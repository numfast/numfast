# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-N (10M) same-T comparison: native-ON vs DuckDB vs Polars, Q1-Q5.

One process per T (argv[1]): POLARS_MAX_THREADS=T + duckdb PRAGMA threads=T.
Native path is single-threaded (Rust primitives are ST by contract; MT only
by host) — reported alongside at every T for reference.
CSV load + prep outside per-engine timing; queries best-of-3, gold-checked.
Appends {"T":..,"native_ms":{},"duckdb_ms":{},"polars_ms":{}} to JSONL OUT.

Usage: POLARS_MAX_THREADS=<T> python compare_sameT_10M.py <T>
"""

import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
FORK = Path(__file__).resolve().parents[3]
CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000
OUT = FORK / "tests" / "research" / "native_cpu" / "compare_sameT_10M.jsonl"

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def best_of(fn, reps=3):
    best, out = 1e18, None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t) * 1000)
    return out, best


def main():
    import pandas as pd

    _spec = importlib.util.spec_from_file_location(
        "native_cpu", str(FORK / "src" / "Drivers" / "CPU" / "_lib" / "native_cpu.py"))
    nc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(nc)
    assert nc.available(), nc.why()

    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    assert len(df) == N
    id1 = df["id1"].to_numpy()
    id2 = df["id2"].to_numpy()
    id3 = df["id3"].to_numpy()
    ID4 = df["id4"].to_numpy().astype(np.int32)
    ID6 = df["id6"].to_numpy().astype(np.int32)
    V1F = df["v1"].to_numpy().astype(np.float64)
    V2F = df["v2"].to_numpy().astype(np.float64)
    V3 = df["v3"].to_numpy().astype(np.float64)

    # native prep (outside timing)
    D1, O1 = nc.build_pattern_buffers(id1)
    D2, O2 = nc.build_pattern_buffers(id2)
    D3, O3 = nc.build_pattern_buffers(id3)
    C1 = nc.pattern_encode_buffers(D1, O1, "id")
    C2 = nc.pattern_encode_buffers(D2, O2, "id")
    C3 = nc.pattern_encode_buffers(D3, O3, "id")
    K4 = (ID4 - int(ID4.min())).astype(np.int32)
    K6 = (ID6 - int(ID6.min())).astype(np.int32)
    del id1, id2, id3
    gc.collect()

    M2 = int(C2[0].max()) + 1
    G1, G2, G3 = int(C1[0].max()) + 1, None, int(C3[0].max()) + 1
    COMP = nc.pack_i32_direct(C1[0], C2[0], M2)
    G2 = int(COMP.max()) + 1
    G4, G6 = int(K4.max()) + 1, int(K6.max()) + 1

    def n_q1():
        return nc.fused_sum_count(C1[0], V1F, G1)

    def n_q2():
        return nc.fused_sum_count(COMP, V1F, G2)

    def n_q3():
        return nc.multi_sum_count(C3[0], [V1F, V3], G3)

    def n_q4():
        return nc.multi_sum_count(K4, [V1F, V2F, V3], G4)

    def n_q5():
        return nc.multi_sum_count(K6, [V1F, V2F, V3], G6)

    native_ms = {}
    for q, fn in (("Q1", n_q1), ("Q2", n_q2), ("Q3", n_q3), ("Q4", n_q4), ("Q5", n_q5)):
        _, ms = best_of(fn, 3)
        native_ms[q] = ms
    # native gold spot-check
    (s, _), _ = best_of(n_q1, 1)
    assert int(s.sum()) == 29998789

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
        duck_ms[q] = ms
    assert sum(r[1] for r in con.execute(DQ["Q1"]).fetchall()) == 29998789

    import polars as pl

    pl.Config.set_verbose(False)
    pf = pl.from_pandas(df)
    PQ = {
        "Q1": lambda: pf.group_by("id1").agg(pl.col("v1").sum()),
        "Q2": lambda: pf.group_by(["id1", "id2"]).agg(pl.col("v1").sum()),
        "Q3": lambda: pf.group_by("id3").agg(pl.col("v1").sum(), pl.col("v3").mean()),
        "Q4": lambda: pf.group_by("id4").agg(pl.col("v1").mean(), pl.col("v2").mean(), pl.col("v3").mean()),
        "Q5": lambda: pf.group_by("id6").agg(pl.col("v1").sum(), pl.col("v2").sum(), pl.col("v3").sum()),
    }
    pol_ms = {}
    for q, fn in PQ.items():
        (r, ms) = best_of(lambda fn=fn: fn().to_series(-1).sum(), 3)
        pol_ms[q] = ms
    assert int(pf.group_by("id1").agg(pl.col("v1").sum()).to_series(-1).sum()) == 29998789

    row = {"T": T, "n": N, "native_st_ms": native_ms, "duckdb_ms": duck_ms, "polars_ms": pol_ms}
    with open(str(OUT), "a") as f:
        f.write(json.dumps(row) + "\n")
    print("T=%d native=%s" % (T, {k: round(v, 1) for k, v in native_ms.items()}), flush=True)
    print("T=%d duckdb=%s" % (T, {k: round(v, 1) for k, v in duck_ms.items()}), flush=True)
    print("T=%d polars=%s" % (T, {k: round(v, 1) for k, v in pol_ms.items()}), flush=True)


if __name__ == "__main__":
    main()
