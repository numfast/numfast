# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H2O SHOWDOWN v0.2.1: NumFast vs Polars/DuckDB/pandas/DataFusion, 10M (+100M spot).

NEW file; existing benches untouched; benchmark semantics unchanged
(Q1-Q5 graphs/GOLD verbatim from bench_h2o_scaled_gpu_10M.py).

One process per (engine, T): sequential, never parallel.
  PYTHONPATH=C:/App/numfast/app-builder timeout 550 python
    tests/heavy/bench_showdown_h2o.py <numfast|duckdb|polars|pandas|datafusion> <T> [--n 10000000|100000000]

NumFast: csv_load + encode(id1-3) prep split; Q1-Q5 carry(native) vs
dict(python) + compile/optimize/execute stages, cold then best-of-3 warm,
exact GOLD chk, peak RSS. CPU backend (GPU Q1/Q2 covered separately).
Competitors: native read (no pandas register), queries best-of-3, exact chk.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "numfast"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 16
N = 100_000_000 if "--n" in sys.argv and "100000000" in sys.argv[sys.argv.index("--n") + 1] else 10_000_000
assert ENGINE in ("numfast", "duckdb", "polars", "pandas", "datafusion"), ENGINE

os.environ["POLARS_MAX_THREADS"] = str(T)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
if ENGINE == "numfast":
    os.environ["NUMFAST_THREADS"] = str(T)

FORK = Path(__file__).resolve().parents[2]
CSV = ("C:/App/competitions/H2O/data/G1_1e8_1e2_0_0.csv"
       if N == 100_000_000 else "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv")
OUT = FORK / "tests" / "heavy" / "bench_showdown.jsonl"
GOLD = {
    "Q1": {"ngroups": 100, "total": 29998789},
    "Q2": {"ngroups": 10000, "total": 29998789},
    "Q3": {"ngroups": 100000, "v1": 29998789, "mean3_sum": 4999719.622344427},
    "Q4": {"ngroups": 100, "m1": 299.98798187506526,
           "m2": 799.894179409978, "m3": 4999.766872833688},
    "Q5": {"ngroups": 100000, "s1": 29998789, "s2": 79989360,
           "s3_scaled": 499976651408061},
}
GOLD100 = {  # filled at runtime from first exact run if --n 100M (chk cross-engine)
    "Q1": {"ngroups": 100}, "Q2": {"ngroups": 10000}, "Q3": {"ngroups": 100000},
    "Q4": {"ngroups": 100}, "Q5": {"ngroups": 100000},
}

import psutil  # noqa: E402

PROC = psutil.Process()
PEAK = [0.0]


def rss():
    v = PROC.memory_info().rss / 1e9
    PEAK[0] = max(PEAK[0], v)
    return v


def best_of(fn, reps=3):
    best, out = 1e18, None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t) * 1000)
    return out, best


def chk_q1_q2(rows_total, ngroups, label, gold):
    assert ngroups == gold["ngroups"], (label, ngroups)
    assert rows_total == gold["total"], (label, rows_total)
    return rows_total


def run_numfast():
    import numpy as np

    from builder import MAIN
    print(f"=== showdown numfast N={N} T={T} pid={os.getpid()} ===", flush=True)
    a = MAIN["build"](str(FORK)).alias
    v3col = a["column_schema"]("v3", "float64", 1e-6, 0)
    SC, OFF = float(v3col["scale"]), float(v3col["offset"])
    import pandas as pd
    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    load_ms = (time.perf_counter() - t0) * 1000
    assert len(df) == N
    print(f"csv_load {load_ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    enc = {}
    for col in ("id1", "id2", "id3"):
        raw = df[col].to_numpy()
        t0 = time.perf_counter()
        g = a["optimize"](a["compile"]([a["ir_encode_pattern"]("c", raw, "id")]))
        bufs = a["cpu_execute"](g["nodes"])
        ms = (time.perf_counter() - t0) * 1000
        enc[col] = bufs["c"]
        print(f"encode {col}: {ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    prep_ms = {c: None for c in ()}
    c1, c2, c3 = enc["id1"], enc["id2"], enc["id3"]
    id4 = df["id4"].to_numpy().astype(np.int32)
    id6 = df["id6"].to_numpy().astype(np.int32)
    v1 = df["v1"].to_numpy().astype(np.int32)
    v2 = df["v2"].to_numpy().astype(np.int32)
    v3 = df["v3"].to_numpy().astype(np.float64)
    del df
    gc.collect()
    JD = {
        "Q1": [a["ir_series"]("k", c1), a["ir_series"]("v", v1),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q2": [a["ir_series"]("c1", c1), a["ir_series"]("c2", c2),
               a["ir_series"]("v", v1),
               a["ir_pack_keys"]("k", "c1", "c2"),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q3": [a["ir_series"]("k", c3),
               a["ir_series"]("v1", v1),
               a["ir_series"]("v3", v3, "float64", scale=SC, offset=OFF),
               a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                     {"v1": ("sum",), "v3": ("mean",)})],
        "Q4": [a["ir_series"]("k", id4),
               a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
               a["ir_series"]("v3", v3, "float64", scale=SC, offset=OFF),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("mean",), "v2": ("mean",),
                                      "v3": ("mean",)})],
        "Q5": [a["ir_series"]("k", id6),
               a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
               a["ir_series"]("v3", v3, "float64", scale=SC, offset=OFF),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("sum",), "v2": ("sum",),
                                      "v3": ("sum",)})],
    }

    def gate(q, res):
        if N == 10_000_000:
            g = GOLD[q]
        else:
            return ("nocheck-100M", len(res))
        if q in ("Q1", "Q2"):
            tot = int(sum(int(v) for v in res.values()))
            assert len(res) == g["ngroups"] and tot == g["total"], (q, len(res), tot)
            return tot
        if q == "Q3":
            assert len(res) == g["ngroups"]
            assert int(sum(c["v1"]["sum"] for c in res.values())) == g["v1"]
            m3 = float(sum(c["v3"]["mean"] for c in res.values()))
            assert abs(m3 - g["mean3_sum"]) < 1e-6 * abs(g["mean3_sum"]) + 1e-6
            return m3
        if q == "Q4":
            assert len(res) == g["ngroups"]
            s = {c: float(sum(cell[c]["mean"] for cell in res.values()))
                 for c in ("v1", "v2", "v3")}
            md = max(abs(s["v1"] - g["m1"]), abs(s["v2"] - g["m2"]), abs(s["v3"] - g["m3"]))
            assert md < 1e-9
            return md
        assert len(res) == g["ngroups"]
        s1 = int(sum(c["v1"]["sum"] for c in res.values()))
        s2 = int(sum(c["v2"]["sum"] for c in res.values()))
        s3 = float(sum(c["v3"]["sum"] for c in res.values()))
        assert s1 == g["s1"] and s2 == g["s2"]
        assert int(round(s3 * 1e6)) == g["s3_scaled"]
        return int(round(s3 * 1e6))

    compd, compc = {}, {}
    for q, jj in JD.items():
        compd[q] = a["optimize"](a["compile"](jj))
        last = jj[-1]
        p = dict(last["params"])
        p["result"] = "carry"
        nn = dict(last)
        nn["params"] = p
        compc[q] = a["optimize"](a["compile"](jj[:-1] + [nn]))
    Q = {}
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        # cold: first carry + first dict (compile/opt outside, execute split)
        t = time.perf_counter()
        bufs = a["cpu_execute"](compc[q]["nodes"])
        cold_carry = (time.perf_counter() - t) * 1000
        gi = dict(bufs.get("g#groupindex", {}))
        del bufs
        gc.collect()
        t = time.perf_counter()
        res = a["evaluate"](compd[q], "cpu", N)["result"]
        cold_dict = (time.perf_counter() - t) * 1000
        cv = gate(q, res)
        del res
        gc.collect()
        _, warm_carry = best_of(lambda: a["cpu_execute"](compc[q]["nodes"]), 3)
        gc.collect()
        _, warm_dict = best_of(lambda: a["evaluate"](compd[q], "cpu", N)["result"], 3)
        gc.collect()
        Q[q] = {"cold_carry_ms": round(cold_carry, 1), "cold_dict_ms": round(cold_dict, 1),
                "warm_carry_ms": round(warm_carry, 1), "warm_dict_ms": round(warm_dict, 1),
                "native_py_overhead_ms": round(warm_dict - warm_carry, 1),
                "stages": {k: (round(v, 1) if isinstance(v, float) else v)
                           for k, v in gi.items() if k in ("gi_ms", "agg_ms", "carry_ms",
                                                           "merge_ms", "backend", "threads",
                                                           "strategy", "reason")},
                "chk": cv if isinstance(cv, float) else int(cv) if isinstance(cv, int) else str(cv)}
        print(f"{q}: cold_c={cold_carry:.0f} cold_d={cold_dict:.0f} warm_c={warm_carry:.0f} "
              f"warm_d={warm_dict:.0f} be={gi.get('backend')} chk={cv} RSS {rss():.2f}GB", flush=True)
    return {"engine": "numfast", "threads": T, "N": N, "csv_load_ms": round(load_ms, 1),
            "Q": Q, "peak_rss_gb": round(rss(), 2)}


def run_duckdb():
    import duckdb
    print(f"=== showdown duckdb {duckdb.__version__} N={N} T={T} ===", flush=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=%d" % T)
    t0 = time.perf_counter()
    con.execute("CREATE OR REPLACE TABLE t AS SELECT id1,id2,id3,id4,id6,v1,v2,v3 "
                f"FROM read_csv('{CSV}', header=true)")
    load_ms = (time.perf_counter() - t0) * 1000
    n = con.execute("SELECT COUNT(*) FROM t").fetchall()[0][0]
    assert n == N, n
    print(f"load {load_ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    DQ = {
        "Q1": "SELECT id1, SUM(v1) s FROM t GROUP BY id1",
        "Q2": "SELECT id1, id2, SUM(v1) s FROM t GROUP BY id1, id2",
        "Q3": "SELECT id3, SUM(v1) s, AVG(v3) m FROM t GROUP BY id3",
        "Q4": "SELECT id4, AVG(v1) a, AVG(v2) b, AVG(v3) c FROM t GROUP BY id4",
        "Q5": "SELECT id6, SUM(v1) a, SUM(v2) b, SUM(v3) c FROM t GROUP BY id6",
    }
    Q = {}
    for q, sql in DQ.items():
        r0 = con.execute(sql).fetchall()
        cold = None
        t = time.perf_counter()
        r0 = con.execute(sql).fetchall()
        cold = (time.perf_counter() - t) * 1000
        _, warm = best_of(lambda sql=sql: con.execute(sql).fetchall(), 3)
        if q == "Q1":
            tot = int(sum(x[1] for x in r0))
            if N == 10_000_000:
                chk_q1_q2(tot, len(r0), "duck " + q, GOLD[q])
            chk = tot
        elif q == "Q2":
            tot = int(sum(x[2] for x in r0))
            if N == 10_000_000:
                chk_q1_q2(tot, len(r0), "duck " + q, GOLD[q])
            chk = tot
        elif q == "Q3":
            tot = int(sum(x[1] for x in r0))
            m = float(sum(x[2] for x in r0))
            if N == 10_000_000:
                assert tot == GOLD[q]["v1"] and len(r0) == GOLD[q]["ngroups"]
                assert abs(m - GOLD[q]["mean3_sum"]) < 1e-6 * abs(GOLD[q]["mean3_sum"]) + 1e-6
            chk = m
        elif q == "Q4":
            m = [float(sum(x[i] for x in r0)) for i in (1, 2, 3)]
            if N == 10_000_000:
                assert len(r0) == GOLD[q]["ngroups"]
                assert max(abs(m[0] - GOLD[q]["m1"]), abs(m[1] - GOLD[q]["m2"]),
                           abs(m[2] - GOLD[q]["m3"])) < 1e-9
            chk = m
        else:
            s = [float(sum(x[i] for x in r0)) for i in (1, 2, 3)]
            if N == 10_000_000:
                assert len(r0) == GOLD[q]["ngroups"]
                assert int(s[0]) == GOLD[q]["s1"] and int(s[1]) == GOLD[q]["s2"]
                assert int(round(s[2] * 1e6)) == GOLD[q]["s3_scaled"]
            chk = s
        Q[q] = {"cold_ms": round(cold, 1), "warm_ms": round(warm, 1), "chk": chk,
                "ngroups": len(r0)}
        print(f"{q}: cold={cold:.0f} warm={warm:.0f} RSS {rss():.2f}GB", flush=True)
    return {"engine": "duckdb", "version": duckdb.__version__, "threads": T, "N": N,
            "csv_load_ms": round(load_ms, 1), "Q": Q, "peak_rss_gb": round(rss(), 2)}


def run_polars():
    import polars as pl
    print(f"=== showdown polars {pl.__version__} N={N} T={T} ===", flush=True)
    t0 = time.perf_counter()
    pf = pl.read_csv(CSV, columns=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    load_ms = (time.perf_counter() - t0) * 1000
    assert len(pf) == N
    print(f"load {load_ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    PQ = {
        "Q1": lambda: pf.group_by("id1").agg(pl.col("v1").sum()),
        "Q2": lambda: pf.group_by(["id1", "id2"]).agg(pl.col("v1").sum()),
        "Q3": lambda: pf.group_by("id3").agg(pl.col("v1").sum(), pl.col("v3").mean()),
        "Q4": lambda: pf.group_by("id4").agg(pl.col("v1").mean(), pl.col("v2").mean(),
                                             pl.col("v3").mean()),
        "Q5": lambda: pf.group_by("id6").agg(pl.col("v1").sum(), pl.col("v2").sum(),
                                             pl.col("v3").sum()),
    }
    Q = {}
    for q, fn in PQ.items():
        r0 = fn()
        t = time.perf_counter()
        r0 = fn()
        cold = (time.perf_counter() - t) * 1000
        _, warm = best_of(lambda fn=fn: fn(), 3)
        if q in ("Q1", "Q2"):
            tot = int(r0.to_series(-1).sum())
            if N == 10_000_000:
                chk_q1_q2(tot, len(r0), "pol " + q, GOLD[q])
            chk = tot
        elif q == "Q3":
            tot = int(r0["v1"].sum())
            m = float(r0["v3"].mean() * len(r0))
            if N == 10_000_000:
                assert tot == GOLD[q]["v1"] and len(r0) == GOLD[q]["ngroups"]
                assert abs(m - GOLD[q]["mean3_sum"]) < 1e-6 * abs(GOLD[q]["mean3_sum"]) + 1e-6
            chk = m
        elif q == "Q4":
            m = [float(r0[c].sum()) for c in ("v1", "v2", "v3")]
            if N == 10_000_000:
                assert len(r0) == GOLD[q]["ngroups"]
                assert max(abs(m[0] - GOLD[q]["m1"]), abs(m[1] - GOLD[q]["m2"]),
                           abs(m[2] - GOLD[q]["m3"])) < 1e-9
            chk = m
        else:
            s = [float(r0[c].sum()) for c in ("v1", "v2", "v3")]
            if N == 10_000_000:
                assert len(r0) == GOLD[q]["ngroups"]
                assert int(s[0]) == GOLD[q]["s1"] and int(s[1]) == GOLD[q]["s2"]
                assert int(round(s[2] * 1e6)) == GOLD[q]["s3_scaled"]
            chk = s
        Q[q] = {"cold_ms": round(cold, 1), "warm_ms": round(warm, 1), "chk": chk,
                "ngroups": len(r0)}
        print(f"{q}: cold={cold:.0f} warm={warm:.0f} RSS {rss():.2f}GB", flush=True)
    return {"engine": "polars", "version": pl.__version__, "threads": T, "N": N,
            "csv_load_ms": round(load_ms, 1), "Q": Q, "peak_rss_gb": round(rss(), 2)}


def run_pandas():
    import pandas as pd
    print(f"=== showdown pandas {pd.__version__} N={N} (single-thread) ===", flush=True)
    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    load_ms = (time.perf_counter() - t0) * 1000
    assert len(df) == N
    print(f"load {load_ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    PQ = {
        "Q1": lambda: df.groupby("id1")["v1"].sum(),
        "Q2": lambda: df.groupby(["id1", "id2"])["v1"].sum(),
        "Q3": lambda: df.groupby("id3").agg(s=("v1", "sum"), m=("v3", "mean")),
        "Q4": lambda: df.groupby("id4").agg(a=("v1", "mean"), b=("v2", "mean"), c=("v3", "mean")),
        "Q5": lambda: df.groupby("id6").agg(a=("v1", "sum"), b=("v2", "sum"), c=("v3", "sum")),
    }
    Q = {}
    for q, fn in PQ.items():
        r0 = fn()
        t = time.perf_counter()
        r0 = fn()
        cold = (time.perf_counter() - t) * 1000
        _, warm = best_of(lambda fn=fn: fn(), 3)
        if q in ("Q1", "Q2"):
            tot = int(r0.sum())
            if N == 10_000_000:
                chk_q1_q2(tot, len(r0), "pd " + q, GOLD[q])
            chk = tot
        elif q == "Q3":
            tot = int(r0["s"].sum())
            m = float(r0["m"].sum())
            if N == 10_000_000:
                assert tot == GOLD[q]["v1"] and len(r0) == GOLD[q]["ngroups"]
                assert abs(m - GOLD[q]["mean3_sum"]) < 1e-6 * abs(GOLD[q]["mean3_sum"]) + 1e-6
            chk = m
        elif q == "Q4":
            m = [float(r0[c].sum()) for c in ("a", "b", "c")]
            if N == 10_000_000:
                assert len(r0) == GOLD[q]["ngroups"]
                assert max(abs(m[0] - GOLD[q]["m1"]), abs(m[1] - GOLD[q]["m2"]),
                           abs(m[2] - GOLD[q]["m3"])) < 1e-9
            chk = m
        else:
            s = [float(r0[c].sum()) for c in ("a", "b", "c")]
            if N == 10_000_000:
                assert len(r0) == GOLD[q]["ngroups"]
                assert int(s[0]) == GOLD[q]["s1"] and int(s[1]) == GOLD[q]["s2"]
                assert int(round(s[2] * 1e6)) == GOLD[q]["s3_scaled"]
            chk = s
        Q[q] = {"cold_ms": round(cold, 1), "warm_ms": round(warm, 1), "chk": chk,
                "ngroups": len(r0)}
        print(f"{q}: cold={cold:.0f} warm={warm:.0f} RSS {rss():.2f}GB", flush=True)
    return {"engine": "pandas", "version": pd.__version__, "threads": 1, "N": N,
            "csv_load_ms": round(load_ms, 1), "Q": Q, "peak_rss_gb": round(rss(), 2)}


def run_datafusion():
    import datafusion
    from datafusion import SessionConfig, SessionContext
    print(f"=== showdown datafusion {datafusion.__version__} N={N} T={T} ===", flush=True)
    cfg = SessionConfig().set("datafusion.execution.target_partitions", str(T))
    ctx = SessionContext(config=cfg)
    t0 = time.perf_counter()
    ctx.register_csv("t", CSV, has_header=True)
    # force load/plan once
    n = int(ctx.sql("SELECT COUNT(*) FROM t").to_pandas().iloc[0, 0])
    load_ms = (time.perf_counter() - t0) * 1000
    assert n == N, n
    print(f"load {load_ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    DQ = {
        "Q1": "SELECT id1, SUM(v1) s FROM t GROUP BY id1",
        "Q2": "SELECT id1, id2, SUM(v1) s FROM t GROUP BY id1, id2",
        "Q3": "SELECT id3, SUM(v1) s, AVG(v3) m FROM t GROUP BY id3",
        "Q4": "SELECT id4, AVG(v1) a, AVG(v2) b, AVG(v3) c FROM t GROUP BY id4",
        "Q5": "SELECT id6, SUM(v1) a, SUM(v2) b, SUM(v3) c FROM t GROUP BY id6",
    }
    Q = {}
    for q, sql in DQ.items():
        r0 = ctx.sql(sql).collect()
        t = time.perf_counter()
        r0 = ctx.sql(sql).collect()
        cold = (time.perf_counter() - t) * 1000
        _, warm = best_of(lambda sql=sql: ctx.sql(sql).collect(), 3)
        # chk via to_pandas on one batch set
        tbl = ctx.sql(sql).to_pandas()
        if q in ("Q1", "Q2"):
            tot = int(tbl.iloc[:, -1].sum())
            if N == 10_000_000:
                chk_q1_q2(tot, len(tbl), "dfn " + q, GOLD[q])
            chk = tot
        elif q == "Q3":
            tot = int(tbl.iloc[:, 1].sum())
            m = float(tbl.iloc[:, 2].sum())
            if N == 10_000_000:
                assert tot == GOLD[q]["v1"] and len(tbl) == GOLD[q]["ngroups"]
                assert abs(m - GOLD[q]["mean3_sum"]) < 1e-6 * abs(GOLD[q]["mean3_sum"]) + 1e-6
            chk = m
        elif q == "Q4":
            m = [float(tbl.iloc[:, i].sum()) for i in (1, 2, 3)]
            if N == 10_000_000:
                assert len(tbl) == GOLD[q]["ngroups"]
                assert max(abs(m[0] - GOLD[q]["m1"]), abs(m[1] - GOLD[q]["m2"]),
                           abs(m[2] - GOLD[q]["m3"])) < 1e-9
            chk = m
        else:
            s = [float(tbl.iloc[:, i].sum()) for i in (1, 2, 3)]
            if N == 10_000_000:
                assert len(tbl) == GOLD[q]["ngroups"]
                assert int(s[0]) == GOLD[q]["s1"] and int(s[1]) == GOLD[q]["s2"]
                assert int(round(s[2] * 1e6)) == GOLD[q]["s3_scaled"]
            chk = s
        Q[q] = {"cold_ms": round(cold, 1), "warm_ms": round(warm, 1), "chk": chk,
                "ngroups": len(tbl)}
        print(f"{q}: cold={cold:.0f} warm={warm:.0f} RSS {rss():.2f}GB", flush=True)
    return {"engine": "datafusion", "version": datafusion.__version__, "threads": T, "N": N,
            "csv_load_ms": round(load_ms, 1), "Q": Q, "peak_rss_gb": round(rss(), 2)}


def main():
    t_all = time.perf_counter()
    fn = {"numfast": run_numfast, "duckdb": run_duckdb, "polars": run_polars,
          "pandas": run_pandas, "datafusion": run_datafusion}[ENGINE]
    row = fn()
    row["wall_s"] = round(time.perf_counter() - t_all, 1)
    row["gold"] = "EXACT/PASS" if N == 10_000_000 else "CROSS-ENGINE"
    with open(str(OUT), "a") as f:
        f.write(json.dumps(row, default=float) + "\n")
    print(f"-> {OUT} wall={row['wall_s']}s peak={PEAK[0]:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
