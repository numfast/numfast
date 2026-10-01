# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resident typed columns B on H2O 100M (heavy, B-only).

Memory: ~15GB RSS transient (pandas load + Arrow strings + resident int32).
Time: ~10-20 min (CSV load ~2-3 min, prep ~1 min, Qx2 ~6-8 min, pandas refs ~2 min).
Proves: resident B scales 10M -> 100M via PUBLIC path only
(CSV -> Schema -> resident_prepare -> typed resident -> IR -> Planner ->
Runtime -> CPU Driver -> GroupBy). No benchmark bypass. No 1B. GPU untouched.

Dataset: G1_1e8_1e2_0_0 (100M, na=0) ONLY.
Groups (structural, same as 10M): Q1=100 Q2=10000 Q3=100000 Q4=100 Q5=100000.
Totals are data-dependent -> validated against independent numpy column sums
(int exact) and pandas groupby refs where cheap (Q1/Q4); float only via
specs/conformance-profile.toml thresholds.

Usage (Git Bash, fork-first path, H2O python explicit):
  C:/App/competitions/H2O/python310/python.exe tests/heavy/bench_resident_100M.py
"""

import gc
import json
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))

CSV = "C:/App/competitions/H2O/data/G1_1e8_1e2_0_0.csv"
N = 100_000_000
GROUPS = {"Q1": 100, "Q2": 10000, "Q3": 100000, "Q4": 100, "Q5": 100000}

import numpy as np
import psutil

PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 1e9


def build_kernel():
    from builder import MAIN

    return MAIN["build"](str(FORK))


def run_graph(a, jobs, n):
    st = {}
    s = time.perf_counter()
    g = a["compile"](jobs)
    st["compile"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    g = a["optimize"](g)
    st["optimize"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    r = a["evaluate"](g, "cpu", n)
    st["execute"] = (time.perf_counter() - s) * 1000
    st["total"] = st["compile"] + st["optimize"] + st["execute"]
    return r["result"], st


def main():
    import pandas as pd
    import pyarrow as pa

    print("=== resident-100M B (public path only) ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    kernel = build_kernel()
    a = kernel.alias
    out = {"N": N, "dataset": "G1_1e8_1e2_0_0", "mode": "B-resident-typed"}

    # ---- shared input: CSV -> typed/raw columns (common cost, outside B) ----
    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    out["csv_load_ms"] = (time.perf_counter() - t0) * 1000
    assert len(df) == N, len(df)
    print(f"csv_load {out['csv_load_ms']:.0f}ms RSS {rss():.2f}GB", flush=True)

    # independent refs BEFORE extract (numpy sums = int-exact oracles)
    t0 = time.perf_counter()
    ref_v1 = int(df["v1"].to_numpy().astype(np.int64).sum())
    ref_v2 = int(df["v2"].to_numpy().astype(np.int64).sum())
    ref_v3 = float(df["v3"].to_numpy().astype(np.float64).sum())
    print(f"numpy col sums {1000*(time.perf_counter()-t0):.0f}ms "
          f"v1={ref_v1} v2={ref_v2} v3={ref_v3!r} RSS {rss():.2f}GB", flush=True)

    t0 = time.perf_counter()
    s_id1 = pa.array(df["id1"].to_numpy(), type=pa.string())
    s_id2 = pa.array(df["id2"].to_numpy(), type=pa.string())
    s_id3 = pa.array(df["id3"].to_numpy(), type=pa.string())
    id4 = df["id4"].to_numpy().astype(np.int32)
    id6 = df["id6"].to_numpy().astype(np.int32)
    v1 = df["v1"].to_numpy().astype(np.int32)
    v2 = df["v2"].to_numpy().astype(np.int32)
    v3 = df["v3"].to_numpy().astype(np.float64)
    out["extract_ms"] = (time.perf_counter() - t0) * 1000
    out["str_input_bytes"] = (s_id1.get_total_buffer_size() + s_id2.get_total_buffer_size()
                              + s_id3.get_total_buffer_size())
    out["num_input_bytes"] = sum(x.nbytes for x in (id4, id6, v1, v2, v3))
    print(f"extract {out['extract_ms']:.0f}ms str={out['str_input_bytes']/1e9:.2f}GB "
          f"num={out['num_input_bytes']/1e9:.2f}GB RSS {rss():.2f}GB", flush=True)

    # cheap independent pandas refs: Q1 (100 groups) + Q4 (100 groups)
    t0 = time.perf_counter()
    q1ref = df.groupby("id1", sort=True)["v1"].sum()
    out["pandas_Q1_ms"] = (time.perf_counter() - t0) * 1000
    assert len(q1ref) == 100 and int(q1ref.sum()) == ref_v1
    print(f"pandas Q1ref {out['pandas_Q1_ms']:.0f}ms groups={len(q1ref)} "
          f"total={int(q1ref.sum())} RSS {rss():.2f}GB", flush=True)
    t0 = time.perf_counter()
    q4ref = df.groupby("id4", sort=True)[["v1", "v2", "v3"]].mean(numeric_only=True)
    out["pandas_Q4_ms"] = (time.perf_counter() - t0) * 1000
    assert len(q4ref) == 100
    print(f"pandas Q4ref {out['pandas_Q4_ms']:.0f}ms RSS {rss():.2f}GB", flush=True)
    del df
    gc.collect()

    # ---- resident_prepare ONCE per dataset (never per Q) ----
    pre = {}
    for name, col in (("id1", {"values": s_id1, "prefix": "id"}),
                      ("id2", {"values": s_id2, "prefix": "id"}),
                      ("id3", {"values": s_id3, "prefix": "id"})):
        t = time.perf_counter()
        c = a["resident_prepare"]({name: col})
        ms = (time.perf_counter() - t) * 1000
        pre[name] = {"ms": ms, "col": c[name]}
        print(f"prep {name}: {ms:.0f}ms codes={pre[name]['col']['codes'].size} "
              f"RSS {rss():.2f}GB", flush=True)
    t = time.perf_counter()
    c = a["resident_prepare"]({
        "id4": {"values": id4, "dtype": "int32"}, "id6": {"values": id6, "dtype": "int32"},
        "v1": {"values": v1, "dtype": "int32"}, "v2": {"values": v2, "dtype": "int32"},
        "v3": {"values": v3, "dtype": "float64"}})
    num_ms = (time.perf_counter() - t) * 1000
    for k, v in c.items():
        pre[k] = {"ms": 0.0, "col": v}
    res = {k: pre[k]["col"] for k in ("id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3")}
    out["prep_ms"] = sum(v["ms"] for v in pre.values()) + num_ms
    out["prep_detail_ms"] = {k: round(v["ms"], 1) for k, v in pre.items()}
    out["prep_detail_ms"]["__numeric_block"] = round(num_ms, 1)
    out["resident_bytes"] = (sum(res[k]["codes"].nbytes for k in res)
                             + sum(res[k]["validity"].nbytes for k in ("id1", "id2", "id3")))
    print(f"prep TOTAL {out['prep_ms']:.0f}ms resident={out['resident_bytes']/1e9:.2f}GB "
          f"(vs str {out['str_input_bytes']/1e9:.2f}GB) RSS {rss():.2f}GB", flush=True)
    assert np.shares_memory(res["v1"]["codes"], v1) and np.shares_memory(res["id4"]["codes"], id4)
    print("zero-copy numeric resident: OK", flush=True)
    del s_id1, s_id2, s_id3
    gc.collect()

    Q = {}
    k1, k2, k3 = res["id1"]["codes"], res["id2"]["codes"], res["id3"]["codes"]

    def consume_ms(fn):
        t = time.perf_counter()
        val = fn()
        return val, (time.perf_counter() - t) * 1000

    # ---- Q1 ----
    r, st = run_graph(a, [a["ir_series"]("k", k1), a["ir_series"]("v", res["v1"]["codes"]),
                          a["ir_groupby"]("g", "v", "k", "sum")], N)
    tot, c_ms = consume_ms(lambda: int(sum(int(v) for v in r.values())))
    assert len(r) == GROUPS["Q1"] and tot == ref_v1, (len(r), tot, ref_v1)
    assert tot == int(q1ref.sum())
    Q["Q1"] = {"cold_ms": st["total"], "stages": st, "chk": tot, "consume_ms": c_ms,
               "rss_gb": rss()}
    r2, st2 = run_graph(a, [a["ir_series"]("k", k1), a["ir_series"]("v", res["v1"]["codes"]),
                            a["ir_groupby"]("g", "v", "k", "sum")], N)
    Q["Q1"]["warm_ms"] = st2["total"]
    assert int(sum(int(v) for v in r2.values())) == ref_v1
    print(f"Q1 cold={st['total']:.0f} (exec {st['execute']:.0f}) warm={st2['total']:.0f}ms "
          f"chk={tot} RSS {rss():.2f}GB", flush=True)

    # ---- Q2 (pack_keys stays runtime) ----
    r, st = run_graph(a, [a["ir_series"]("c1", k1), a["ir_series"]("c2", k2),
                          a["ir_series"]("v", res["v1"]["codes"]),
                          a["ir_pack_keys"]("k", "c1", "c2"),
                          a["ir_groupby"]("g", "v", "k", "sum")], N)
    tot, c_ms = consume_ms(lambda: int(sum(int(v) for v in r.values())))
    assert len(r) == GROUPS["Q2"] and tot == ref_v1, (len(r), tot)
    Q["Q2"] = {"cold_ms": st["total"], "stages": st, "chk": tot, "consume_ms": c_ms,
               "rss_gb": rss()}
    r2, st2 = run_graph(a, [a["ir_series"]("c1", k1), a["ir_series"]("c2", k2),
                            a["ir_series"]("v", res["v1"]["codes"]),
                            a["ir_pack_keys"]("k", "c1", "c2"),
                            a["ir_groupby"]("g", "v", "k", "sum")], N)
    Q["Q2"]["warm_ms"] = st2["total"]
    assert int(sum(int(v) for v in r2.values())) == ref_v1
    print(f"Q2 cold={st['total']:.0f} (exec {st['execute']:.0f}) warm={st2['total']:.0f}ms "
          f"chk={tot} RSS {rss():.2f}GB", flush=True)

    # ---- Q3 ----
    r, st = run_graph(a, [a["ir_series"]("k", k3),
                          a["ir_series"]("v1", res["v1"]["codes"]),
                          a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                          a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                                {"v1": ("sum",), "v3": ("mean",)})], N)
    (s1, m3), c_ms = consume_ms(lambda: (
        int(sum(cell["v1"]["sum"] for cell in r.values())),
        float(sum(cell["v3"]["mean"] for cell in r.values()))))
    exp_q3 = int(np.unique(np.asarray(k3)).size)  # id cardinality grows with N (1M at 100M)
    assert len(r) == exp_q3 and s1 == ref_v1, (len(r), exp_q3, s1)
    Q["Q3"] = {"cold_ms": st["total"], "stages": st, "chk_v1": s1, "mean3_sum": m3,
               "ngroups": len(r), "consume_ms": c_ms, "rss_gb": rss()}
    r2, st2 = run_graph(a, [a["ir_series"]("k", k3),
                            a["ir_series"]("v1", res["v1"]["codes"]),
                            a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                            a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                                  {"v1": ("sum",), "v3": ("mean",)})], N)
    Q["Q3"]["warm_ms"] = st2["total"]
    m3w = float(sum(cell["v3"]["mean"] for cell in r2.values()))
    Q["Q3"]["mean3_warm"] = m3w
    Q["Q3"]["mean3_cold_warm_diff"] = abs(m3 - m3w)
    assert int(sum(cell["v1"]["sum"] for cell in r2.values())) == ref_v1
    assert abs(m3 - m3w) <= max(1e-12, 1e-12 * abs(m3))  # profile f64
    print(f"Q3 cold={st['total']:.0f} (exec {st['execute']:.0f}) warm={st2['total']:.0f}ms "
          f"v1={s1} mean3={m3!r} cold-warm diff={abs(m3-m3w):.2e} RSS {rss():.2f}GB", flush=True)

    # ---- Q4 (pandas independent ref, profile f64 tolerance) ----
    r, st = run_graph(a, [a["ir_series"]("k", res["id4"]["codes"]),
                          a["ir_series"]("v1", res["v1"]["codes"]),
                          a["ir_series"]("v2", res["v2"]["codes"]),
                          a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                          a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                {"v1": ("mean",), "v2": ("mean",),
                                                 "v3": ("mean",)})], N)
    sums, c_ms = consume_ms(lambda: {c: float(sum(cell[c]["mean"] for cell in r.values()))
                                     for c in ("v1", "v2", "v3")})
    assert len(r) == GROUPS["Q4"]
    prefs = {"v1": float(q4ref["v1"].sum()), "v2": float(q4ref["v2"].sum()),
             "v3": float(q4ref["v3"].sum())}
    diffs = {c: abs(sums[c] - prefs[c]) for c in sums}
    tol_ok = all(d <= max(1e-12, 1e-12 * abs(prefs[c])) for c, d in diffs.items())
    assert tol_ok, (diffs, prefs, sums)
    Q["Q4"] = {"cold_ms": st["total"], "stages": st, "sums": sums, "pandas_ref": prefs,
               "maxdiff": max(diffs.values()), "consume_ms": c_ms, "rss_gb": rss()}
    r2, st2 = run_graph(a, [a["ir_series"]("k", res["id4"]["codes"]),
                            a["ir_series"]("v1", res["v1"]["codes"]),
                            a["ir_series"]("v2", res["v2"]["codes"]),
                            a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                            a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                  {"v1": ("mean",), "v2": ("mean",),
                                                   "v3": ("mean",)})], N)
    Q["Q4"]["warm_ms"] = st2["total"]
    print(f"Q4 cold={st['total']:.0f} (exec {st['execute']:.0f}) warm={st2['total']:.0f}ms "
          f"maxdiff vs pandas={max(diffs.values()):.2e} RSS {rss():.2f}GB", flush=True)

    # ---- Q5 ----
    r, st = run_graph(a, [a["ir_series"]("k", res["id6"]["codes"]),
                          a["ir_series"]("v1", res["v1"]["codes"]),
                          a["ir_series"]("v2", res["v2"]["codes"]),
                          a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                          a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                {"v1": ("sum",), "v2": ("sum",),
                                                 "v3": ("sum",)})], N)
    (s1, s2, s3), c_ms = consume_ms(lambda: (
        int(sum(cell["v1"]["sum"] for cell in r.values())),
        int(sum(cell["v2"]["sum"] for cell in r.values())),
        float(sum(cell["v3"]["sum"] for cell in r.values()))))
    assert s1 == ref_v1 and s2 == ref_v2, (len(r), s1, s2)
    assert abs(s3 - ref_v3) <= max(1e-12, 1e-12 * abs(ref_v3)) * max(1.0, abs(ref_v3)), (s3, ref_v3)
    sc = int(round(s3 * 1e6))
    exp_q5 = int(np.unique(np.asarray(res["id6"]["codes"])).size)
    assert len(r) == exp_q5, (len(r), exp_q5)
    Q["Q5"] = {"cold_ms": st["total"], "stages": st, "s1": s1, "s2": s2, "s3": s3,
               "s3_scaled": sc, "s3_vs_colsum_diff": abs(s3 - ref_v3), "ngroups": len(r),
               "consume_ms": c_ms, "rss_gb": rss()}
    r2, st2 = run_graph(a, [a["ir_series"]("k", res["id6"]["codes"]),
                            a["ir_series"]("v1", res["v1"]["codes"]),
                            a["ir_series"]("v2", res["v2"]["codes"]),
                            a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                            a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                  {"v1": ("sum",), "v2": ("sum",),
                                                   "v3": ("sum",)})], N)
    Q["Q5"]["warm_ms"] = st2["total"]
    s3w = float(sum(cell["v3"]["sum"] for cell in r2.values()))
    Q["Q5"]["s3_warm"] = s3w
    Q["Q5"]["s3_scaled_warm"] = int(round(s3w * 1e6))
    assert Q["Q5"]["s3_scaled_warm"] == sc  # scaled-int exact determinism
    print(f"Q5 cold={st['total']:.0f} (exec {st['execute']:.0f}) warm={st2['total']:.0f}ms "
          f"s1={s1} s2={s2} s3_scaled={sc} RSS {rss():.2f}GB", flush=True)

    # ---- diagnostic bottleneck split (NOT public path; labels diagnostic) ----
    diag = {}
    for label, keys, ncols in (("Q3_keys", k3, 2), ("Q5_keys", res["id6"]["codes"], 3)):
        t = time.perf_counter()
        uk, inv = np.unique(np.asarray(keys), return_inverse=True)
        t_uniq = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        w = np.ones(keys.size, dtype=np.int64)
        np.bincount(inv, weights=w)
        t_bin = (time.perf_counter() - t) * 1000
        diag[label] = {"ngroups": int(uk.size), "np_unique_ms": round(t_uniq, 1),
                       "bincount1_ms": round(t_bin, 1)}
    # pack_keys-only diagnostic for Q2 composite share
    s = time.perf_counter()
    _g = a["compile"]([a["ir_series"]("c1", k1), a["ir_series"]("c2", k2),
                       a["ir_pack_keys"]("k", "c1", "c2")])
    _g = a["optimize"](_g)
    _r = a["evaluate"](_g, "cpu", N)
    diag["pack_keys_only_ms"] = round((time.perf_counter() - s) * 1000, 1)
    out["diagnostic"] = diag
    print(f"diagnostic unique/bincount: {diag}", flush=True)

    out["queries"] = Q
    out["refs"] = {"v1_sum": ref_v1, "v2_sum": ref_v2, "v3_sum": ref_v3}
    out["peak_rss_gb"] = rss()
    with open(FORK / "tests" / "heavy" / "bench_resident_100M.json", "w") as f:
        json.dump(json.loads(json.dumps(out, default=float)), f, indent=1)
    print(f"JSON written peak RSS {rss():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
