# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resident typed columns vs query-time encode: A/B/C on H2O 10M (heavy).

Memory: ~2GB RSS (pandas load transient 1.3GB + Arrow strings ~0.3GB +
resident int32 ~0.3GB). Time: minutes (CSV load ~15s, encodes ~1-3s each,
groupbys ~1-4s each). Proves: encode paid once (B prep) vs per-query (A);
no aggregation cached (every Q runs full groupby); Q2 pack_keys stays runtime.

Dataset: G1_1e7_1e2_0_0 (10M, na=0) ONLY. No 100M. GPU untouched (CPU path).
Goldens (independent pandas reference, probe-verified):
  Q1 groups=100 sum=29998789 | Q2 groups=10000 sum=29998789
  Q3 groups=100000 v1=29998789 mean3_sum=4999719.622344427
  Q4 groups=100 m1/m2/m3 sums (see GOLD)
  Q5 groups=100000 s1=29998789 s2=79989360 s3=499976651.40806097 (x1e6=499976651408061)

Usage (Git Bash, fork-first path):
  /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_resident_10M.py
"""

import gc
import json
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000

GOLD = {
    "Q1": {"ngroups": 100, "total": 29998789},
    "Q2": {"ngroups": 10000, "total": 29998789},
    "Q3": {"ngroups": 100000, "v1": 29998789, "mean3_sum": 4999719.622344427},
    "Q4": {"ngroups": 100, "m1": 299.98798187506526,
           "m2": 799.894179409978, "m3": 4999.766872833688},
    "Q5": {"ngroups": 100000, "s1": 29998789, "s2": 79989360,
           "s3": 499976651.40806097, "s3_scaled": 499976651408061},
}

import numpy as np
import psutil

PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 1e9


def build_kernel():
    from builder import MAIN

    return MAIN["build"](str(FORK))


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t) * 1000


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


def run_q(a, jobs, n):
    (res, st), _total = timed(lambda: run_graph(a, jobs, n))
    return res, st, st["total"]


def int_check(res, gold_total, gold_groups, label):
    assert isinstance(res, dict), f"{label}: groupby must return dict"
    assert len(res) == gold_groups, f"{label}: groups {len(res)} != {gold_groups}"
    tot = int(sum(int(v) for v in res.values()))
    assert tot == gold_total, f"{label}: total {tot} != {gold_total}"
    return tot


def main():
    import pandas as pd
    import pyarrow as pa

    print("=== resident-10M A/B/C ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    kernel = build_kernel()
    a = kernel.alias

    # ---- shared input: CSV -> typed/raw columns (common cost, outside A/B) ----
    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    load_ms = (time.perf_counter() - t0) * 1000
    assert len(df) == N
    print(f"csv_load {load_ms:.0f}ms RSS {rss():.2f}GB", flush=True)

    t0 = time.perf_counter()
    s_id1 = pa.array(df["id1"].to_numpy(), type=pa.string())
    s_id2 = pa.array(df["id2"].to_numpy(), type=pa.string())
    s_id3 = pa.array(df["id3"].to_numpy(), type=pa.string())
    id4 = df["id4"].to_numpy().astype(np.int32)
    id6 = df["id6"].to_numpy().astype(np.int32)
    v1 = df["v1"].to_numpy().astype(np.int32)
    v2 = df["v2"].to_numpy().astype(np.int32)
    v3 = df["v3"].to_numpy().astype(np.float64)
    str_bytes = (s_id1.get_total_buffer_size() + s_id2.get_total_buffer_size()
                 + s_id3.get_total_buffer_size())
    num_bytes = sum(x.nbytes for x in (id4, id6, v1, v2, v3))
    extract_ms = (time.perf_counter() - t0) * 1000
    print(f"extract {extract_ms:.0f}ms str_input={str_bytes/1e6:.1f}MB "
          f"num={num_bytes/1e6:.1f}MB RSS {rss():.2f}GB", flush=True)
    del df
    gc.collect()

    out = {"N": N, "csv_load_ms": load_ms, "extract_ms": extract_ms,
           "str_input_bytes": str_bytes, "num_input_bytes": num_bytes, "modes": {}}

    # ---------------- MODE A: query-time pattern encode ----------------
    A = {}
    r, st, _ = run_q(a, [a["ir_encode_pattern"]("k", s_id1, "id"),
                                        a["ir_series"]("v", v1),
                                        a["ir_groupby"]("g", "v", "k", "sum")], N)
    A["Q1"] = {"cold_ms": st["total"], "stages": st, "chk": int_check(r, 29998789, 100, "A.Q1")}
    r, st, _ = run_q(a, [a["ir_encode_pattern"]("k", s_id1, "id"),
                                        a["ir_series"]("v", v1),
                                        a["ir_groupby"]("g", "v", "k", "sum")], N)
    A["Q1"]["warm_ms"] = st["total"]
    print(f"A.Q1 cold={A['Q1']['cold_ms']:.0f}ms warm={st['total']:.0f}ms "
          f"(exec {st['execute']:.0f}ms) chk={A['Q1']['chk']} RSS {rss():.2f}GB", flush=True)

    r, st, _ = run_q(a, [a["ir_encode_pattern"]("c1", s_id1, "id"),
                                        a["ir_encode_pattern"]("c2", s_id2, "id"),
                                        a["ir_series"]("v", v1),
                                        a["ir_pack_keys"]("k", "c1", "c2"),
                                        a["ir_groupby"]("g", "v", "k", "sum")], N)
    A["Q2"] = {"cold_ms": st["total"], "stages": st, "chk": int_check(r, 29998789, 10000, "A.Q2")}
    r, st, _ = run_q(a, [a["ir_encode_pattern"]("c1", s_id1, "id"),
                                        a["ir_encode_pattern"]("c2", s_id2, "id"),
                                        a["ir_series"]("v", v1),
                                        a["ir_pack_keys"]("k", "c1", "c2"),
                                        a["ir_groupby"]("g", "v", "k", "sum")], N)
    A["Q2"]["warm_ms"] = st["total"]
    print(f"A.Q2 cold={A['Q2']['cold_ms']:.0f}ms warm={st['total']:.0f}ms "
          f"(exec {st['execute']:.0f}ms) chk={A['Q2']['chk']} RSS {rss():.2f}GB", flush=True)

    r, st, _ = run_q(a, [a["ir_encode_pattern"]("k", s_id3, "id"),
                                        a["ir_series"]("v1", v1), a["ir_series"]("v3", v3, "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                                              {"v1": ("sum",), "v3": ("mean",)})], N)
    m3 = float(sum(cell["v3"]["mean"] for cell in r.values()))
    assert len(r) == 100000
    assert int(sum(cell["v1"]["sum"] for cell in r.values())) == 29998789
    A["Q3"] = {"cold_ms": st["total"], "stages": st, "mean3_sum": m3,
               "mean3_diff": abs(m3 - GOLD["Q3"]["mean3_sum"])}
    assert A["Q3"]["mean3_diff"] < 1e-6 * abs(GOLD["Q3"]["mean3_sum"]) + 1e-6
    r, st, _ = run_q(a, [a["ir_encode_pattern"]("k", s_id3, "id"),
                                        a["ir_series"]("v1", v1), a["ir_series"]("v3", v3, "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                                              {"v1": ("sum",), "v3": ("mean",)})], N)
    A["Q3"]["warm_ms"] = st["total"]
    print(f"A.Q3 cold={A['Q3']['cold_ms']:.0f}ms warm={st['total']:.0f}ms "
          f"(exec {st['execute']:.0f}ms) mean3={m3!r} RSS {rss():.2f}GB", flush=True)

    r, st, _ = run_q(a, [a["ir_series"]("k", id4),
                                        a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
                                        a["ir_series"]("v3", v3, "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                              {"v1": ("mean",), "v2": ("mean",),
                                                               "v3": ("mean",)})], N)
    sums = {c: float(sum(cell[c]["mean"] for cell in r.values())) for c in ("v1", "v2", "v3")}
    assert len(r) == 100
    A["Q4"] = {"cold_ms": st["total"], "stages": st, "sums": sums,
               "maxdiff": max(abs(sums["v1"] - GOLD["Q4"]["m1"]),
                              abs(sums["v2"] - GOLD["Q4"]["m2"]),
                              abs(sums["v3"] - GOLD["Q4"]["m3"]))}
    assert A["Q4"]["maxdiff"] < 1e-9
    r, st, _ = run_q(a, [a["ir_series"]("k", id4),
                                        a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
                                        a["ir_series"]("v3", v3, "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                              {"v1": ("mean",), "v2": ("mean",),
                                                               "v3": ("mean",)})], N)
    A["Q4"]["warm_ms"] = st["total"]
    print(f"A.Q4 cold={A['Q4']['cold_ms']:.0f}ms warm={st['total']:.0f}ms "
          f"(exec {st['execute']:.0f}ms) maxdiff={A['Q4']['maxdiff']:.2e} RSS {rss():.2f}GB", flush=True)

    r, st, _ = run_q(a, [a["ir_series"]("k", id6),
                                        a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
                                        a["ir_series"]("v3", v3, "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                              {"v1": ("sum",), "v2": ("sum",),
                                                               "v3": ("sum",)})], N)
    s1 = int(sum(cell["v1"]["sum"] for cell in r.values()))
    s2 = int(sum(cell["v2"]["sum"] for cell in r.values()))
    s3 = float(sum(cell["v3"]["sum"] for cell in r.values()))
    assert len(r) == 100000 and s1 == 29998789 and s2 == 79989360
    A["Q5"] = {"cold_ms": st["total"], "stages": st, "s3": s3,
               "s3_scaled": int(round(s3 * 1e6)),
               "s3_diff": abs(s3 - GOLD["Q5"]["s3"])}
    assert A["Q5"]["s3_scaled"] == 499976651408061, A["Q5"]["s3_scaled"]
    r, st, _ = run_q(a, [a["ir_series"]("k", id6),
                                        a["ir_series"]("v1", v1), a["ir_series"]("v2", v2),
                                        a["ir_series"]("v3", v3, "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                              {"v1": ("sum",), "v2": ("sum",),
                                                               "v3": ("sum",)})], N)
    A["Q5"]["warm_ms"] = st["total"]
    print(f"A.Q5 cold={A['Q5']['cold_ms']:.0f}ms warm={st['total']:.0f}ms "
          f"(exec {st['execute']:.0f}ms) s3_scaled={A['Q5']['s3_scaled']} RSS {rss():.2f}GB", flush=True)
    A["rss_gb"] = rss()
    out["modes"]["A"] = A

    # ---------------- MODE B: resident typed (encode once) ----------------
    B = {}
    pre = {}
    for name, col in (("id1", {"values": s_id1, "prefix": "id"}),
                      ("id2", {"values": s_id2, "prefix": "id"}),
                      ("id3", {"values": s_id3, "prefix": "id"})):
        c, ms = timed(lambda col=col: a["resident_prepare"]({name: col}))
        pre[name] = {"ms": ms, "col": c[name]}
        print(f"B.prep {name}: {ms:.0f}ms codes={pre[name]['col']['codes'].size} "
              f"RSS {rss():.2f}GB", flush=True)
    c, ms = timed(lambda: a["resident_prepare"]({
        "id4": {"values": id4, "dtype": "int32"}, "id6": {"values": id6, "dtype": "int32"},
        "v1": {"values": v1, "dtype": "int32"}, "v2": {"values": v2, "dtype": "int32"},
        "v3": {"values": v3, "dtype": "float64"}}))
    pre["num"] = {"ms": ms}
    for k, v in c.items():
        pre[k] = {"ms": 0.0, "col": v}
    res = {k: pre[k]["col"] for k in ("id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3")}
    prep_ms = sum(v["ms"] for v in pre.values())
    res_bytes = sum(res[k]["codes"].nbytes for k in res)
    res_bytes += sum(res[k]["validity"].nbytes for k in ("id1", "id2", "id3"))
    B["prep_ms"] = prep_ms
    B["resident_bytes"] = res_bytes
    print(f"B.prep total {prep_ms:.0f}ms resident={res_bytes/1e6:.1f}MB "
          f"(vs str {str_bytes/1e6:.1f}MB) RSS {rss():.2f}GB", flush=True)
    # zero-copy check: numeric resident must view input buffers
    assert np.shares_memory(res["v1"]["codes"], v1) and np.shares_memory(res["id4"]["codes"], id4)

    k1, k2, k3 = res["id1"]["codes"], res["id2"]["codes"], res["id3"]["codes"]
    r, st, _ = run_q(a, [a["ir_series"]("k", k1), a["ir_series"]("v", res["v1"]["codes"]),
                                        a["ir_groupby"]("g", "v", "k", "sum")], N)
    B["Q1"] = {"cold_ms": st["total"], "stages": st, "chk": int_check(r, 29998789, 100, "B.Q1")}
    r, st, _ = run_q(a, [a["ir_series"]("k", k1), a["ir_series"]("v", res["v1"]["codes"]),
                                        a["ir_groupby"]("g", "v", "k", "sum")], N)
    B["Q1"]["warm_ms"] = st["total"]
    print(f"B.Q1 cold={B['Q1']['cold_ms']:.0f}ms warm={st['total']:.0f}ms chk={B['Q1']['chk']}", flush=True)

    r, st, _ = run_q(a, [a["ir_series"]("c1", k1), a["ir_series"]("c2", k2),
                                        a["ir_series"]("v", res["v1"]["codes"]),
                                        a["ir_pack_keys"]("k", "c1", "c2"),
                                        a["ir_groupby"]("g", "v", "k", "sum")], N)
    B["Q2"] = {"cold_ms": st["total"], "stages": st, "chk": int_check(r, 29998789, 10000, "B.Q2")}
    r, st, _ = run_q(a, [a["ir_series"]("c1", k1), a["ir_series"]("c2", k2),
                                        a["ir_series"]("v", res["v1"]["codes"]),
                                        a["ir_pack_keys"]("k", "c1", "c2"),
                                        a["ir_groupby"]("g", "v", "k", "sum")], N)
    B["Q2"]["warm_ms"] = st["total"]
    print(f"B.Q2 cold={B['Q2']['cold_ms']:.0f}ms warm={st['total']:.0f}ms chk={B['Q2']['chk']}", flush=True)

    r, st, _ = run_q(a, [a["ir_series"]("k", k3),
                                        a["ir_series"]("v1", res["v1"]["codes"]),
                                        a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                                              {"v1": ("sum",), "v3": ("mean",)})], N)
    m3 = float(sum(cell["v3"]["mean"] for cell in r.values()))
    assert len(r) == 100000 and int(sum(cell["v1"]["sum"] for cell in r.values())) == 29998789
    B["Q3"] = {"cold_ms": st["total"], "stages": st, "mean3_sum": m3,
               "mean3_diff": abs(m3 - GOLD["Q3"]["mean3_sum"])}
    assert B["Q3"]["mean3_diff"] < 1e-6 * abs(GOLD["Q3"]["mean3_sum"]) + 1e-6
    r, st, _ = run_q(a, [a["ir_series"]("k", k3),
                                        a["ir_series"]("v1", res["v1"]["codes"]),
                                        a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                                              {"v1": ("sum",), "v3": ("mean",)})], N)
    B["Q3"]["warm_ms"] = st["total"]
    print(f"B.Q3 cold={B['Q3']['cold_ms']:.0f}ms warm={st['total']:.0f}ms mean3={m3!r}", flush=True)

    r, st, _ = run_q(a, [a["ir_series"]("k", res["id4"]["codes"]),
                                        a["ir_series"]("v1", res["v1"]["codes"]),
                                        a["ir_series"]("v2", res["v2"]["codes"]),
                                        a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                              {"v1": ("mean",), "v2": ("mean",),
                                                               "v3": ("mean",)})], N)
    sums = {c: float(sum(cell[c]["mean"] for cell in r.values())) for c in ("v1", "v2", "v3")}
    assert len(r) == 100
    B["Q4"] = {"cold_ms": st["total"], "stages": st, "sums": sums,
               "maxdiff": max(abs(sums["v1"] - GOLD["Q4"]["m1"]),
                              abs(sums["v2"] - GOLD["Q4"]["m2"]),
                              abs(sums["v3"] - GOLD["Q4"]["m3"]))}
    assert B["Q4"]["maxdiff"] < 1e-9
    r, st, _ = run_q(a, [a["ir_series"]("k", res["id4"]["codes"]),
                                        a["ir_series"]("v1", res["v1"]["codes"]),
                                        a["ir_series"]("v2", res["v2"]["codes"]),
                                        a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                              {"v1": ("mean",), "v2": ("mean",),
                                                               "v3": ("mean",)})], N)
    B["Q4"]["warm_ms"] = st["total"]
    print(f"B.Q4 cold={B['Q4']['cold_ms']:.0f}ms warm={st['total']:.0f}ms maxdiff={B['Q4']['maxdiff']:.2e}",
          flush=True)

    r, st, _ = run_q(a, [a["ir_series"]("k", res["id6"]["codes"]),
                                        a["ir_series"]("v1", res["v1"]["codes"]),
                                        a["ir_series"]("v2", res["v2"]["codes"]),
                                        a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                              {"v1": ("sum",), "v2": ("sum",),
                                                               "v3": ("sum",)})], N)
    s1 = int(sum(cell["v1"]["sum"] for cell in r.values()))
    s2 = int(sum(cell["v2"]["sum"] for cell in r.values()))
    s3 = float(sum(cell["v3"]["sum"] for cell in r.values()))
    assert len(r) == 100000 and s1 == 29998789 and s2 == 79989360
    B["Q5"] = {"cold_ms": st["total"], "stages": st, "s3": s3,
               "s3_scaled": int(round(s3 * 1e6))}
    assert B["Q5"]["s3_scaled"] == 499976651408061, B["Q5"]["s3_scaled"]
    r, st, _ = run_q(a, [a["ir_series"]("k", res["id6"]["codes"]),
                                        a["ir_series"]("v1", res["v1"]["codes"]),
                                        a["ir_series"]("v2", res["v2"]["codes"]),
                                        a["ir_series"]("v3", res["v3"]["codes"], "float64"),
                                        a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                                              {"v1": ("sum",), "v2": ("sum",),
                                                               "v3": ("sum",)})], N)
    B["Q5"]["warm_ms"] = st["total"]
    print(f"B.Q5 cold={B['Q5']['cold_ms']:.0f}ms warm={st['total']:.0f}ms "
          f"s3_scaled={B['Q5']['s3_scaled']}", flush=True)
    B["rss_gb"] = rss()
    out["modes"]["B"] = {k: v for k, v in B.items() if not isinstance(v, np.ndarray)}

    # ---------------- summary A vs B ----------------
    print("\n--- A/B (cold ms, 10M) ---", flush=True)
    totA = sum(A[q]["cold_ms"] for q in ("Q1", "Q2", "Q3", "Q4", "Q5"))
    totBq = sum(B[q]["cold_ms"] for q in ("Q1", "Q2", "Q3", "Q4", "Q5"))
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        o, n = A[q]["cold_ms"], B[q]["cold_ms"]
        print(f"{q}: OLD={o:.0f} NEW={n:.0f} RATIO={o/n:.2f}x "
              f"thr {N/o/1e3:.0f}k/s -> {N/n/1e3:.0f}k/s", flush=True)
    print(f"queries: OLD={totA:.0f} NEW={totBq:.0f} RATIO={totA/totBq:.2f}x", flush=True)
    print(f"end-to-end: A={totA:.0f} B={prep_ms+totBq:.0f} (prep {prep_ms:.0f}) "
          f"RATIO={totA/(prep_ms+totBq):.2f}x", flush=True)
    enc_once = pre["id1"]["ms"] + pre["id2"]["ms"] + pre["id3"]["ms"]
    enc_per_q = (A["Q1"]["stages"]["execute"] - B["Q1"]["stages"]["execute"]
                 + A["Q2"]["stages"]["execute"] - B["Q2"]["stages"]["execute"]
                 + A["Q3"]["stages"]["execute"] - B["Q3"]["stages"]["execute"])
    print(f"encode: paid-once={enc_once:.0f}ms vs per-query sum~{enc_per_q:.0f}ms", flush=True)

    json_safe = json.loads(json.dumps(out, default=float))
    with open(FORK / "tests" / "heavy" / "bench_resident_10M.json", "w") as f:
        json.dump(json_safe, f, indent=1)
    print("JSON written", flush=True)
    return out, res, a


if __name__ == "__main__":
    main()
