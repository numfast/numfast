# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Honest B+D production 10M T=16 query-only (NEW file; existing untouched).

Production path ONLY (builder src/* via MAIN["build"], no research kernels):
B = resident snapshot (encode-once K1..V3, integrity fp gate);
D = production CPU driver MT (NUMFAST_THREADS=16, native+routed, M from data).
Query-only: snapshot/compile/warmup outside timing; best-of-3 execute; chk exact.
Plus: Q2 pack split (pack-only vs agg-only vs e2e) + generic build-time
pre-encoded composite (probe kmin/kmax/m2, int32-direct when safe, no H2O
hardcode); Q4 breakdown (per-column single-mean, carry vs dict delta, f64).
 comparators DuckDB-16/Polars-16 reused from past bench_compete_trim (no rerun).
GPU untouched. No 1B. No query-cache, no fixed-M. Seed 42 (data fixed H2O).

Usage (Git Bash, ONE process, sequential):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
  timeout 550 /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_prod16_honest.py
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ["NUMFAST_THREADS"] = "16"
T = 16

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))

import numpy as np
import psutil

PROC = psutil.Process()
CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000
SNAP = FORK / "scratch" / "snap_G1_1e7_1e2_0_0"
COLS = ("K1", "K2", "K3", "K4", "K6", "V1", "V2", "V3")
OUT = FORK / "tests" / "heavy" / "bench_prod16_honest.json"
GOLD = {
    "Q1": {"ngroups": 100, "total": 29998789},
    "Q2": {"ngroups": 10000, "total": 29998789},
    "Q3": {"ngroups": 100000, "v1": 29998789, "mean3_sum": 4999719.622344427},
    "Q4": {"ngroups": 100, "m1": 299.98798187506526,
           "m2": 799.894179409978, "m3": 4999.766872833688},
    "Q5": {"ngroups": 100000, "s1": 29998789, "s2": 79989360,
           "s3_scaled": 499976651408061},
}


def rss():
    return PROC.memory_info().rss / 1e9


def fp_of(a):
    return {k: {"max": int(a[k].max()), "min": int(a[k].min()),
                "sum": int(a[k].astype(np.int64).sum())} for k in COLS}


def best_of(fn, reps=3):
    best, out = 1e18, None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t) * 1000)
    return out, best


def carry_chk(q, bufs):
    c = bufs["g#carry"]
    if q in ("Q1", "Q2"):
        return {"total": int(c.sums["v"].sum()), "ngroups": int(c.ngroups)}
    if q == "Q3":
        return {"v1": int(c.sums["v1"].sum()),
                "mean3_sum": float(c.means("v3").sum()),
                "ngroups": int(c.ngroups)}
    if q == "Q4":
        return {"means": {col: float(c.means(col).sum())
                          for col in ("v1", "v2", "v3")},
                "ngroups": int(c.ngroups)}
    return {"s1": int(c.sums["v1"].sum()), "s2": int(c.sums["v2"].sum()),
            "s3": float(c.sums["v3"].sum()), "ngroups": int(c.ngroups)}


def dict_chk(q, res):
    if q in ("Q1", "Q2"):
        return {"total": int(sum(int(v) for v in res.values())),
                "ngroups": len(res)}
    if q == "Q3":
        return {"v1": int(sum(int(c["v1"]["sum"]) for c in res.values())),
                "mean3_sum": float(sum(c["v3"]["mean"] for c in res.values())),
                "ngroups": len(res)}
    if q == "Q4":
        return {"means": {c: float(sum(cell[c]["mean"] for cell in res.values()))
                          for c in ("v1", "v2", "v3")}, "ngroups": len(res)}
    return {"s1": int(sum(int(c["v1"]["sum"]) for c in res.values())),
            "s2": int(sum(int(c["v2"]["sum"]) for c in res.values())),
            "s3": float(sum(float(c["v3"]["sum"]) for c in res.values())),
            "ngroups": len(res)}


def gate(q, chk):
    g = GOLD[q]
    if q in ("Q1", "Q2"):
        assert chk["total"] == g["total"] and chk["ngroups"] == g["ngroups"], (q, chk)
    elif q == "Q3":
        assert chk["v1"] == g["v1"] and chk["ngroups"] == g["ngroups"], (q, chk)
        assert abs(chk["mean3_sum"] - g["mean3_sum"]) <= 1e-6 * abs(g["mean3_sum"]) + 1e-6, (q, chk)
    elif q == "Q4":
        assert chk["ngroups"] == g["ngroups"], (q, chk)
        md = max(abs(chk["means"]["v1"] - g["m1"]),
                 abs(chk["means"]["v2"] - g["m2"]),
                 abs(chk["means"]["v3"] - g["m3"]))
        assert md < 1e-9, (q, md, chk)
    else:
        assert chk["s1"] == g["s1"] and chk["s2"] == g["s2"], (q, chk)
        assert chk["ngroups"] == g["ngroups"], (q, chk)
        assert int(round(chk["s3"] * 1e6)) == g["s3_scaled"], (q, chk)


def generic_preencode(c1, c2):
    """Generic build-time composite: probe minima/maxima/M2 from data
    (Python ints, no dataset branches); int32 c1*M2+c2 when bound fits,
    else int64 radix. Returns (key, meta, build_ms_one_shot_outside_query)."""
    c1 = np.ascontiguousarray(c1)
    c2 = np.ascontiguousarray(c2)
    t = time.perf_counter()
    k1min, k1max = int(c1.min()), int(c1.max())
    k2min, k2max = int(c2.min()), int(c2.max())
    m2 = k2max + 1 if k2min >= 0 else None
    if k1min >= 0 and k2min >= 0 and m2 is not None:
        bound = k1max * m2 + k2max
        if bound <= 2 ** 31 - 1:
            out = np.empty(c1.size, dtype=np.int32)
            np.multiply(c1, np.int32(m2), out=out)
            np.add(out, c2, out=out)
            ms = (time.perf_counter() - t) * 1000
            return out, {"dtype": "int32", "m2": m2, "bound": bound,
                         "k1": [k1min, k1max], "k2": [k2min, k2max]}, ms
    comp = c1.astype(np.int64, copy=False) * np.int64(m2 or 1) + c2.astype(np.int64, copy=False)
    ms = (time.perf_counter() - t) * 1000
    return np.ascontiguousarray(comp), {"dtype": "int64", "m2": m2}, ms


def main():
    from builder import MAIN
    t_all = time.perf_counter()
    print(f"=== prod16-honest 10M T={T} pid={os.getpid()} ===", flush=True)
    a = MAIN["build"](str(FORK)).alias
    meta = json.loads((SNAP / "meta.json").read_text())
    t = time.perf_counter()
    arr = {k: np.load(str(SNAP / f"{k}.npy")) for k in COLS}
    load_ms = (time.perf_counter() - t) * 1000
    assert fp_of(arr) == meta["fp"], "snapshot integrity STOP"
    assert all(arr[k].shape == (N,) for k in COLS)
    assert int(arr["V1"].astype(np.int64).sum()) == GOLD["Q1"]["total"]
    print(f"snap_load {load_ms:.0f}ms RSS {rss():.2f}GB fp OK", flush=True)
    K1, K2, K3, K4, K6, V1, V2, V3 = (arr[k] for k in COLS)

    JD = {
        "Q1": [a["ir_series"]("k", K1), a["ir_series"]("v", V1),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q2": [a["ir_series"]("c1", K1), a["ir_series"]("c2", K2),
               a["ir_series"]("v", V1),
               a["ir_pack_keys"]("k", "c1", "c2", mode="radix"),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q3": [a["ir_series"]("k", K3), a["ir_series"]("v1", V1),
               a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                     {"v1": ("sum",), "v3": ("mean",)})],
        "Q4": [a["ir_series"]("k", K4), a["ir_series"]("v1", V1),
               a["ir_series"]("v2", V2), a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("mean",), "v2": ("mean",),
                                      "v3": ("mean",)})],
        "Q5": [a["ir_series"]("k", K6), a["ir_series"]("v1", V1),
               a["ir_series"]("v2", V2), a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("sum",), "v2": ("sum",),
                                      "v3": ("sum",)})],
    }
    compd, compc = {}, {}
    for q, jj in JD.items():
        compd[q] = a["optimize"](a["compile"](jj))
        last = jj[-1]
        p = dict(last["params"])
        p["result"] = "carry"
        nn = dict(last)
        nn["params"] = p
        compc[q] = a["optimize"](a["compile"](jj[:-1] + [nn]))
    bufs = a["cpu_execute"](compc["Q1"]["nodes"])
    gate("Q1", carry_chk("Q1", bufs))
    assert bufs["g#groupindex"]["threads"] == T
    del bufs
    gc.collect()
    print(f"compile+warmup ok RSS {rss():.2f}GB", flush=True)

    # ---- PART 1: honest production Q1-Q5 query-only T=16 ----
    prod = {}
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        bufs, cms = best_of(lambda: a["cpu_execute"](compc[q]["nodes"]), 3)
        cc = carry_chk(q, bufs)
        gate(q, cc)
        gi = dict(bufs.get("g#groupindex", {}))
        assert gi.get("threads") == T, (q, gi)
        del bufs
        gc.collect()
        res, dms = best_of(lambda: a["evaluate"](compd[q], "cpu", N)["result"], 3)
        dc = dict_chk(q, res)
        gate(q, dc)
        del res
        gc.collect()
        prod[q] = {"carry_ms": round(cms, 1), "dict_ms": round(dms, 1),
                   "materialize_ms": round(dms - cms, 1),
                   "stages": {k: (round(v, 1) if isinstance(v, float) else v)
                              for k, v in gi.items()
                              if k in ("gi_ms", "agg_ms", "carry_ms", "merge_ms",
                                       "backend", "threads", "strategy", "reason")},
                   "chk": "EXACT/PASS"}
        print(f"{q}: carry={cms:.1f} dict={dms:.1f} "
              f"agg={gi.get('agg_ms')} carry_st={gi.get('carry_ms')} "
              f"merge={gi.get('merge_ms')} be={gi.get('backend')} "
              f"chk_ng={cc['ngroups']} RSS {rss():.2f}GB", flush=True)

    # ---- PART 2: Q2 pack split + generic build-time pre-encode ----
    pack_g = a["optimize"](a["compile"](
        [a["ir_series"]("c1", K1), a["ir_series"]("c2", K2),
         a["ir_pack_keys"]("k", "c1", "c2", mode="radix")]))
    _, pack_ms = best_of(lambda: a["cpu_execute"](pack_g["nodes"]), 3)
    P2, pre_meta, build_ms = generic_preencode(K1, K2)
    assert str(P2.dtype) == pre_meta["dtype"]
    gbp = a["optimize"](a["compile"](
        [a["ir_series"]("k", P2), a["ir_series"]("v", V1),
         a["ir_groupby"]("g", "v", "k", "sum")]))
    last = gbp
    res_pre, agg_pre_ms = best_of(lambda: a["evaluate"](gbp, "cpu", N)["result"], 3)
    dc_pre = dict_chk("Q2", res_pre)
    gate("Q2", dc_pre)
    # per-group exact vs baseline Q2 dict (sorted items compare, exact ints)
    res_base = a["evaluate"](compd["Q2"], "cpu", N)["result"]
    exact_groups = (len(res_pre) == len(res_base) == GOLD["Q2"]["ngroups"]
                    and sorted(res_pre.items()) == sorted(res_base.items()))
    assert exact_groups, "pre-encoded Q2 per-group mismatch"
    del res_pre, res_base
    gc.collect()
    # agg-only on query-time packed key (pack outside timing) for split check
    Pq = a["cpu_execute"](pack_g["nodes"])["k"]
    gq = a["optimize"](a["compile"](
        [a["ir_series"]("k", Pq), a["ir_series"]("v", V1),
         a["ir_groupby"]("g", "v", "k", "sum")]))
    _, agg_only_ms = best_of(lambda: a["cpu_execute"](gq["nodes"]), 3)
    del Pq
    gc.collect()
    q2 = {"pack_query_ms": round(pack_ms, 1),
          "agg_only_ms": round(agg_only_ms, 1),
          "e2e_carry_ms": prod["Q2"]["carry_ms"],
          "e2e_dict_ms": prod["Q2"]["dict_ms"],
          "preenc_build_ms": round(build_ms, 1),
          "preenc_query_dict_ms": round(agg_pre_ms, 1),
          "preenc_meta": pre_meta,
          "preenc_exact_groups": bool(exact_groups),
          "works": bool(exact_groups)}
    print(f"Q2split: pack={pack_ms:.1f} agg_only={agg_only_ms:.1f} "
          f"e2e_carry={prod['Q2']['carry_ms']} preenc_build={build_ms:.1f} "
          f"preenc_q={agg_pre_ms:.1f} exact={exact_groups}", flush=True)

    # ---- PART 3: Q4 breakdown (1-mean x3, carry vs dict, f64 vs int) ----
    def single_mean(col, v, dtype=None):
        kw = {} if dtype is None else {"dtype": dtype}
        jj = [a["ir_series"]("k", K4), a["ir_series"]("v", v, **kw),
              a["ir_groupby"]("g", "v", "k", "mean")]
        cd = a["optimize"](a["compile"](jj))
        p = dict(jj[-1]["params"])
        p["result"] = "carry"
        nn = dict(jj[-1])
        nn["params"] = p
        cc = a["optimize"](a["compile"](jj[:-1] + [nn]))
        b, cm = best_of(lambda: a["cpu_execute"](cc["nodes"]), 3)
        tot = float(b["g#carry"].means("v").sum())
        gi = dict(b.get("g#groupindex", {}))
        del b
        gc.collect()
        r, dm = best_of(lambda: a["evaluate"](cd, "cpu", N)["result"], 3)
        del r
        gc.collect()
        return cm, dm, tot, gi

    q4b = {}
    for name, v, dt in (("v1_int", V1, None), ("v2_int", V2, None),
                        ("v3_f64", V3, "float64")):
        cm, dm, tot, gi = single_mean(name, v, dt)
        q4b[name] = {"carry_ms": round(cm, 1), "dict_ms": round(dm, 1),
                     "mean_sum": tot,
                     "agg_ms": gi.get("agg_ms"), "carry_ms_st": gi.get("carry_ms"),
                     "merge_ms": gi.get("merge_ms"), "backend": gi.get("backend")}
        print(f"Q4.{name}: carry={cm:.1f} dict={dm:.1f} agg={gi.get('agg_ms')} "
              f"sum={tot!r}", flush=True)
    assert abs(q4b["v1_int"]["mean_sum"] - GOLD["Q4"]["m1"]) < 1e-9
    assert abs(q4b["v2_int"]["mean_sum"] - GOLD["Q4"]["m2"]) < 1e-9
    assert abs(q4b["v3_f64"]["mean_sum"] - GOLD["Q4"]["m3"]) < 1e-9

    out = {"N": N, "T": T, "dataset": "G1_1e7_1e2_0_0",
           "snap_load_ms": round(load_ms, 1),
           "production": prod, "Q2": q2, "Q4_breakdown": q4b,
           "correctness": "EXACT/PASS", "rss_gb": round(rss(), 2),
           "wall_s": round(time.perf_counter() - t_all, 1)}
    with open(str(OUT), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {OUT} wall={out['wall_s']}s RSS {rss():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
