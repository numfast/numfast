# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native integration: capability on/off x H2O 10M Q1-Q5 (groupby-only).

Backend = src/Drivers/CPU/_lib/native_cpu.py (NEW file, prod untouched).
ON  : NUMFAST_NATIVE_DLL (assert available)
OFF : NUMFAST_NATIVE_DISABLE=1 -> numpy fallback (same outputs required)

Maps each primitive at least once at 10M:
  Q1: pattern_encode(id1) + fused_sum_count             (gold 100 / 29998789)
  Q2: pattern_encode(id1,id2) + pack + fused_sum_count  (gold 10000 / 29998789)
  Q3: pattern_encode(id3) + multi_soa [v1 sum, v3 mean] (gold 100000 + mean3)
  Q3c: dense bincount + carry_build compact (same gold, alt path)
  Q4: multi_soa means [v1,v2,v3] over id4              (gold means m1/m2/m3)
  Q5: multi_soa sums [v1,v2,v3] over id6               (gold s1/s2/s3)
  QS: sorted_run over sorted Q1 codes (gold total; sorted strategy)
Pattern fallback (py loop) is a correctness oracle: verified on a 100k slice
only, excluded from the >=20% gate (gate = bincount-class fallbacks).

Usage (Git Bash):
  /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/research/native_cpu/bench_native_integ_10M.py
"""

import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[3]
CSV = os.environ.get("NF_CSV", "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv")
N = int(os.environ.get("NF_N", "10000000"))
OUT = FORK / "tests" / "research" / "native_cpu" / os.environ.get(
    "NF_OUT", "bench_native_integ_10M.json")

GOLD10 = {
    "Q1": {"ngroups": 100, "total": 29998789},
    "Q2": {"ngroups": 10000, "total": 29998789},
    "Q3": {"ngroups": 100000, "v1": 29998789, "mean3_sum": 4999719.622344427},
    "Q4": {"ngroups": 100, "m1": 299.98798187506526,
           "m2": 799.894179409978, "m3": 4999.766872833688},
    "Q5": {"ngroups": 100000, "s1": 29998789, "s2": 79989360,
           "s3_scaled": 499976651408061},
}
# 100M goldens: resident-100M prod bench (pandas-verified refs).
GOLD100 = {
    "Q1": {"ngroups": 100, "total": 299991302},
    "Q2": {"ngroups": 10000, "total": 299991302},
    "Q3": {"ngroups": 1000000, "v1": 299991302, "mean3_sum": 50001192.355178125},
    "Q4": {"ngroups": 100, "m1": 299.99132111201084,
           "m2": 799.9782349278245, "m3": 5000.104101055998},
    "Q5": {"ngroups": 1000000, "s1": 299991302, "s2": 799978221,
           "s3_scaled": 5000103937771570},
}

import numpy as np
import psutil

PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 1e9


_spec = importlib.util.spec_from_file_location(
    "native_cpu", str(FORK / "src" / "Drivers" / "CPU" / "_lib" / "native_cpu.py"))
nc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nc)


def best_of(fn, reps=3):
    best, out = 1e18, None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t) * 1000)
    return out, best


def run_all(C1, C2, C3):
    GOLD = GOLD100 if N == 100_000_000 else GOLD10
    R = {}
    # Q1 (codes shared; encode timed once in phase E)
    codes, valid, w = C1
    g = int(codes.max()) + 1
    (s, c), ms_a = best_of(lambda: nc.fused_sum_count(codes, V1F, g), 3)
    R["Q1"] = {"agg_ms": ms_a, "total": int(s.sum()),
               "ngroups": int((c > 0).sum()), "width": int(w)}
    assert R["Q1"]["total"] == GOLD["Q1"]["total"] and R["Q1"]["ngroups"] == GOLD["Q1"]["ngroups"], R["Q1"]
    # Q2
    c1, c2 = C1[0], C2[0]
    m2v = int(c2.max()) + 1
    comp, ms_p = best_of(lambda: nc.pack_i32_direct(c1, c2, m2v), 3)
    g2 = int(comp.max()) + 1
    (s, c), ms_a = best_of(lambda: nc.fused_sum_count(comp, V1F, g2), 3)
    R["Q2"] = {"pack_ms": ms_p, "agg_ms": ms_a,
               "total": int(s.sum()), "ngroups": int((c > 0).sum())}
    assert R["Q2"]["total"] == GOLD["Q2"]["total"] and R["Q2"]["ngroups"] == GOLD["Q2"]["ngroups"], R["Q2"]
    # Q3 (direct multi)
    ck = C3[0]
    g3 = int(ck.max()) + 1
    (ss, cc), ms_a = best_of(lambda: nc.multi_sum_count(ck, [V1F, V3], g3), 3)
    mean3 = float((ss[1] / np.maximum(cc, 1))[cc > 0].sum())
    R["Q3"] = {"agg_ms": ms_a, "v1": int(ss[0].sum()),
               "mean3_sum": mean3,
               "mean3_diff": abs(mean3 - GOLD["Q3"]["mean3_sum"])}
    assert R["Q3"]["v1"] == GOLD["Q3"]["v1"], R["Q3"]
    assert R["Q3"]["mean3_diff"] < 1e-6 * abs(GOLD["Q3"]["mean3_sum"]) + 1e-6, R["Q3"]
    # Q3c (dense bincount + carry compact, alt path)
    def _q3c():
        cc_m = np.bincount(ck, minlength=g3).astype(np.int64)
        s1_m = np.bincount(ck, weights=V1F, minlength=g3)
        return nc.carry_build(cc_m, s1_m)
    (uk, cc2, ss2), ms_c = best_of(_q3c, 3)
    R["Q3c"] = {"carry_ms": ms_c, "v1": int(ss2.sum()), "ngroups": int(uk.size)}
    assert R["Q3c"]["v1"] == GOLD["Q3"]["v1"] and R["Q3c"]["ngroups"] == GOLD["Q3"]["ngroups"], R["Q3c"]
    # Q4 (id4 int keys -> dense shift)
    k4 = (ID4 - int(ID4.min())).astype(np.int32)
    g4 = int(k4.max()) + 1
    (ss, cc), ms_a = best_of(lambda: nc.multi_sum_count(k4, [V1F, V2F, V3], g4), 3)
    means = [float((ss[i] / cc).sum()) for i in range(3)]
    R["Q4"] = {"agg_ms": ms_a, "means": means,
               "maxdiff": max(abs(means[0] - GOLD["Q4"]["m1"]),
                              abs(means[1] - GOLD["Q4"]["m2"]),
                              abs(means[2] - GOLD["Q4"]["m3"]))}
    assert R["Q4"]["maxdiff"] < 1e-9, R["Q4"]
    # Q5 (id6 int keys -> dense shift)
    k6 = (ID6 - int(ID6.min())).astype(np.int32)
    g6 = int(k6.max()) + 1
    (ss, cc), ms_a = best_of(lambda: nc.multi_sum_count(k6, [V1F, V2F, V3], g6), 3)
    R["Q5"] = {"agg_ms": ms_a, "s1": int(ss[0].sum()), "s2": int(ss[1].sum()),
               "s3_scaled": int(round(float(ss[2].sum()) * 1e6)),
               "ngroups": int((cc > 0).sum())}
    assert R["Q5"]["s1"] == GOLD["Q5"]["s1"] and R["Q5"]["s2"] == GOLD["Q5"]["s2"], R["Q5"]
    assert R["Q5"]["s3_scaled"] == GOLD["Q5"]["s3_scaled"], R["Q5"]
    assert R["Q5"]["ngroups"] == GOLD["Q5"]["ngroups"], R["Q5"]
    # QS (sorted run over sorted Q1 codes)
    (uk_qs, ss, cc), ms_a = best_of(lambda: nc.sorted_run(SK, SV), 3)
    R["QS"] = {"agg_ms": ms_a, "total": int(ss.sum()), "ngroups": int(ss.size)}
    assert R["QS"]["total"] == GOLD["Q1"]["total"] and R["QS"]["ngroups"] == GOLD["Q1"]["ngroups"], R["QS"]
    return R


def main():
    import pandas as pd

    global D1, O1, D2, O2, D3, O3, ID4, ID6, V1F, V2F, V3, SK, SV
    print("=== native-integ 10M: ON vs OFF ===", flush=True)
    print("backend: %s" % nc.why(), flush=True)
    assert nc.available(), "native backend must load for ON path"
    print("RSS start %.2fGB" % rss(), flush=True)

    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    load_ms = (time.perf_counter() - t0) * 1000
    assert len(df) == N
    id1 = df["id1"].to_numpy()
    id2 = df["id2"].to_numpy()
    id3 = df["id3"].to_numpy()
    ID4 = df["id4"].to_numpy().astype(np.int32)
    ID6 = df["id6"].to_numpy().astype(np.int32)
    V1F = df["v1"].to_numpy().astype(np.float64)
    V2F = df["v2"].to_numpy().astype(np.float64)
    V3 = df["v3"].to_numpy().astype(np.float64)
    SV = df["v1"].to_numpy().astype(np.int64)
    del df
    gc.collect()

    t0 = time.perf_counter()
    D1, O1 = nc.build_pattern_buffers(id1)
    D2, O2 = nc.build_pattern_buffers(id2)
    D3, O3 = nc.build_pattern_buffers(id3)
    prep_ms = (time.perf_counter() - t0) * 1000
    del id1, id2, id3
    gc.collect()
    t0 = time.perf_counter()
    codes0, _, _ = nc.pattern_encode_buffers(D1, O1, "id")
    SK = np.sort(codes0)
    SV = SV[np.argsort(codes0, kind="stable")]
    del codes0
    gc.collect()
    sort_ms = (time.perf_counter() - t0) * 1000
    print("csv_load %.0fms pattern_prep %.0fms sort10M %.0fms RSS %.2fGB"
          % (load_ms, prep_ms, sort_ms, rss()), flush=True)

    # pattern fallback oracle on 100k slice (correctness only)
    os.environ["NUMFAST_NATIVE_DISABLE"] = "1"
    sl = slice(0, 100_000)
    ds = D1[O1[sl.start]:O1[sl.stop]]
    oos = (O1[sl.start:sl.stop + 1] - O1[sl.start]).astype(np.int32)
    co, vo, wo = nc.pattern_encode_buffers(ds, oos, "id")
    del os.environ["NUMFAST_NATIVE_DISABLE"]
    assert nc.available()
    cn, vn, wn = nc.pattern_encode_buffers(ds, oos, "id")
    assert (co == cn).all() and (vo == vn).all() and wo == wn, "pattern oracle DIFF"
    print("pattern oracle 100k: fallback==native EXACT (excluded from gate)", flush=True)

    # phase E: native encodes once (shared codes for ON and OFF agg paths)
    C1, E1 = best_of(lambda: nc.pattern_encode_buffers(D1, O1, "id"), 3)
    C2, E2 = best_of(lambda: nc.pattern_encode_buffers(D2, O2, "id"), 3)
    C3, E3 = best_of(lambda: nc.pattern_encode_buffers(D3, O3, "id"), 3)

    on = run_all(C1, C2, C3)
    on_ms = sum(v.get("pack_ms", 0) + v.get("agg_ms", 0) + v.get("carry_ms", 0)
                for v in on.values())
    os.environ["NUMFAST_NATIVE_DISABLE"] = "1"
    assert not nc.available(), "OFF path must see no backend"
    off = run_all(C1, C2, C3)
    del os.environ["NUMFAST_NATIVE_DISABLE"]
    assert nc.available()
    off_ms = sum(v.get("pack_ms", 0) + v.get("agg_ms", 0) + v.get("carry_ms", 0)
                 for v in off.values())
    # ON vs OFF equality (correctness: absence never breaks results)
    for q in on:
        a, b = dict(on[q]), dict(off[q])
        assert a.keys() == b.keys(), q
        for k in a:
            if "ms" in k:
                continue
            va, vb = a[k], b[k]
            if isinstance(va, float):
                assert abs(va - vb) <= 1e-9 * max(1.0, abs(va)), (q, k, va, vb)
            else:
                assert va == vb, (q, k, va, vb)
    print("ON==OFF on all queries: EXACT", flush=True)

    gate_on = on_ms
    gate_off = off_ms
    win = (gate_off - gate_on) / gate_off * 100
    print("phase E (native pattern_encode 10M): id1=%.0f id2=%.0f id3=%.0f ms"
          % (E1, E2, E3), flush=True)
    for q in ("Q1", "Q2", "Q3", "Q3c", "Q4", "Q5", "QS"):
        o, f = on[q], off[q]
        om = o.get("pack_ms", 0) + o.get("agg_ms", 0) + o.get("carry_ms", 0)
        fm = f.get("pack_ms", 0) + f.get("agg_ms", 0) + f.get("carry_ms", 0)
        print("%-4s ON %8.1fms OFF %8.1fms win %+6.1f%% chk=%s"
              % (q, om, fm, (fm - om) / fm * 100,
                 {k: v for k, v in o.items() if "ms" not in k}), flush=True)
    print("GATE (agg/pack/carry): ON %.0fms OFF %.0fms win %.1f%% %s"
          % (gate_on, gate_off, win, "-> 100M" if win >= 20 else "(<20%: no 100M)"), flush=True)
    json.dump({"n": N, "load_ms": load_ms, "prep_ms": prep_ms, "sort_ms": sort_ms,
               "enc_ms": {"id1": E1, "id2": E2, "id3": E3},
               "rss_gb": rss(), "on": on, "off": off,
               "gate_win_pct": win, "go_100M": bool(win >= 20)},
              open(str(OUT), "w"), indent=1)
    print("wrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
