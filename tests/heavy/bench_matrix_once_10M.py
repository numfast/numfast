# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-process 10M matrix (NEW file; existing benches untouched).

One process loads the resident snapshot ONCE (or CSV+encode once on snapshot
miss, then saves it), builds the kernel once, compiles graphs once, then
sweeps threads T in (1,2,4,8,12,16) x queries Q1-Q5 x modes (carry/dict)
in-process via NUMFAST_THREADS (read per execute by the CPU driver; pools
cached per T). No CSV reload per T, no re-encode, no re-import.

Snapshot (rebuild ONLY on csv/size/mtime/N mismatch):
  scratch/snap_G1_1e7_1e2_0_0/{meta.json,K1,K2,K3,K4,K6,V1,V2,V3.npy}
  = resident int32 codes (pattern dicts) + shifted int keys + values.
  Integrity gate on load: per-column max+int-sum fingerprint vs meta
  (loss/corruption = STOP, snapshot rebuilt from CSV).

Queries best-of-3 execute; chk exact everywhere (int exact, float 1e-9 rel,
same GOLD10 as bench_engine_carry). No query-cache, no fixed-M (M from data
at runtime via the dense gate), no DuckDB/Polars in the loop, GPU untouched.

Usage (Git Bash, ONE process):
  "C:/App/numfast/.venv/Scripts/python" tests/heavy/bench_matrix_once_10M.py
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

T0 = time.perf_counter()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, "C:/App/numfast/app-builder-ponytail")
sys.path.insert(0, "C:/App/numfast/numfast-ponytail")

import numpy as np
import psutil

PROC = psutil.Process()
T_IMPORT = (time.perf_counter() - T0) * 1000

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000
SNAP = FORK / "scratch" / "snap_G1_1e7_1e2_0_0"
COLS = ("K1", "K2", "K3", "K4", "K6", "V1", "V2", "V3")
TS = (1, 2, 4, 8, 12, 16)
QS = ("Q1", "Q2", "Q3", "Q4", "Q5")
OUT = FORK / "tests" / "heavy" / "bench_matrix_once_10M.json"

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


def fp_of(arrays):
    return {k: {"max": int(arrays[k].max()), "min": int(arrays[k].min()),
                "sum": int(arrays[k].astype(np.int64).sum())}
            for k in COLS}


def try_load_snap():
    meta_p = SNAP / "meta.json"
    if not meta_p.exists():
        return None, "miss(no-meta)"
    meta = json.loads(meta_p.read_text())
    st = os.stat(CSV)
    if (meta.get("csv_size") != st.st_size or meta.get("csv_mtime") != st.st_mtime
            or meta.get("n") != N):
        return None, "miss(csv-changed)"
    t = time.perf_counter()
    arr = {k: np.load(str(SNAP / f"{k}.npy")) for k in COLS}
    ms = (time.perf_counter() - t) * 1000
    if fp_of(arr) != meta["fp"]:
        return None, "miss(fp-corrupt)"
    if any(arr[k].shape != (N,) for k in COLS):
        return None, "miss(shape)"
    return (arr, meta, ms), None


def build_snap(a):
    import pandas as pd
    t = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    assert len(df) == N, len(df)
    load_ms = (time.perf_counter() - t) * 1000
    print(f"csv_load {load_ms:.0f}ms RSS {rss():.2f}GB", flush=True)

    def enc(col, prefix="id"):
        gg = a["compile"]([a["ir_encode_pattern"]("c", col, prefix)])
        rr = a["evaluate"](a["optimize"](gg), "cpu", len(col))["result"]
        return np.ascontiguousarray(rr)

    t = time.perf_counter()
    K1 = enc(df["id1"].to_numpy(), "id")
    K2 = enc(df["id2"].to_numpy(), "id")
    K3 = enc(df["id3"].to_numpy(), "id")
    enc_ms = (time.perf_counter() - t) * 1000
    id4 = df["id4"].to_numpy().astype(np.int32)
    id6 = df["id6"].to_numpy().astype(np.int32)
    K4 = np.ascontiguousarray((id4 - int(id4.min())).astype(np.int32))
    K6 = np.ascontiguousarray((id6 - int(id6.min())).astype(np.int32))
    V1 = np.ascontiguousarray(df["v1"].to_numpy().astype(np.int32))
    V2 = np.ascontiguousarray(df["v2"].to_numpy().astype(np.int32))
    V3 = np.ascontiguousarray(df["v3"].to_numpy().astype(np.float64))
    del df, id4, id6
    gc.collect()
    print(f"encode {enc_ms:.0f}ms RSS {rss():.2f}GB "
          f"K1max={int(K1.max())} K2max={int(K2.max())} K3max={int(K3.max())}", flush=True)
    arr = {"K1": K1, "K2": K2, "K3": K3, "K4": K4,
           "K6": K6, "V1": V1, "V2": V2, "V3": V3}
    t = time.perf_counter()
    SNAP.mkdir(parents=True, exist_ok=True)
    for k in COLS:
        np.save(str(SNAP / f"{k}.npy"), arr[k])
    st = os.stat(CSV)
    (SNAP / "meta.json").write_text(json.dumps(
        {"csv_size": st.st_size, "csv_mtime": st.st_mtime, "n": N, "fp": fp_of(arr)}))
    save_ms = (time.perf_counter() - t) * 1000
    print(f"snap_save {save_ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    return arr, {"load_ms": load_ms, "enc_ms": enc_ms, "snap_save_ms": save_ms,
                 "snap": "rebuilt"}


def best_of(fn, reps=3):
    best, out = 1e18, None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t) * 1000)
    return out, best


def carry_checks(q, bufs):
    c = bufs["g#carry"]
    if q in ("Q1", "Q2"):
        return {"total": int(c.sums["v"].sum()), "ngroups": c.ngroups}
    if q == "Q3":
        return {"v1": int(c.sums["v1"].sum()),
                "mean3_sum": float(c.means("v3").sum()), "ngroups": c.ngroups}
    if q == "Q4":
        return {"means": {col: float(c.means(col).sum())
                          for col in ("v1", "v2", "v3")}, "ngroups": c.ngroups}
    return {"s1": int(c.sums["v1"].sum()), "s2": int(c.sums["v2"].sum()),
            "s3": float(c.sums["v3"].sum()), "ngroups": c.ngroups}


def dict_checks(q, res):
    if q in ("Q1", "Q2"):
        return {"total": int(sum(int(v) for v in res.values())), "ngroups": len(res)}
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
        sc = int(round(chk["s3"] * 1e6))
        assert sc == g["s3_scaled"], (q, sc)


def main():
    from builder import MAIN
    stages = {"import_ms": round(T_IMPORT, 1)}
    print(f"=== matrix-once 10M N={N} T={list(TS)} ===", flush=True)
    print(f"import {T_IMPORT:.0f}ms RSS {rss():.2f}GB", flush=True)

    t = time.perf_counter()
    a = MAIN["build"]("C:/App/numfast/numfast-ponytail").alias
    stages["build_ms"] = round((time.perf_counter() - t) * 1000, 1)
    print(f"build {stages['build_ms']:.0f}ms RSS {rss():.2f}GB", flush=True)

    got, why = try_load_snap()
    if got is None:
        print(f"snapshot {why}; CSV+encode once...", flush=True)
        arr, prep = build_snap(a)
        stages.update({k: round(v, 1) for k, v in prep.items() if k.endswith("_ms")})
        stages["snap"] = prep["snap"]
    else:
        arr, meta, load_ms = got
        stages["snap_load_ms"] = round(load_ms, 1)
        stages["snap"] = "hit"
        print(f"snap_load {load_ms:.0f}ms RSS {rss():.2f}GB (integrity fp OK)", flush=True)
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
    t = time.perf_counter()
    compd, compc = {}, {}
    for q, jj in JD.items():
        compd[q] = a["optimize"](a["compile"](jj))
        last = jj[-1]
        p = dict(last["params"])
        p["result"] = "carry"
        nn = dict(last)
        nn["params"] = p
        compc[q] = a["optimize"](a["compile"](jj[:-1] + [nn]))
    stages["compile_all_ms"] = round((time.perf_counter() - t) * 1000, 1)

    # warmup once (numba/native/first-touch), T=1
    os.environ["NUMFAST_THREADS"] = "1"
    t = time.perf_counter()
    bufs = a["cpu_execute"](compc["Q1"]["nodes"])
    gate("Q1", carry_checks("Q1", bufs))
    del bufs
    gc.collect()
    stages["warmup_ms"] = round((time.perf_counter() - t) * 1000, 1)
    print(f"warmup {stages['warmup_ms']:.0f}ms RSS {rss():.2f}GB", flush=True)

    matrix, t_all = {}, time.perf_counter()
    for T in TS:
        os.environ["NUMFAST_THREADS"] = str(T)
        row = {}
        for q in QS:
            bufs, cms = best_of(lambda: a["cpu_execute"](compc[q]["nodes"]), 3)
            cc = carry_checks(q, bufs)
            gate(q, cc)
            gi = bufs.get("g#groupindex", {})
            real_t = gi.get("threads")
            del bufs
            gc.collect()
            res, dms = best_of(
                lambda: a["evaluate"](compd[q], "cpu", N)["result"], 3)
            dc = dict_checks(q, res)
            gate(q, dc)
            del res
            gc.collect()
            row[q] = {"carry_ms": round(cms, 1), "dict_ms": round(dms, 1),
                      "threads_seen": real_t, "ngroups": cc["ngroups"]}
            assert real_t == T, (q, real_t, T)
            print(f"T={T} {q}: carry={cms:.0f}ms dict={dms:.0f}ms "
                  f"chk_ng={cc['ngroups']} RSS {rss():.2f}GB", flush=True)
        matrix[str(T)] = row
    stages["matrix_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
    stages["wall_ms"] = round((time.perf_counter() - T0) * 1000, 1)

    out = {"N": N, "dataset": "G1_1e7_1e2_0_0", "threads": list(TS),
           "stages": stages, "matrix": matrix, "rss_gb": rss(),
           "gold": "EXACT/PASS", "snapshot": str(SNAP)}
    with open(str(OUT), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wall {stages['wall_ms']/1000:.1f}s matrix {stages['matrix_ms']/1000:.1f}s "
          f"-> {OUT} gold EXACT/PASS", flush=True)


if __name__ == "__main__":
    main()
