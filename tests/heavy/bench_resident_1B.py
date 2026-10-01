# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resident typed columns on H2O 1B (heavy, B-only, CPU production path).

Same methodology as bench_resident_100M.py (CSV -> extract -> resident_prepare
ONCE -> typed resident -> IR -> Planner -> Runtime -> CPU Driver -> GroupBy),
but N=1B (G1_1e9_1e2_0_0.csv, 51GB). No GPU. No Planner/kernel/calibration
change. No ClickBench.

Why sharded ingest: single pd.read_csv of 51GB extrapolates to ~130GB+ transient
(10M slice = 1.36GB RSS) > avail RAM; chunked/Sharded ingest keeps peak ~40GB
(resident 36GB + one 100M shard transient). Pattern encode is row-local
(prefix+int parse), so per-shard resident_prepare == single-shot semantics;
numeric resident stays zero-copy views per shard, copied once into prealloc.
Q graphs are byte-identical to 100M bench (ir_series/ir_groupby[_multi],
ir_pack_keys runtime). Q3/Q5 groups are data-dependent -> dynamic np.unique
(like 100M), NOT hardcoded: 1B file has ~6.3M distinct id3/id6 per 10M slice
(vs 100k at 10M file), full-1B groups measured at runtime.

Frozen Runtime caps a single graph at 536870911 rows (chunk_plan raises:
chunked execution not implemented; Planner/drivers untouched), so Q run
CHUNKED 10x100M at bench level: identical public graph per shard
(n=shard_N), exact host combine (int sums/counts, tol-checked f64 means).
Single-graph 1B Q is blocked by the frozen limit -> recorded, not bypassed.

Usage (Git Bash, fork-first, H2O python explicit, strictly sequential):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/competitions/H2O/python310/python.exe tests/heavy/bench_resident_1B.py ingest-shard <idx> <nshard>
  .../python.exe tests/heavy/bench_resident_1B.py merge
  .../python.exe tests/heavy/bench_resident_1B.py q1|q2|q3|q4|q5
Shard CSVs: split -l must have produced scratch/resident_1B/csv_shard_XX.csv
  (see split_csv_1B.py step, header preserved per shard).
Resident shards: scratch/resident_1B/shard_<idx>.npz (k1,k2,k3,id4,id6,v1,v2,v3)
Merged: scratch/resident_1B/merged/*.npy + refs.json
Per-Q JSON: tests/heavy/bench_resident_1B_<q>.json ; summary: bench_resident_1B.json
"""

import gc
import json
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CSV = "C:/App/competitions/H2O/data/G1_1e9_1e2_0_0.csv"
N = 1_000_000_000
NSHARD = 10
SHARD_ROWS = N // NSHARD
SCR = FORK / "scratch" / "resident_1B"
CSVD = SCR / "csv_shards"
SHD = SCR / "shards"
MRG = SCR / "merged"

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


def cmd_ingest_shard(idx):
    import pandas as pd
    import pyarrow as pa

    print(f"=== 1B ingest shard {idx}/{NSHARD} ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    kernel = build_kernel()
    a = kernel.alias
    f = CSVD / f"csv_shard_{idx:02d}.csv"
    assert f.exists(), f"missing {f} (run split step first)"
    t0 = time.perf_counter()
    # headerless body shards (split of tail -n +2): explicit names, no header row
    df = pd.read_csv(str(f), names=["id1", "id2", "id3", "id4", "id5", "id6", "v1", "v2", "v3"],
                     header=None, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    load_ms = (time.perf_counter() - t0) * 1000
    n = len(df)
    print(f"csv_load shard {n} rows {load_ms:.0f}ms RSS {rss():.2f}GB", flush=True)

    t0 = time.perf_counter()
    s_id1 = pa.array(df["id1"].to_numpy(), type=pa.string())
    s_id2 = pa.array(df["id2"].to_numpy(), type=pa.string())
    s_id3 = pa.array(df["id3"].to_numpy(), type=pa.string())
    id4 = df["id4"].to_numpy().astype(np.int32)
    id6 = df["id6"].to_numpy().astype(np.int32)
    v1 = df["v1"].to_numpy().astype(np.int32)
    v2 = df["v2"].to_numpy().astype(np.int32)
    v3 = df["v3"].to_numpy().astype(np.float64)
    extract_ms = (time.perf_counter() - t0) * 1000
    print(f"extract {extract_ms:.0f}ms RSS {rss():.2f}GB", flush=True)
    ref = {"n": n, "v1": int(v1.astype(np.int64).sum()),
           "v2": int(v2.astype(np.int64).sum()), "v3": float(v3.astype(np.float64).sum())}
    print(f"refs v1={ref['v1']} v2={ref['v2']} v3={ref['v3']!r}", flush=True)

    t0 = time.perf_counter()
    r1 = a["resident_prepare"]({"id1": {"values": s_id1, "prefix": "id"}})
    ms1 = (time.perf_counter() - t0) * 1000
    del s_id1
    gc.collect()
    t = time.perf_counter()
    r2 = a["resident_prepare"]({"id2": {"values": s_id2, "prefix": "id"}})
    ms2 = (time.perf_counter() - t) * 1000
    del s_id2
    gc.collect()
    t = time.perf_counter()
    r3 = a["resident_prepare"]({"id3": {"values": s_id3, "prefix": "id"}})
    ms3 = (time.perf_counter() - t) * 1000
    del s_id3
    gc.collect()
    t = time.perf_counter()
    rn = a["resident_prepare"]({
        "id4": {"values": id4, "dtype": "int32"}, "id6": {"values": id6, "dtype": "int32"},
        "v1": {"values": v1, "dtype": "int32"}, "v2": {"values": v2, "dtype": "int32"},
        "v3": {"values": v3, "dtype": "float64"}})
    msn = (time.perf_counter() - t) * 1000
    prep_ms = ms1 + ms2 + ms3 + msn
    assert np.shares_memory(rn["v1"]["codes"], v1)
    print(f"prep id1={ms1:.0f} id2={ms2:.0f} id3={ms3:.0f} num={msn:.0f} total={prep_ms:.0f}ms "
          f"RSS {rss():.2f}GB", flush=True)
    print("zero-copy numeric resident: OK", flush=True)

    SHD.mkdir(parents=True, exist_ok=True)
    out = SHD / f"shard_{idx:02d}.npz"
    t = time.perf_counter()
    np.savez(
        str(out),
        k1=np.asarray(r1["id1"]["codes"]), k2=np.asarray(r2["id2"]["codes"]),
        k3=np.asarray(r3["id3"]["codes"]), id4=np.asarray(rn["id4"]["codes"]),
        id6=np.asarray(rn["id6"]["codes"]), v1=np.asarray(rn["v1"]["codes"]),
        v2=np.asarray(rn["v2"]["codes"]), v3=np.asarray(rn["v3"]["codes"]))
    save_ms = (time.perf_counter() - t) * 1000
    meta = {"idx": idx, "n": n, "csv_load_ms": load_ms, "extract_ms": extract_ms,
            "prep_ms": prep_ms, "prep_detail_ms": {"id1": ms1, "id2": ms2, "id3": ms3, "num": msn},
            "save_ms": save_ms, "refs": ref, "rss_gb": rss()}
    with open(SHD / f"shard_{idx:02d}.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"saved {out} {save_ms:.0f}ms RSS {rss():.2f}GB", flush=True)


def _load_merged():
    assert (MRG / "refs.json").exists(), "missing merged (run merge first)"
    d = {}
    for k in ("k1", "k2", "k3", "id4", "id6", "v1", "v2"):
        d[k] = np.load(str(MRG / f"{k}.npy"), mmap_mode="r")
    d["v3"] = np.load(str(MRG / "v3.npy"), mmap_mode="r")
    with open(MRG / "refs.json") as fh:
        refs = json.load(fh)
    return d, refs


def cmd_merge():
    SHD.mkdir(parents=True, exist_ok=True)
    MRG.mkdir(parents=True, exist_ok=True)
    metas, tot = [], 0
    for i in range(NSHARD):
        m = SHD / f"shard_{i:02d}.json"
        assert m.exists(), f"missing shard {i}"
        with open(m) as fh:
            metas.append(json.load(fh))
        tot += metas[-1]["n"]
    assert tot == N, (tot, N)
    refs = {"n": N, "v1": sum(m["refs"]["v1"] for m in metas),
            "v2": sum(m["refs"]["v2"] for m in metas),
            "v3": sum(m["refs"]["v3"] for m in metas)}
    print(f"merge {NSHARD} shards N={tot} refs v1={refs['v1']} v2={refs['v2']} v3={refs['v3']!r} "
          f"RSS {rss():.2f}GB", flush=True)
    # concat each column shard-by-shard (peak: one full col + one shard col)
    t0 = time.perf_counter()
    for k in ("k1", "k2", "k3", "id4", "id6", "v1", "v2", "v3"):
        dt = np.float64 if k == "v3" else np.int32
        out = np.empty(N, dtype=dt)
        a = 0
        for i in range(NSHARD):
            z = np.load(str(SHD / f"shard_{i:02d}.npz"))[k]
            b = a + z.size
            out[a:b] = z
            a = b
            del z
            gc.collect()
        np.save(str(MRG / f"{k}.npy"), out)
        del out
        gc.collect()
        print(f"  merged {k} RSS {rss():.2f}GB", flush=True)
    with open(MRG / "refs.json", "w") as fh:
        json.dump({"refs": refs, "shards": metas,
                   "merge_ms": (time.perf_counter() - t0) * 1000}, fh, indent=1)
    print(f"merge done {(time.perf_counter()-t0)*1000:.0f}ms RSS {rss():.2f}GB", flush=True)


def _shard_ref(i):
    with open(SHD / f"shard_{i:02d}.json") as fh:
        return json.load(fh)["refs"]


def cmd_q3shard(idx):
    """Q3 phase A: one shard main+count graphs -> columnar part file."""
    print(f"=== 1B q3 shard {idx} ===", flush=True)
    kernel = build_kernel()
    a = kernel.alias
    z = np.load(str(SHD / f"shard_{idx:02d}.npz"))
    n = int(z["k1"].size)
    sr = _shard_ref(idx)
    k3, v1c, v3c = np.asarray(z["k3"]), np.asarray(z["v1"]), np.asarray(z["v3"])
    jobs = [a["ir_series"]("k", k3), a["ir_series"]("v1", v1c),
            a["ir_series"]("v3", v3c, "float64"),
            a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                  {"v1": ("sum",), "v3": ("mean",)})]
    r, st = run_graph(a, jobs, n)
    s1 = int(sum(c["v1"]["sum"] for c in r.values()))
    assert s1 == sr["v1"], (idx, s1, sr["v1"])
    rc, stc = run_graph(a, [a["ir_series"]("k", k3),
                            a["ir_series"]("v", v1c),
                            a["ir_groupby"]("g", "v", "k", "count")], n)
    assert int(sum(int(v) for v in rc.values())) == n, idx
    assert len(rc) == len(r), (idx, len(rc), len(r))
    t = time.perf_counter()
    m = len(r)
    keys = np.fromiter(r.keys(), dtype=np.int32, count=m)
    a1 = np.fromiter((c["v1"]["sum"] for c in r.values()), dtype=np.int64, count=m)
    mu = np.fromiter((c["v3"]["mean"] for c in r.values()), dtype=np.float64, count=m)
    cc = np.fromiter((rc[k] for k in r.keys()), dtype=np.int64, count=m)
    conv_ms = (time.perf_counter() - t) * 1000
    out = SCR / f"q3part_{idx:02d}.npz"
    np.savez(str(out), k=keys, s1=a1, s3=mu * cc, c=cc)
    meta = {"shard": idx, "n": n, "cold_ms": st["total"], "exec_ms": st["execute"],
            "count_ms": stc["total"], "conv_ms": conv_ms, "ngroups": m, "rss_gb": rss()}
    if idx == 0:
        r2, st2 = run_graph(a, jobs, n)
        m3c = float(sum(c["v3"]["mean"] for c in r2.values()))
        m3w = float(sum(c["v3"]["mean"] for c in r.values()))
        assert abs(m3c - m3w) <= max(1e-12, 1e-12 * abs(m3w))
        meta["warm_ms"] = st2["total"]
        del r2
    with open(SCR / f"q3part_{idx:02d}.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"shard {idx}: cold={st['total']:.0f} count={stc['total']:.0f} "
          f"conv={conv_ms:.0f}ms groups={m} RSS {rss():.2f}GB", flush=True)


def cmd_q3combine():
    """Q3 phase B: concat 10 parts -> global unique/bincount -> validate."""
    print("=== 1B q3 combine ===", flush=True)
    with open(MRG / "refs.json") as fh:
        refs = json.load(fh)["refs"]
    per = []
    K, S1, S3, CC = [], [], [], []
    for i in range(NSHARD):
        with open(SCR / f"q3part_{i:02d}.json") as fh:
            per.append(json.load(fh))
        p = np.load(str(SCR / f"q3part_{i:02d}.npz"))
        K.append(np.asarray(p["k"]))
        S1.append(np.asarray(p["s1"]))
        S3.append(np.asarray(p["s3"]))
        CC.append(np.asarray(p["c"]))
        del p
        gc.collect()
    t = time.perf_counter()
    AK = np.concatenate(K)
    del K
    gc.collect()
    AS1 = np.concatenate(S1)
    del S1
    gc.collect()
    AS3 = np.concatenate(S3)
    del S3
    gc.collect()
    ACC = np.concatenate(CC)
    del CC
    gc.collect()
    uk, inv = np.unique(AK, return_inverse=True)
    del AK
    gc.collect()
    gs1 = np.bincount(inv, weights=AS1.astype(np.float64)).astype(np.int64)
    del AS1
    gc.collect()
    gs3 = np.bincount(inv, weights=AS3)
    del AS3
    gcc = np.bincount(inv, weights=ACC.astype(np.float64)).astype(np.int64)
    del ACC, inv
    gc.collect()
    chk_v1 = int(gs1.sum())
    assert chk_v1 == refs["v1"], (chk_v1, refs["v1"])
    assert int(gcc.sum()) == N
    v3x = float(gs3.sum())
    assert abs(v3x - refs["v3"]) <= max(1e-9, 1e-12 * abs(refs["v3"])) * max(1.0, abs(refs["v3"]) / 1e6), (v3x, refs["v3"])
    mean3 = float((gs3 / gcc).sum())
    combine_ms = (time.perf_counter() - t) * 1000
    out = {"mode": "chunked-10x100M", "cold_ms": sum(p["cold_ms"] for p in per),
           "exec_ms": sum(p["exec_ms"] for p in per), "warm_ms_shard0": per[0].get("warm_ms"),
           "chk_v1": chk_v1, "mean3_sum": mean3, "ngroups": int(uk.size),
           "v3_cross_diff": abs(v3x - refs["v3"]), "combine_ms": combine_ms,
           "rss_gb": rss(), "per_shard": per}
    with open(FORK / "tests" / "heavy" / "bench_resident_1B_q3.json", "w") as fh:
        json.dump(json.loads(json.dumps(out, default=float)), fh, indent=1)
    print(f"q3 groups={uk.size} v1={chk_v1} mean3={mean3!r} combine={combine_ms:.0f}ms "
          f"RSS {rss():.2f}GB JSON written", flush=True)


def cmd_q5shard(idx):
    """Q5 phase A: one shard multi-sum graph -> columnar part file."""
    print(f"=== 1B q5 shard {idx} ===", flush=True)
    kernel = build_kernel()
    a = kernel.alias
    z = np.load(str(SHD / f"shard_{idx:02d}.npz"))
    n = int(z["k1"].size)
    sr = _shard_ref(idx)
    id6 = np.asarray(z["id6"])
    v1c, v2c, v3c = np.asarray(z["v1"]), np.asarray(z["v2"]), np.asarray(z["v3"])
    jobs = [a["ir_series"]("k", id6), a["ir_series"]("v1", v1c),
            a["ir_series"]("v2", v2c), a["ir_series"]("v3", v3c, "float64"),
            a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                  {"v1": ("sum",), "v2": ("sum",), "v3": ("sum",)})]
    r, st = run_graph(a, jobs, n)
    s1 = int(sum(c["v1"]["sum"] for c in r.values()))
    s2 = int(sum(c["v2"]["sum"] for c in r.values()))
    assert (s1, s2) == (sr["v1"], sr["v2"]), (idx, s1, s2)
    t = time.perf_counter()
    m = len(r)
    keys = np.fromiter(r.keys(), dtype=np.int32, count=m)
    aa = np.fromiter((c["v1"]["sum"] for c in r.values()), dtype=np.int64, count=m)
    bb = np.fromiter((c["v2"]["sum"] for c in r.values()), dtype=np.int64, count=m)
    cc = np.fromiter((c["v3"]["sum"] for c in r.values()), dtype=np.float64, count=m)
    conv_ms = (time.perf_counter() - t) * 1000
    np.savez(str(SCR / f"q5part_{idx:02d}.npz"), k=keys, s1=aa, s2=bb, s3=cc)
    meta = {"shard": idx, "n": n, "cold_ms": st["total"], "exec_ms": st["execute"],
            "conv_ms": conv_ms, "ngroups": m, "rss_gb": rss()}
    if idx == 0:
        r2, st2 = run_graph(a, jobs, n)
        s3c = float(sum(c["v3"]["sum"] for c in r.values()))
        s3w = float(sum(c["v3"]["sum"] for c in r2.values()))
        assert int(round(s3c * 1e6)) == int(round(s3w * 1e6))
        meta["warm_ms"] = st2["total"]
        del r2
    with open(SCR / f"q5part_{idx:02d}.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"shard {idx}: cold={st['total']:.0f} conv={conv_ms:.0f}ms groups={m} "
          f"RSS {rss():.2f}GB", flush=True)


def cmd_q5combine():
    """Q5 phase B: concat 10 parts -> global unique/bincount -> validate."""
    print("=== 1B q5 combine ===", flush=True)
    with open(MRG / "refs.json") as fh:
        refs = json.load(fh)["refs"]
    per = []
    K, S1, S2, S3 = [], [], [], []
    for i in range(NSHARD):
        with open(SCR / f"q5part_{i:02d}.json") as fh:
            per.append(json.load(fh))
        p = np.load(str(SCR / f"q5part_{i:02d}.npz"))
        K.append(np.asarray(p["k"]))
        S1.append(np.asarray(p["s1"]))
        S2.append(np.asarray(p["s2"]))
        S3.append(np.asarray(p["s3"]))
        del p
        gc.collect()
    t = time.perf_counter()
    AK = np.concatenate(K)
    del K
    gc.collect()
    A1 = np.concatenate(S1)
    del S1
    gc.collect()
    A2 = np.concatenate(S2)
    del S2
    gc.collect()
    A3 = np.concatenate(S3)
    del S3
    gc.collect()
    uk, inv = np.unique(AK, return_inverse=True)
    del AK
    gc.collect()
    gs1 = np.bincount(inv, weights=A1.astype(np.float64)).astype(np.int64)
    del A1
    gc.collect()
    gs2 = np.bincount(inv, weights=A2.astype(np.float64)).astype(np.int64)
    del A2
    gs3 = np.bincount(inv, weights=A3)
    del A3, inv
    gc.collect()
    s1, s2, s3 = int(gs1.sum()), int(gs2.sum()), float(gs3.sum())
    assert (s1, s2) == (refs["v1"], refs["v2"]), (s1, s2)
    assert abs(s3 - refs["v3"]) <= max(1e-9, 1e-12 * abs(refs["v3"])) * max(1.0, abs(refs["v3"]) / 1e6), (s3, refs["v3"])
    sc = int(round(s3 * 1e6))
    combine_ms = (time.perf_counter() - t) * 1000
    out = {"mode": "chunked-10x100M", "cold_ms": sum(p["cold_ms"] for p in per),
           "exec_ms": sum(p["exec_ms"] for p in per), "warm_ms_shard0": per[0].get("warm_ms"),
           "s1": s1, "s2": s2, "s3": s3, "s3_scaled": sc, "ngroups": int(uk.size),
           "s3_vs_colsum_diff": abs(s3 - refs["v3"]), "combine_ms": combine_ms,
           "rss_gb": rss(), "per_shard": per}
    with open(FORK / "tests" / "heavy" / "bench_resident_1B_q5.json", "w") as fh:
        json.dump(json.loads(json.dumps(out, default=float)), fh, indent=1)
    print(f"q5 groups={uk.size} s1={s1} s2={s2} s3_scaled={sc} combine={combine_ms:.0f}ms "
          f"RSS {rss():.2f}GB JSON written", flush=True)



def cmd_q(q):
    """Chunked CPU Q over 10x100M resident shards (frozen Runtime caps single
    graph at 536870911 rows with chunked execution unimplemented; Planner and
    drivers untouched). Every row flows through the byte-identical public
    graphs as bench_resident_100M.py (n=shard_N each); only the final
    cross-shard reduction is host-side (exact int add for sums/counts,
    tolerance-checked f64 for means). Codes are raw global ints (pattern
    parse, no per-shard factorization), so keys are directly comparable."""
    print(f"=== 1B {q} (chunked 10x100M resident, CPU public path) ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    kernel = build_kernel()
    a = kernel.alias
    with open(MRG / "refs.json") as fh:
        refs = json.load(fh)["refs"]
    print(f"global refs v1={refs['v1']} v2={refs['v2']} v3={refs['v3']!r}", flush=True)

    per, t_combine0 = [], None

    def shard_n(z):
        return int(z["k1"].size)

    if q in ("q1", "q2", "q4"):
        acc = {}
        for i in range(NSHARD):
            z = np.load(str(SHD / f"shard_{i:02d}.npz"))
            n = shard_n(z)
            sr = _shard_ref(i)
            if q == "q1":
                jobs = [a["ir_series"]("k", np.asarray(z["k1"])),
                        a["ir_series"]("v", np.asarray(z["v1"])),
                        a["ir_groupby"]("g", "v", "k", "sum")]
                exp_g, exp_tot = 100, sr["v1"]
            elif q == "q2":
                jobs = [a["ir_series"]("c1", np.asarray(z["k1"])),
                        a["ir_series"]("c2", np.asarray(z["k2"])),
                        a["ir_series"]("v", np.asarray(z["v1"])),
                        a["ir_pack_keys"]("k", "c1", "c2"),
                        a["ir_groupby"]("g", "v", "k", "sum")]
                exp_g, exp_tot = 10000, sr["v1"]
            else:
                jobs = [a["ir_series"]("k", np.asarray(z["id4"])),
                        a["ir_series"]("v1", np.asarray(z["v1"])),
                        a["ir_series"]("v2", np.asarray(z["v2"])),
                        a["ir_series"]("v3", np.asarray(z["v3"]), "float64"),
                        a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                              {"v1": ("mean",), "v2": ("mean",),
                                               "v3": ("mean",)})]
                exp_g, exp_tot = 100, None
            r, st = run_graph(a, jobs, n)
            t = time.perf_counter()
            if q in ("q1", "q2"):
                assert len(r) == exp_g, (i, len(r))
                tot = 0
                for kk, vv in r.items():
                    acc[kk] = acc.get(kk, 0) + int(vv)
                    tot += int(vv)
                assert tot == exp_tot, (i, tot, exp_tot)
            else:
                assert len(r) == exp_g, (i, len(r))
                for kk, cell in r.items():
                    dst = acc.get(kk)
                    if dst is None:
                        acc[kk] = {"v1": [float(cell["v1"]["mean"])],
                                   "v2": [float(cell["v2"]["mean"])],
                                   "v3": [float(cell["v3"]["mean"])]}
                    else:
                        dst["v1"].append(float(cell["v1"]["mean"]))
                        dst["v2"].append(float(cell["v2"]["mean"]))
                        dst["v3"].append(float(cell["v3"]["mean"]))
            merge_ms = (time.perf_counter() - t) * 1000
            per.append({"shard": i, "n": n, "cold_ms": st["total"],
                        "exec_ms": st["execute"], "merge_ms": merge_ms})
            if i == 0:
                r2, st2 = run_graph(a, jobs, n)
                warm_ms = st2["total"]
                if q in ("q1", "q2"):
                    assert int(sum(int(v) for v in r2.values())) == exp_tot
                del r2
            print(f"  shard {i}: cold={st['total']:.0f} (exec {st['execute']:.0f}) "
                  f"merge={merge_ms:.1f}ms RSS {rss():.2f}GB", flush=True)
            del z, r
            gc.collect()
        t = time.perf_counter()
        if q in ("q1", "q2"):
            tot = int(sum(acc.values()))
            assert len(acc) == exp_g and tot == refs["v1"], (len(acc), tot)
            out = {"mode": "chunked-10x100M", "cold_ms": sum(p["cold_ms"] for p in per),
                   "exec_ms": sum(p["exec_ms"] for p in per), "warm_ms_shard0": warm_ms,
                   "chk": tot, "ngroups": len(acc), "rss_gb": rss(), "per_shard": per}
        else:
            # Q4 global means need counts: exact per-shard counts via
            # single-col count graphs (dict path, proven), then weight.
            ci = {}
            for i in range(NSHARD):
                z = np.load(str(SHD / f"shard_{i:02d}.npz"))
                n = shard_n(z)
                rc, stc = run_graph(a, [a["ir_series"]("k", np.asarray(z["id4"])),
                                        a["ir_series"]("v", np.asarray(z["v1"])),
                                        a["ir_groupby"]("g", "v", "k", "count")], n)
                assert int(sum(int(v) for v in rc.values())) == n
                per[i]["count_ms"] = stc["total"]
                ci[i] = {k: int(v) for k, v in rc.items()}
                del z, rc
                gc.collect()
            gmeans = {}
            for kk in acc:
                num = {"v1": 0.0, "v2": 0.0, "v3": 0.0}
                den = 0
                for i in range(NSHARD):
                    c = int(ci[i][kk])
                    den += c
                    num["v1"] += acc[kk]["v1"][i] * c
                    num["v2"] += acc[kk]["v2"][i] * c
                    num["v3"] += acc[kk]["v3"][i] * c
                assert den == sum(int(ci[i][kk]) for i in range(NSHARD))
                gmeans[kk] = {c: num[c] / den for c in num}
            sums = {c: float(sum(gmeans[k][c] for k in gmeans)) for c in ("v1", "v2", "v3")}
            assert len(gmeans) == 100
            out = {"mode": "chunked-10x100M", "cold_ms": sum(p["cold_ms"] for p in per),
                   "warm_ms_shard0": warm_ms, "sums": sums, "ngroups": 100,
                   "combine_ms": (time.perf_counter() - t) * 1000,
                   "per_shard": per, "rss_gb": rss()}
        print(f"{q} chunks cold_total={sum(p['cold_ms'] for p in per):.0f}ms "
              f"RSS {rss():.2f}GB chk={out.get('chk', out.get('sums'))}", flush=True)
    elif q == "q3":
        K, S1, S3, CC = [], [], [], []
        for i in range(NSHARD):
            z = np.load(str(SHD / f"shard_{i:02d}.npz"))
            n = shard_n(z)
            sr = _shard_ref(i)
            k3, v1c, v3c = np.asarray(z["k3"]), np.asarray(z["v1"]), np.asarray(z["v3"])
            jobs = [a["ir_series"]("k", k3), a["ir_series"]("v1", v1c),
                    a["ir_series"]("v3", v3c, "float64"),
                    a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                          {"v1": ("sum",), "v3": ("mean",)})]
            r, st = run_graph(a, jobs, n)
            s1 = int(sum(c["v1"]["sum"] for c in r.values()))
            assert s1 == sr["v1"], (i, s1, sr["v1"])
            rc, stc = run_graph(a, [a["ir_series"]("k", k3),
                                    a["ir_series"]("v", v1c),
                                    a["ir_groupby"]("g", "v", "k", "count")], n)
            assert int(sum(int(v) for v in rc.values())) == n, i
            assert len(rc) == len(r), (i, len(rc), len(r))
            t = time.perf_counter()
            m = len(r)
            keys = np.fromiter(r.keys(), dtype=np.int32, count=m)
            a1 = np.fromiter((c["v1"]["sum"] for c in r.values()), dtype=np.int64, count=m)
            mu = np.fromiter((c["v3"]["mean"] for c in r.values()), dtype=np.float64, count=m)
            cc = np.fromiter((rc[k] for k in r.keys()), dtype=np.int64, count=m)
            K.append(keys)
            S1.append(a1)
            S3.append(mu * cc)
            CC.append(cc)
            conv_ms = (time.perf_counter() - t) * 1000
            per.append({"shard": i, "n": n, "cold_ms": st["total"], "exec_ms": st["execute"],
                        "count_ms": stc["total"], "conv_ms": conv_ms, "ngroups": m})
            if i == 0:
                r2, st2 = run_graph(a, jobs, n)
                m3c = float(sum(c["v3"]["mean"] for c in r2.values()))
                m3w = float(sum(c["v3"]["mean"] for c in r.values()))
                assert abs(m3c - m3w) <= max(1e-12, 1e-12 * abs(m3w))
                warm_ms = st2["total"]
                del r2
            print(f"  shard {i}: cold={st['total']:.0f} (exec {st['execute']:.0f}) "
                  f"count={stc['total']:.0f} conv={conv_ms:.0f}ms groups={m} "
                  f"RSS {rss():.2f}GB", flush=True)
            del z, r, rc, keys, a1, mu, cc
            gc.collect()
        t = time.perf_counter()
        AK = np.concatenate(K)
        del K
        gc.collect()
        AS1 = np.concatenate(S1)
        del S1
        gc.collect()
        AS3 = np.concatenate(S3)
        del S3
        gc.collect()
        ACC = np.concatenate(CC)
        del CC
        gc.collect()
        uk, inv = np.unique(AK, return_inverse=True)
        del AK
        gc.collect()
        gs1 = np.bincount(inv, weights=AS1.astype(np.float64)).astype(np.int64)
        del AS1
        gc.collect()
        gs3 = np.bincount(inv, weights=AS3)
        del AS3
        gcc = np.bincount(inv, weights=ACC.astype(np.float64)).astype(np.int64)
        del ACC, inv
        gc.collect()
        chk_v1 = int(gs1.sum())
        assert chk_v1 == refs["v1"], (chk_v1, refs["v1"])
        assert int(gcc.sum()) == N
        assert int(uk.size) == int(gs1.size)
        v3x = float(gs3.sum())
        assert abs(v3x - refs["v3"]) <= max(1e-9, 1e-12 * abs(refs["v3"])) * max(1.0, abs(refs["v3"]) / 1e6), (v3x, refs["v3"])
        gmean = gs3 / gcc
        mean3 = float(gmean.sum())
        combine_ms = (time.perf_counter() - t) * 1000
        out = {"mode": "chunked-10x100M", "cold_ms": sum(p["cold_ms"] for p in per),
               "exec_ms": sum(p["exec_ms"] for p in per), "warm_ms_shard0": warm_ms,
               "chk_v1": chk_v1, "mean3_sum": mean3, "ngroups": int(uk.size),
               "v3_cross_diff": abs(v3x - refs["v3"]), "combine_ms": combine_ms,
               "rss_gb": rss(), "per_shard": per}
        print(f"q3 chunks cold_total={out['cold_ms']:.0f} combine={combine_ms:.0f}ms "
              f"groups={uk.size} v1={chk_v1} mean3={mean3!r} RSS {rss():.2f}GB", flush=True)
    elif q == "q5":
        K, S1, S2, S3 = [], [], [], []
        for i in range(NSHARD):
            z = np.load(str(SHD / f"shard_{i:02d}.npz"))
            n = shard_n(z)
            sr = _shard_ref(i)
            id6 = np.asarray(z["id6"])
            v1c, v2c, v3c = np.asarray(z["v1"]), np.asarray(z["v2"]), np.asarray(z["v3"])
            jobs = [a["ir_series"]("k", id6), a["ir_series"]("v1", v1c),
                    a["ir_series"]("v2", v2c), a["ir_series"]("v3", v3c, "float64"),
                    a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                          {"v1": ("sum",), "v2": ("sum",), "v3": ("sum",)})]
            r, st = run_graph(a, jobs, n)
            s1 = int(sum(c["v1"]["sum"] for c in r.values()))
            s2 = int(sum(c["v2"]["sum"] for c in r.values()))
            assert (s1, s2) == (sr["v1"], sr["v2"]), (i, s1, s2)
            t = time.perf_counter()
            m = len(r)
            keys = np.fromiter(r.keys(), dtype=np.int32, count=m)
            aa = np.fromiter((c["v1"]["sum"] for c in r.values()), dtype=np.int64, count=m)
            bb = np.fromiter((c["v2"]["sum"] for c in r.values()), dtype=np.int64, count=m)
            cc = np.fromiter((c["v3"]["sum"] for c in r.values()), dtype=np.float64, count=m)
            K.append(keys)
            S1.append(aa)
            S2.append(bb)
            S3.append(cc)
            conv_ms = (time.perf_counter() - t) * 1000
            per.append({"shard": i, "n": n, "cold_ms": st["total"], "exec_ms": st["execute"],
                        "conv_ms": conv_ms, "ngroups": m})
            if i == 0:
                r2, st2 = run_graph(a, jobs, n)
                s3c = float(sum(c["v3"]["sum"] for c in r.values()))
                s3w = float(sum(c["v3"]["sum"] for c in r2.values()))
                assert int(round(s3c * 1e6)) == int(round(s3w * 1e6))
                warm_ms = st2["total"]
                del r2
            print(f"  shard {i}: cold={st['total']:.0f} (exec {st['execute']:.0f}) "
                  f"conv={conv_ms:.0f}ms groups={m} RSS {rss():.2f}GB", flush=True)
            del z, r, keys, aa, bb, cc
            gc.collect()
        t = time.perf_counter()
        AK = np.concatenate(K)
        del K
        gc.collect()
        A1 = np.concatenate(S1)
        del S1
        gc.collect()
        A2 = np.concatenate(S2)
        del S2
        gc.collect()
        A3 = np.concatenate(S3)
        del S3
        gc.collect()
        uk, inv = np.unique(AK, return_inverse=True)
        del AK
        gc.collect()
        gs1 = np.bincount(inv, weights=A1.astype(np.float64)).astype(np.int64)
        del A1
        gc.collect()
        gs2 = np.bincount(inv, weights=A2.astype(np.float64)).astype(np.int64)
        del A2
        gs3 = np.bincount(inv, weights=A3)
        del A3, inv
        gc.collect()
        s1, s2, s3 = int(gs1.sum()), int(gs2.sum()), float(gs3.sum())
        assert (s1, s2) == (refs["v1"], refs["v2"]), (s1, s2)
        assert abs(s3 - refs["v3"]) <= max(1e-9, 1e-12 * abs(refs["v3"])) * max(1.0, abs(refs["v3"]) / 1e6), (s3, refs["v3"])
        sc = int(round(s3 * 1e6))
        combine_ms = (time.perf_counter() - t) * 1000
        out = {"mode": "chunked-10x100M", "cold_ms": sum(p["cold_ms"] for p in per),
               "exec_ms": sum(p["exec_ms"] for p in per), "warm_ms_shard0": warm_ms,
               "s1": s1, "s2": s2, "s3": s3, "s3_scaled": sc, "ngroups": int(uk.size),
               "s3_vs_colsum_diff": abs(s3 - refs["v3"]), "combine_ms": combine_ms,
               "rss_gb": rss(), "per_shard": per}
        print(f"q5 chunks cold_total={out['cold_ms']:.0f} combine={combine_ms:.0f}ms "
              f"groups={uk.size} s1={s1} s2={s2} s3_scaled={sc} RSS {rss():.2f}GB", flush=True)
    else:
        raise SystemExit(f"unknown q {q}")
    with open(FORK / "tests" / "heavy" / f"bench_resident_1B_{q}.json", "w") as fh:
        json.dump(json.loads(json.dumps(out, default=float)), fh, indent=1)
    print("JSON written", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ingest-shard":
        cmd_ingest_shard(int(sys.argv[2]))
    elif cmd == "merge":
        cmd_merge()
    elif cmd in ("q1", "q2", "q3", "q4", "q5"):
        cmd_q(cmd)
    elif cmd == "q3shard":
        cmd_q3shard(int(sys.argv[2]))
    elif cmd == "q3combine":
        cmd_q3combine()
    elif cmd == "q5shard":
        cmd_q5shard(int(sys.argv[2]))
    elif cmd == "q5combine":
        cmd_q5combine()
    else:
        raise SystemExit("usage: bench_resident_1B.py ingest-shard <idx> | merge | q1..q5 | q3shard <idx> | q3combine | q5shard <idx> | q5combine")
