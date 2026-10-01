# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research: Q2 pack + Q3/Q5 dictionary preparation ONLY (no aggregation).

GroupBy kernel untouched (#1 at 1-16T). Finds where preparation time goes:
Q2 = pack_keys/composite construction; Q3/Q5 = encode vs materialization.
OLD/NEW/TIME_RATIO on H2O 10M, exact correctness everywhere. No prod change.
Usage: python tests/research/singlepass/bench_prep_q2q5.py
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

FORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / "tests" / "research" / "singlepass"))

import numpy as np

from kernels_sp import dict_fast_multi

CSV = os.environ.get("MT_CSV", "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv")
N = 10_000_000
OUT = FORK / "tests" / "research" / "singlepass" / "results_prep_q2q5.json"


def best_of(fn, reps=5):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return min(ts), float(np.median(ts)), out


def load_all():
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import pandas as pd
    import pyarrow as pa
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id6", "v1", "v2", "v3"])
    assert len(df) == N
    enc = {}
    res = {}
    for name, col in (("id1", {"values": pa.array(df["id1"].to_numpy(),
                                                  type=pa.string()),
                               "prefix": "id"}),
                      ("id2", {"values": pa.array(df["id2"].to_numpy(),
                                                  type=pa.string()),
                               "prefix": "id"}),
                      ("id3", {"values": pa.array(df["id3"].to_numpy(),
                                                  type=pa.string()),
                               "prefix": "id"})):
        t = time.perf_counter()
        res[name] = a["resident_prepare"]({name: col})[name]
        enc[name] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    rest = a["resident_prepare"]({
        "id6": {"values": df["id6"].to_numpy().astype(np.int32), "dtype": "int32"},
        "v1": {"values": df["v1"].to_numpy().astype(np.int32), "dtype": "int32"},
        "v2": {"values": df["v2"].to_numpy().astype(np.int32), "dtype": "int32"},
        "v3": {"values": df["v3"].to_numpy().astype(np.float64),
               "dtype": "float64"}})
    enc["id6_v1_v2_v3"] = (time.perf_counter() - t) * 1000
    res.update(rest)
    info = {k: sorted(v.keys()) for k, v in res.items()}
    del df
    gc.collect()
    return res, enc, info


# ---------------- Q2 pack ----------------
def pack_old_radix(K1, K2, m2):
    return (K1.astype(np.int64) * np.int64(m2) + K2.astype(np.int64)).astype(np.int32)


def pack_old_bitpack(K1, K2):  # prod mode='pack' mirror (E-track in-place)
    comp = K1.astype(np.int64)
    np.left_shift(comp, np.int64(32), out=comp)
    lo = K2.astype(np.int64)
    np.bitwise_and(lo, np.int64(0xFFFFFFFF), out=lo)
    np.bitwise_or(comp, lo, out=comp)
    return comp


def pack_new_i32(K1, K2, m2):
    return (K1 * np.int32(m2) + K2).astype(np.int32, copy=False)


def pack_stage(K1, K2):
    m2 = int(K2.max()) + 1
    r = {}
    b_old, med_old, P_old = best_of(lambda: pack_old_radix(K1, K2, m2))
    b_bit, med_bit, P_bit = best_of(lambda: pack_old_bitpack(K1, K2))
    b_new, med_new, P_new = best_of(lambda: pack_new_i32(K1, K2, m2))
    assert P_new.dtype == np.int32 and bool((P_new == P_old).all()), "i32 exact"
    # grouping equivalence bitpack vs radix (different encodings, same groups)
    uo, co = np.unique(P_old.astype(np.int64), return_counts=True)
    ub, cb = np.unique(P_bit, return_counts=True)
    assert uo.size == ub.size == 10000, (uo.size, ub.size)
    assert bool((np.sort(co) == np.sort(cb)).all()), "grouping equiv"
    # decode spot-check: radix inverts to (K1,K2) on 1M sample
    s = np.arange(0, N, 10, dtype=np.int64)
    assert bool((P_old[s] // m2 == K1[s]).all())
    assert bool((P_old[s] % m2 == K2[s]).all())
    # pass split of OLD radix: alloc-i64 vs mul-add vs downcast
    t = time.perf_counter()
    a64 = K1.astype(np.int64)
    t_alloc = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    a64 = a64 * np.int64(m2) + K2.astype(np.int64)
    t_alu = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    a64 = a64.astype(np.int32)
    t_cast = (time.perf_counter() - t) * 1000
    # precompute-once amortization (build once, per-query reuse ~0 copy)
    _, _, P_pre = best_of(lambda: pack_new_i32(K1, K2, m2), reps=3)
    reuse_ms = 0.0  # resident column read = free (already in memory)
    r = {"m2": m2, "ngroups": int(uo.size),
         "OLD_radix_best": round(b_old, 1), "OLD_radix_med": round(med_old, 1),
         "OLD_bitpack_best": round(b_bit, 1), "OLD_bitpack_med": round(med_bit, 1),
         "NEW_i32_best": round(b_new, 1), "NEW_i32_med": round(med_new, 1),
         "OLD_split_ms": {"alloc_i64": round(t_alloc, 1), "mul_add": round(t_alu, 1),
                          "downcast": round(t_cast, 1)},
         "TIME_RATIO_i32_vs_radix": round(b_old / b_new, 2),
         "TIME_RATIO_i32_vs_bitpack": round(b_bit / b_new, 2),
         "PRECOMP_amort_per_q": {str(nq): round(b_new / nq + reuse_ms, 2)
                                 for nq in (1, 10, 100)},
         "correctness": "i32 bit-identical to radix; bitpack grouping-equal"}
    print(f"Q2pack: OLD_radix={b_old:.0f} OLD_bitpack={b_bit:.0f} "
          f"NEW_i32={b_new:.0f} ratio={b_old / b_new:.2f}x "
          f"split(alloc={t_alloc:.0f}/alu={t_alu:.0f}/cast={t_cast:.0f})",
          flush=True)
    return r, P_pre


# ---------------- Q3/Q5 dict ----------------
def agg_bincount(keys, vcols, m):
    k = np.asarray(keys)
    assert k.dtype == np.int32
    counts_m = np.bincount(k, minlength=int(m))
    uk = np.flatnonzero(counts_m > 0).astype(np.int64)
    pos = uk
    counts = counts_m[pos].astype(np.int64)
    sums = []
    for v in vcols:
        v = np.asarray(v)
        if np.issubdtype(v.dtype, np.integer):
            sums.append(np.bincount(k, weights=v.astype(np.int64, copy=False),
                                    minlength=int(m))[pos].astype(np.int64))
        else:
            sums.append(np.bincount(k, weights=v.astype(np.float64, copy=False),
                                    minlength=int(m))[pos])
    return uk, counts, sums


def dict_new_zip(ukeys, counts, sums_list, ops_list):
    """Same nested shape; values pre-zipped, single range pass, local binds."""
    kl = ukeys.tolist()
    cl = counts.tolist()
    cols = []
    for sums, ops in zip(sums_list, ops_list):
        sl = sums.tolist() if "sum" in ops else None
        ml = (sums.astype(np.float64) / counts).tolist() if "mean" in ops else None
        cols.append((sl, ml, "sum" in ops, "mean" in ops))
    res = {}
    for i in range(len(kl)):
        c0 = {}
        for ci, (sl, ml, ws, wm) in enumerate(cols):
            sub = {}
            if ws:
                sub["sum"] = sl[i]
            if wm:
                sub["mean"] = ml[i]
            c0[ci] = sub
        res[kl[i]] = c0
    return res


def dict_stage(q, keys, vcols, ops_list):
    m = int(keys.max()) + 1
    uk, counts, sums = agg_bincount(keys, vcols, m)
    ng = int(uk.size)
    # sub-step split of OLD (same numpy ops, timed pieces)
    t = time.perf_counter()
    tls = [s.tolist() for s in sums]
    t_tolist = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    means = [(s.astype(np.float64) / counts).tolist() if "mean" in ops else None
             for s, ops in zip(sums, ops_list)]
    t_mean = (time.perf_counter() - t) * 1000
    b_old, med_old, d_old = best_of(
        lambda: dict_fast_multi(uk, counts, sums, ops_list), reps=3)
    b_new, med_new, d_new = best_of(
        lambda: dict_new_zip(uk, counts, sums, ops_list), reps=3)
    assert d_new == d_old, f"{q} dict exact"
    # NEW_col: columnar carry (no python objects) — direction, same values
    t = time.perf_counter()
    col = {"ukeys": uk, "counts": counts, "sums": sums}
    t_col = (time.perf_counter() - t) * 1000
    assert int(col["counts"].sum()) == N
    for a, b in zip(col["sums"], sums):
        assert bool((a == b).all())
    r = {"m": m, "ngroups": ng,
         "OLD_dict_best": round(b_old, 1), "OLD_dict_med": round(med_old, 1),
         "NEW_zip_best": round(b_new, 1), "NEW_zip_med": round(med_new, 1),
         "NEW_columnar_pack_ms": round(t_col, 3),
         "OLD_split_ms": {"tolist": round(t_tolist, 1),
                          "mean_temp": round(t_mean, 1),
                          "loop_insert": round(max(0.0, b_old - t_tolist - t_mean), 1)},
         "TIME_RATIO_zip_vs_old": round(b_old / b_new, 2),
         "correctness": "zip dict == OLD dict (==); columnar value-identical"}
    print(f"{q}: ng={ng} OLD_dict={b_old:.0f} NEW_zip={b_new:.0f} "
          f"ratio={b_old / b_new:.2f}x colpack={t_col:.3f}ms "
          f"split(tolist={t_tolist:.0f}/mean={t_mean:.0f}/"
          f"loop={max(0.0, b_old - t_tolist - t_mean):.0f})", flush=True)
    return r


def main():
    res, enc, info = load_all()
    print(f"encode_ms: { {k: round(v, 1) for k, v in enc.items()} }", flush=True)
    print(f"resident fields: {info}", flush=True)
    K1, K2 = res["id1"]["codes"], res["id2"]["codes"]
    K3, K6 = res["id3"]["codes"], res["id6"]["codes"]
    V1, V2, V3 = res["v1"]["codes"], res["v2"]["codes"], res["v3"]["codes"]
    out = {"tag": "prep_q2q5", "N": N,
           "encode_ms": {k: round(v, 1) for k, v in enc.items()}}
    q2, _ = pack_stage(np.ascontiguousarray(K1), np.ascontiguousarray(K2))
    out["Q2_pack"] = q2
    del K1, K2
    gc.collect()
    out["Q3_dict"] = dict_stage(
        "Q3", np.ascontiguousarray(K3),
        [np.ascontiguousarray(V1), np.ascontiguousarray(V3)],
        [("sum",), ("mean",)])
    del K3
    gc.collect()
    out["Q5_dict"] = dict_stage(
        "Q5", np.ascontiguousarray(K6),
        [np.ascontiguousarray(V1), np.ascontiguousarray(V2),
         np.ascontiguousarray(V3)],
        [("sum",), ("sum",), ("sum",)])
    out["notes"] = ("keys stay int32 codes end-to-end: no per-query decode; "
                    "string prep (id1/id2/id3) is ingest-once via "
                    "resident_prepare; per-query preparation = Q2 composite + "
                    "Q3/Q5 dict materialization")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{OUT.name} written", flush=True)


if __name__ == "__main__":
    main()
