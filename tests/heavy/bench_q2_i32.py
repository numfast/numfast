# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q2 composite int32-direct: production path proof (heavy, 10M only, no 1B).

Public path only (Schema-IR-Planner-Runtime-CPU): resident int32 codes ->
pack_keys mode='radix' (Planner-gated int32-direct when safe, legacy int64
radix fallback otherwise; mode='pack' bitpack untouched) -> groupby sum.
Correctness exact: chk 29998789, 10000 groups, max_diff 0 vs int64-radix
reference (dict-equal) + grouping-equal vs bitpack (sorted counts equal).
Ladder: T in (1,4,8,16) via fresh process env (OMP/MKL/NUMBA=T) vs DuckDB
(PRAGMA threads=T) vs Polars (POLARS_MAX_THREADS=T, fresh process per T).

Usage (Git Bash, explicit python, fork-first):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_q2_i32.py
  T ladder: .../python.exe tests/heavy/bench_q2_i32.py --ladder
"""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CSV = os.environ.get("MT_CSV", "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv")
N = 10_000_000
GOLD_TOTAL = 29998789
GOLD_GROUPS = 10000
OUT = FORK / "tests" / "heavy" / "bench_q2_i32.json"
PY = sys.executable


def best_of(fn, reps=5):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return min(ts), float(sorted(ts)[len(ts) // 2]), out


def load_codes():
    import pandas as pd
    import pyarrow as pa
    from builder import MAIN

    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    df = pd.read_csv(CSV, usecols=["id1", "id2", "v1"])
    assert len(df) == N
    import numpy as np

    r = {}
    for name, col in (("id1", {"values": pa.array(df["id1"].to_numpy(),
                                                  type=pa.string()),
                               "prefix": "id"}),
                      ("id2", {"values": pa.array(df["id2"].to_numpy(),
                                                  type=pa.string()),
                               "prefix": "id"})):
        r[name] = a["resident_prepare"]({name: col})[name]
    v1 = df["v1"].to_numpy().astype(np.int32)
    del df
    gc.collect()
    import numpy as np  # noqa: F811

    K1 = np.ascontiguousarray(r["id1"]["codes"])
    K2 = np.ascontiguousarray(r["id2"]["codes"])
    assert str(K1.dtype) == "int32" and str(K2.dtype) == "int32"
    return a, K1, K2, v1


def pack_stage(a, K1, K2):
    import numpy as np

    m2 = int(K2.max()) + 1
    b_rad, med_rad, _ = best_of(
        lambda: (K1.astype(np.int64) * np.int64(m2)
                 + K2.astype(np.int64)).astype(np.int32))
    comp = K1.astype(np.int64)
    b_bit, med_bit, _ = best_of(lambda: np.bitwise_or(
        np.left_shift(K1.astype(np.int64), np.int64(32)),
        np.bitwise_and(K2.astype(np.int64), np.int64(0xFFFFFFFF))))
    del comp
    jobs = [a["ir_series"]("c1", K1), a["ir_series"]("c2", K2),
            a["ir_pack_keys"]("k", "c1", "c2", mode="radix")]
    g = a["optimize"](a["compile"](jobs))
    b_new, med_new, bufs = best_of(lambda: a["cpu_execute"](g["nodes"]))
    P_new = np.asarray(bufs["k"])
    P_ref = (K1.astype(np.int64) * np.int64(m2)
             + K2.astype(np.int64)).astype(np.int32)
    assert bool((P_new.astype(np.int64) == P_ref.astype(np.int64)).all()), \
        "int32 pack bit-identical to int64 radix"
    # grouping equivalence vs bitpack (different labels, same partition sizes)
    P_bit = np.bitwise_or(np.left_shift(K1.astype(np.int64), np.int64(32)),
                          np.bitwise_and(K2.astype(np.int64),
                                         np.int64(0xFFFFFFFF)))
    co = np.sort(np.unique(P_ref.astype(np.int64), return_counts=True)[1])
    cb = np.sort(np.unique(P_bit, return_counts=True)[1])
    assert co.size == cb.size == GOLD_GROUPS and bool((co == cb).all())
    side = bufs.get("k#pack", {})
    print(f"pack: radix={b_rad:.1f} bitpack={b_bit:.1f} prod={b_new:.1f} "
          f"ratio_radix={b_rad / b_new:.2f}x ratio_bit={b_bit / b_new:.2f}x "
          f"dtype={P_new.dtype} side={side}", flush=True)
    return {"m2": m2, "OLD_radix_best": round(b_rad, 1),
            "OLD_radix_med": round(med_rad, 1),
            "OLD_bitpack_best": round(b_bit, 1),
            "OLD_bitpack_med": round(med_bit, 1),
            "NEW_best": round(b_new, 1), "NEW_med": round(med_new, 1),
            "TIME_RATIO_vs_radix": round(b_rad / b_new, 2),
            "TIME_RATIO_vs_bitpack": round(b_bit / b_new, 2),
            "prod_dtype": str(P_new.dtype), "sidecar": side}


def q2_end_to_end(a, K1, K2, v1, mode, reps=5):
    import numpy as np

    jobs = [a["ir_series"]("c1", K1), a["ir_series"]("c2", K2),
            a["ir_series"]("v", v1),
            a["ir_pack_keys"]("k", "c1", "c2", mode=mode),
            a["ir_groupby"]("g", "v", "k", "sum")]

    def run():
        g = a["optimize"](a["compile"](jobs))
        return a["evaluate"](g, "cpu", N)["result"]

    b, med, res = best_of(run, reps=reps)
    assert len(res) == GOLD_GROUPS, (mode, len(res))
    tot = int(sum(int(v) for v in res.values()))
    assert tot == GOLD_TOTAL, (mode, tot)
    return b, med, res


def correctness(a, K1, K2, v1):
    import numpy as np

    _, _, r_rad = q2_end_to_end(a, K1, K2, v1, "radix")
    _, _, r_bit = q2_end_to_end(a, K1, K2, v1, "pack")
    assert r_rad == r_bit or True  # labels differ by design; groups must match
    co = sorted(r_rad.values())
    cb = sorted(r_bit.values())
    assert len(co) == len(cb) == GOLD_GROUPS
    max_diff = max(abs(x - y) for x, y in zip(co, cb))
    assert max_diff == 0, max_diff
    # int32 prod dict == int64 radix reference dict exactly
    m2 = int(K2.max()) + 1
    ref = {}
    for k1, k2, vv in zip(K1.tolist(), K2.tolist(), v1.tolist()):
        kk = int(k1) * m2 + int(k2)
        ref[kk] = ref.get(kk, 0) + int(vv)
    assert len(ref) == GOLD_GROUPS and sum(ref.values()) == GOLD_TOTAL
    assert dict(sorted(r_rad.items())) == dict(sorted(ref.items())), \
        "prod Q2 dict != int64 radix reference"
    print(f"correctness: chk={GOLD_TOTAL} groups={GOLD_GROUPS} "
          f"max_diff(radix_prod vs bitpack groups)={max_diff} "
          f"ref-dict exact", flush=True)
    return {"chk": GOLD_TOTAL, "ngroups": GOLD_GROUPS, "max_diff": max_diff,
            "ref_dict": "exact"}


def phase_single():
    a, K1, K2, v1 = load_codes()
    out = {"N": N, "pack": pack_stage(a, K1, K2),
           "correctness": correctness(a, K1, K2, v1), "q2": {}}
    for mode in ("pack", "radix"):
        b, med, _ = q2_end_to_end(a, K1, K2, v1, mode)
        out["q2"][mode] = {"best_ms": round(b, 1), "med_ms": round(med, 1)}
        print(f"Q2[{mode}]: best={b:.0f}ms med={med:.0f}ms", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{OUT.name} written", flush=True)
    return out


def q2_one_T():
    """Single-T worker: prints Q2[radix] best for env threads (fresh proc)."""
    a, K1, K2, v1 = load_codes()
    _, _, r = q2_end_to_end(a, K1, K2, v1, "radix", reps=1)  # warm
    b, med, _ = q2_end_to_end(a, K1, K2, v1, "radix", reps=3)
    print(f"Q2_TRESULT best_ms={b:.1f} med_ms={med:.1f}")
    assert len(r) == GOLD_GROUPS


def ladder():
    import numpy as np

    res = {"N": N, "ladder": {}}
    base = {"PYTHONPATH": f"{FORK}{os.pathsep}{FORK.parent / 'app-builder-ponytail'}"}
    for T in (1, 4, 8, 16):
        env = dict(os.environ, OMP_NUM_THREADS=str(T),
                   OPENBLAS_NUM_THREADS=str(T), MKL_NUM_THREADS=str(T),
                   NUMBA_NUM_THREADS=str(T), **base)
        p = subprocess.run([PY, str(FORK / "tests" / "heavy" / "bench_q2_i32.py"),
                            "--one-T"], capture_output=True, text=True, env=env)
        line = [ln for ln in p.stdout.splitlines() if "Q2_TRESULT" in ln]
        assert line, p.stdout[-2000:] + p.stderr[-2000:]
        b = float(line[-1].split("best_ms=")[1].split()[0])
        res["ladder"][f"numfast@{T}"] = round(b, 1)
        print(f"numfast@{T}: {b:.0f}ms", flush=True)
        # DuckDB same thread count, query-only best of 3
        import duckdb

        con = duckdb.connect()
        con.execute(f"CREATE TABLE t AS SELECT * FROM read_csv('{CSV}')")
        con.execute(f"PRAGMA threads={T}")
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            rows = con.execute(
                "SELECT id1, id2, SUM(v1) s FROM t GROUP BY id1, id2").fetchall()
            ts.append((time.perf_counter() - t) * 1000)
        assert len(rows) == GOLD_GROUPS
        res["ladder"][f"duckdb@{T}"] = round(min(ts), 1)
        print(f"duckdb@{T}: {min(ts):.0f}ms", flush=True)
        con.close()
        gc.collect()
    # Polars: one thread count per process
    for T in (1, 4, 8, 16):
        env = dict(os.environ, POLARS_MAX_THREADS=str(T),
                   OMP_NUM_THREADS="1", **base)
        code = ("import polars as pl,time; df=pl.read_csv("
                f"'{CSV}',columns=['id1','id2','v1']);"
                "ts=[]\n"
                "for _ in range(3):\n"
                " t=time.perf_counter();"
                " r=df.group_by(['id1','id2']).agg(pl.col('v1').sum());"
                " ts.append((time.perf_counter()-t)*1000)\n"
                "assert r.height==10000, r.height\n"
                "print(f'Q2_POLARS best_ms={min(ts):.1f}')")
        p = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                           env=env)
        line = [ln for ln in p.stdout.splitlines() if "Q2_POLARS" in ln]
        assert line, p.stdout[-2000:] + p.stderr[-2000:]
        b = float(line[-1].split("best_ms=")[1])
        res["ladder"][f"polars@{T}"] = round(b, 1)
        print(f"polars@{T}: {b:.0f}ms", flush=True)
    with open(FORK / "tests" / "heavy" / "bench_q2_ladder.json", "w") as f:
        json.dump(res, f, indent=1)
    print("bench_q2_ladder.json written", flush=True)


if __name__ == "__main__":
    if "--one-T" in sys.argv:
        q2_one_T()
    elif "--ladder" in sys.argv:
        ladder()
    else:
        phase_single()
