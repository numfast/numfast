# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""bench_h2o: REAL H2O DATA through lab A-O (protocol stage 3+4).

Loads G1_1e7_1e2_0_0.csv id/v columns, encodes string ids to int32 codes
(encode-once, like resident_prepare), then runs lab local() in 128K blocks
+ threaded + one funnel merge per query. Golden-verified per query
(verifier) + ngroups/checksum vs session GOLD anchors.
Usage: python -m tests.research.groupby_lab.bench_h2o OUT.json [QSUBSET]
"""

import gc
import json
import sys
import time

import numpy as np

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
BLOCK = 131072

GOLD_NGROUPS = {"Q1": 100, "Q2": 10000, "Q3": 100000, "Q4": 100, "Q5": 100000}


def warmup(mod):
    rng = np.random.default_rng(0)
    kd = rng.integers(0, 50, size=2048).astype(np.int32)
    vd = rng.integers(-9, 9, size=2048).astype(np.int32)
    mod.local(kd, vd)
    ks = np.sort(rng.integers(0, 1000000, size=8192).astype(np.int32))
    vs = rng.integers(-9, 9, size=8192).astype(np.int32)
    a, _ = mod.local(ks, vs)
    kh = rng.integers(0, 10000000, size=8192).astype(np.int32)
    b, _ = mod.local(kh, vs)
    mod.merge([a, b])


def main():
    from .candidates import ALGOS
    from .runners import merge_all, run_threaded, cpu_sum_ms
    from .verifier import verify

    out = sys.argv[1] if len(sys.argv) > 1 else None
    # argv[2]: queries csv, e.g. Q1,Q3  (default all five)
    # argv[3]: algos joined by +, e.g. A+D+E (default all fifteen)
    qlist = (sys.argv[2].split(",") if len(sys.argv) > 2
             else ["Q1", "Q2", "Q3", "Q4", "Q5"])
    names = (sys.argv[3].split("+") if len(sys.argv) > 3
             else list("ABCDEFGHIJKLMNO"))
    if len(sys.argv) > 4:
        global CSV
        CSV = sys.argv[4]

    import pandas as pd
    t0 = time.perf_counter()
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6",
                                   "v1", "v2", "v3"])
    load_ms = (time.perf_counter() - t0) * 1000.0
    n = len(df)
    print(f"csv {n} rows load={load_ms:.0f}ms", flush=True)
    t0 = time.perf_counter()
    codes = {}
    for c in ["id1", "id2", "id3"]:
        codes[c], _ = pd.factorize(df[c], sort=True)
        codes[c] = codes[c].astype(np.int32)
    for c in ["id4", "id6", "v1", "v2"]:
        codes[c] = df[c].to_numpy().astype(np.int32)
    codes["v3"] = df["v3"].to_numpy().astype(np.float64)
    enc_ms = (time.perf_counter() - t0) * 1000.0
    print(f"encode={enc_ms:.0f}ms nunique id1={codes['id1'].max() + 1} "
          f"id2={codes['id2'].max() + 1} id3={codes['id3'].max() + 1}",
          flush=True)
    del df
    gc.collect()

    k1 = codes["id1"]
    k12 = (k1.astype(np.int64) << np.int64(32)) | (
        codes["id2"].astype(np.int64) & np.int64(0xFFFFFFFF))
    # Q2 keys must fit int32-lab? lab takes int64 fine; keep int64 packed.
    k3 = codes["id3"]
    v1 = codes["v1"]
    queries = {
        "Q1": (k1.astype(np.int64), v1),
        "Q2": (k12, v1),
        "Q3": (k3.astype(np.int64), v1),
        "Q4": (k1.astype(np.int64), v1),
        "Q5": (k3.astype(np.int64), v1),
    }
    names = (sys.argv[3].split("+") if len(sys.argv) > 3
             else list("ABCDEFGHIJKLMNO"))
    for nm in names:
        warmup(ALGOS[nm])

    rows = []
    for qn in qlist:
        keys, vals = queries[qn]
        blks = [{"keys": keys[i:i + BLOCK], "vals": vals[i:i + BLOCK]}
                for i in range(0, n, BLOCK)]
        ref_uk = np.unique(keys).size
        for name in names:
            mod = ALGOS[name]
            t0 = time.perf_counter()
            states, stats, wall = run_threaded(mod, blks, 8)
            fin, info = merge_all(mod, states)
            el = (time.perf_counter() - t0) * 1000.0
            chk = verify(fin, keys, vals, label=f"{name}/{qn}/10M")
            if chk["ngroups"] != GOLD_NGROUPS[qn]:
                print(f"note {name}/{qn}: ngroups {chk['ngroups']} "
                      f"!= 10M-GOLD {GOLD_NGROUPS[qn]} (100M file has more keys)",
                      flush=True)
            assert chk["ngroups"] == ref_uk
            cpu = cpu_sum_ms(stats)
            rows.append({
                "algo": name, "regime": f"10M/{qn}", "query": qn,
                "ms": el, "local_wall_ms": wall, "cpu_ms": cpu,
                "merge_ms": info.get("merge_ms", 0.0),
                "bytes": chk["state_bytes"], "ngroups": chk["ngroups"],
                "throughput_rows_s": n / max(el / 1e3, 1e-9),
            })
            print(f"ok 10M {qn} {name} total={el:.0f}ms wall={wall:.0f}ms "
                  f"merge={info.get('merge_ms', 0.0):.0f}ms "
                  f"groups={chk['ngroups']} "
                  f"thr={n / max(el / 1e3, 1e-9) / 1e6:.2f}M/s", flush=True)
        gc.collect()
    if out:
        with open(out, "w") as f:
            json.dump({"n": n, "load_ms": load_ms, "enc_ms": enc_ms,
                       "rows": rows}, f, indent=1, default=str)
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
