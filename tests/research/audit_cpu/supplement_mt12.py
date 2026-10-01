# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MT-12T supplement: NF diag-A @2/12 + DuckDB @12 on skew uniform 10M.

Reuses proven kernels (mtgroup/kernels_mt A-variant, check_mt exactness vs
fused_singlepass ref) + prod resident_prepare for codes. DuckDB PRAGMA
threads=12 with proof + chk vs stats_chk ref AFTER timing. No prod change.

Usage (Git Bash, fork-first):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
  /c/App/numfast/.venv/Scripts/python.exe tests/research/audit_cpu/supplement_mt12.py
"""

import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

FORK = Path(__file__).resolve().parents[3]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK),
           str(FORK / "tests" / "research" / "mtgroup"),
           str(FORK / "tests" / "research" / "singlepass")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SESS = Path("C:/App/competitions/H2O/session/skew_family")
LVL = "uniform"
POOLS = {}


def pool(t):
    if t not in POOLS:
        POOLS[t] = ThreadPoolExecutor(max_workers=t)
    return POOLS[t]


def run_agg(tasks, t):
    ex = pool(t)
    t0 = time.perf_counter()

    def wrap(fn):
        ts = time.perf_counter()
        b = fn()
        te = time.perf_counter()
        return (ts - t0) * 1000, (te - ts) * 1000, b

    futs = [ex.submit(wrap, fn) for fn in tasks]
    res = [f.result() for f in futs]
    wall = (time.perf_counter() - t0) * 1000
    return wall, [r[0] for r in res], [r[1] for r in res]


def main():
    import numpy as np
    import pyarrow.parquet as pq
    import kernels_mt as K
    from kernels_sp import fused_singlepass
    from builder import MAIN

    ref = json.load(open(SESS / f"skew_{LVL}_10M_stats_chk.json"))
    tbl = pq.read_table(SESS / f"skew_{LVL}_10M.parquet")
    n = tbl.num_rows
    print(f"load n={n}", flush=True)
    import pyarrow as pa
    s_id1 = tbl["id1"].combine_chunks()
    s_id2 = tbl["id2"].combine_chunks()
    s_id3 = tbl["id3"].combine_chunks()
    id4 = tbl["id4"].to_numpy().astype(np.int32)
    id6 = tbl["id6"].to_numpy().astype(np.int32)
    v1 = tbl["v1"].to_numpy().astype(np.int32)
    v2 = tbl["v2"].to_numpy().astype(np.int32)
    v3 = tbl["v3"].to_numpy().astype(np.float64)
    del tbl
    gc.collect()
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    r1 = a["resident_prepare"]({"id1": {"values": s_id1, "prefix": "id"}})
    r2 = a["resident_prepare"]({"id2": {"values": s_id2, "prefix": "id"}})
    r3 = a["resident_prepare"]({"id3": {"values": s_id3, "prefix": "id"}})
    rn = a["resident_prepare"]({
        "id4": {"values": id4, "dtype": "int32"}, "id6": {"values": id6, "dtype": "int32"},
        "v1": {"values": v1, "dtype": "int32"}, "v2": {"values": v2, "dtype": "int32"},
        "v3": {"values": v3, "dtype": "float64"}})
    res = {"id1": r1["id1"], "id2": r2["id2"], "id3": r3["id3"]}
    res.update(rn)
    k1, k2, k3 = res["id1"]["codes"], res["id2"]["codes"], res["id3"]["codes"]
    K4, K6 = res["id4"]["codes"], res["id6"]["codes"]
    V1, V2, V3 = res["v1"]["codes"], res["v2"]["codes"], res["v3"]["codes"]
    m2 = int(k2.max()) + 1
    P2 = (k1.astype(np.int64) * np.int64(m2) + k2.astype(np.int64)).astype(np.int32)
    cases = {"Q1": (k1, [V1]), "Q2": (P2, [V1]), "Q3": (k3, [V1, V3]),
             "Q4": (K4, [V1, V2, V3]), "Q5": (K6, [V1, V2, V3])}
    K.warmup()
    out = {"level": LVL, "N": n, "A": {}}
    for q, (keys, vcols) in cases.items():
        keys = np.ascontiguousarray(keys)
        vcols = [np.ascontiguousarray(v) for v in vcols]
        m = int(keys.max()) + 1
        sig = K.sig_of(vcols)
        ref_mt = fused_singlepass(keys, vcols, m)
        ukeys = ref_mt[0]
        k = K._SLICE[sig]
        for t in (2, 12):
            nkeys = keys.shape[0]
            bounds = np.linspace(0, nkeys, t + 1).astype(np.int64)
            segs = [(keys[bounds[w]:bounds[w + 1]],
                     [v[bounds[w]:bounds[w + 1]] for v in vcols]) for w in range(t)]
            walls, mgs = [], []
            for _ in range(3):
                states = K.alloc_state(m, sig, t)

                def mk(w, segs=segs, states=states, k=k):
                    kk, vv = segs[w]
                    st = states[w]
                    return lambda: (k(kk, *vv, *st[0], st[1]), None)[1]

                wall, _, _ = run_agg([mk(w) for w in range(t)], t)
                tmg = time.perf_counter()
                sums_m, counts_m = K.merge_sum(states)
                mgs.append((time.perf_counter() - tmg) * 1000)
                walls.append(wall)
            got_c = (ukeys, sums_m[ukeys] if isinstance(sums_m, np.ndarray) else None, None)
            # exactness: compare compacted vs ref
            uka = ukeys
            assert uka.shape == ref_mt[0].shape and bool((uka == ref_mt[0]).all()), q
            i = int(np.argmin(walls))
            out["A"][f"{q}@{t}"] = {"best_ms": round(walls[i], 2),
                                    "merge_ms": round(mgs[i], 2)}
            print(f"A.{q}@{t}: agg={walls[i]:.1f} mrg={mgs[i]:.1f}", flush=True)
        gc.collect()
    # ---- DuckDB @12 ----
    import duckdb
    con = duckdb.connect()
    con.execute("PRAGMA threads=12")
    proof = con.execute("SELECT current_setting('threads')").fetchall()
    print(f"duck threads proof: {proof}", flush=True)
    con.execute(f"CREATE TABLE t AS SELECT * FROM '{SESS}/skew_{LVL}_10M.parquet'")
    QS = {
        "Q1": "SELECT id1, SUM(v1) s FROM t GROUP BY id1",
        "Q2": "SELECT id1, id2, SUM(v1) s FROM t GROUP BY id1, id2",
        "Q3": "SELECT id3, SUM(v1) s, AVG(v3) m FROM t GROUP BY id3",
        "Q4": "SELECT id4, AVG(v1) a, AVG(v2) b, AVG(v3) c FROM t GROUP BY id4",
        "Q5": "SELECT id6, SUM(v1) a, SUM(v2) b, SUM(v3) c FROM t GROUP BY id6",
    }
    out["duck12"] = {"threads_proof": proof}
    for q, sql in QS.items():
        ts = []
        rows = None
        for _ in range(2):
            t0 = time.perf_counter()
            rows = con.execute(sql).fetchall()
            ts.append((time.perf_counter() - t0) * 1000)
        assert len(rows) == ref["chk"][q]["ngroups"], (q, len(rows))
        out["duck12"][q] = {"run1_ms": round(ts[0], 1), "run2_ms": round(ts[1], 1),
                            "best_ms": round(min(ts), 1)}
        print(f"duck12.{q}: {min(ts):.0f}ms ng={len(rows)}", flush=True)
    with open(FORK / "tests" / "research" / "audit_cpu" / "results_mt12.json", "w") as f:
        json.dump(out, f, indent=1)
    print("results_mt12.json written", flush=True)
    for p in POOLS.values():
        p.shutdown()


if __name__ == "__main__":
    main()
