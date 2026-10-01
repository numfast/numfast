# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""bench_all: ONE load, all variants back-to-back (user requirement).

Usage (Git Bash, fork-first PYTHONPATH, explicit system python):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \\
  timeout 900 <syspython> -m tests.research.fshift_mt.bench_all \\
    --n 10000000 --g 1000000 --dist uniform --reps 3 \\
    --out tests/research/fshift_mt/results_10M.jsonl
One process per (N,G,dist) cell-config. Heavy jobs sequential.
Golden-exact every cell (bit-identical). Fixed seed 42.
"""

import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

CAND_ORDER = ("A", "B", "C", "D", "E", "F", "Fbig", "G")


def gen(n, g, dist="uniform", seed=42):
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        k = rng.integers(0, g, size=n, dtype=np.int64).astype(np.int32)
    elif dist == "clustered":
        hot = max(1, g // 20)
        nh = int(n * 0.8)
        kh = rng.integers(0, hot, size=nh, dtype=np.int64)
        kc = rng.integers(0, g, size=n - nh, dtype=np.int64)
        k = np.concatenate([kh, kc])
        rng.shuffle(k)
        k = k.astype(np.int32, copy=False)
    else:
        raise ValueError(dist)
    v = rng.integers(-1000, 1000, size=n, dtype=np.int64).astype(np.int32)
    return (np.ascontiguousarray(k), np.ascontiguousarray(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--g", type=int, default=1_000_000)
    ap.add_argument("--dist", default="uniform",
                    choices=("uniform", "clustered"))
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--threads", default="1,4,8,16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cands", default=",".join(CAND_ORDER))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tlad = [int(x) for x in a.threads.split(",")]
    cands = [c for c in a.cands.split(",") if c]

    from . import candidates as C
    from .golden import verify
    from .kernels import geom, warmup

    # 1) compile ALL variants upfront (tiny data, every kernel + runner)
    t0 = time.perf_counter()
    warmup()
    kt, vt = gen(1024, max(16, min(a.g, 1024)), "uniform")
    for c in cands:
        C.CANDS[c](kt, vt, max(16, min(a.g, 1024)), p=a.p, threads=1)
        if 4 in tlad:
            C.CANDS[c](kt, vt, max(16, min(a.g, 1024)), p=a.p, threads=2)
    comp_ms = (time.perf_counter() - t0) * 1000.0
    print(json.dumps({"event": "compiled", "ms": round(comp_ms, 1),
                      "cands": cands}), flush=True)

    # 2) load data ONCE
    t0 = time.perf_counter()
    k, v = gen(a.n, a.g, a.dist)
    gm = geom(a.g, a.p)
    load_ms = (time.perf_counter() - t0) * 1000.0
    try:
        import psutil
        rss0 = psutil.Process().memory_info().rss / 1e9
    except Exception:
        rss0 = -1.0
    print(json.dumps({"event": "loaded", "n": a.n, "g": a.g, "dist": a.dist,
                      "load_ms": round(load_ms, 1),
                      "rss_gb": round(rss0, 2),
                      "shift": gm["shift"], "W": gm["width"]}), flush=True)
    gc.collect()

    # 3) variants back-to-back on the same buffers
    for c in cands:
        fn = C.CANDS[c]
        for t in tlad:
            ts = []
            inf_last = None
            for _ in range(a.reps):
                gc.collect()
                st, inf = fn(k, v, a.g, p=a.p, threads=t)
                ts.append((inf["part_ms"], inf["agg_ms"], inf["merge_ms"],
                           inf["total_ms"]))
                inf_last = inf
            chk = verify(st, k, v, a.g, label=f"{c}/G{a.g}/T{t}/{a.dist}")
            try:
                import psutil
                rss = psutil.Process().memory_info().rss / 1e9
            except Exception:
                rss = -1.0
            med = [round(float(np.median([x[i] for x in ts])), 2)
                   for i in range(4)]
            row = {"cand": c, "n": a.n, "g": a.g, "dist": a.dist,
                   "p": a.p, "threads": t, "reps": a.reps,
                   "part_ms": med[0], "agg_ms": med[1], "merge_ms": med[2],
                   "total_ms": med[3],
                   "rows_per_s": int(a.n / max(med[3] / 1000, 1e-9)),
                   "tmp_bytes": inf_last["tmp_bytes"],
                   "traffic_bytes": inf_last["traffic_bytes"],
                   "ngroups": chk["ngroups"], "correct": True,
                   "rss_gb": round(rss, 3),
                   "W": gm["width"], "shift": gm["shift"]}
            for kk in ("ws_bytes", "partial_rows", "partial_bytes",
                       "partial_ratio", "route_ms", "reads_Nx",
                       "lpt_imb_pct", "sample_rows", "sub_block"):
                if kk in inf_last:
                    row[kk] = inf_last[kk]
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    sys.exit(main())
