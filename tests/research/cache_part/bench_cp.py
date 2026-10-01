# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""bench_cp: one cell per invocation (timeout-friendly chunks).
Usage (Git Bash, fork-first PYTHONPATH, explicit system python):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \\
  timeout 300 python -m tests.research.cache_part.bench_cp B --n 10000000 --g 1000000 --p 16
CAND = A|B|C|D. Golden-verify every cell (bit-exact). Fixed seed 42. best-of-3.
"""
import argparse
import gc
import json
import sys
import time

import numpy as np


def gen(n, g, seed=42):
    rng = np.random.default_rng(seed)
    k = rng.integers(0, g, size=n, dtype=np.int64).astype(np.int32)
    v = rng.integers(-1000, 1000, size=n).astype(np.int32)
    return k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--g", type=int, default=1_000_000)
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from . import a_flat as A
    from . import b_rangepart as B
    from . import c_blocklocal as C
    from . import d_scatterpart as D
    from .golden import verify
    from .kernels import HAS_NUMBA, scatter_by_pid

    mods = {"A": A, "B": B, "C": C, "D": D}
    mod = mods[a.cand]
    kw = {}
    if a.cand in ("B", "D"):
        kw = {"p": a.p, "threads": a.threads}
    elif a.cand == "C":
        kw = {"p": a.p, "threads": a.threads,
              "block": a.block or None}

    k, v = gen(a.n, a.g)
    gc.collect()
    if HAS_NUMBA:  # warmup outside timers
        kk = np.arange(64, dtype=np.int32)
        vv = np.ones(64, dtype=np.int32)
        pp = np.zeros(64, dtype=np.int64)
        cu = np.array([0], dtype=np.int64)
        scatter_by_pid(kk, vv, pp, cu, kk.copy(), vv.copy())
    mod.run(k[:1024], v[:1024], a.g,
            **{kk_: vv_ for kk_, vv_ in kw.items()
               if not (a.cand == "C" and kk_ == "block" and vv_ is None)})
    gc.collect()
    parts, aggs, merges, tots = [], [], [], []
    for _ in range(a.reps):
        gc.collect()
        st, inf = mod.run(k, v, a.g, **kw)
        parts.append(inf["part_ms"])
        aggs.append(inf["agg_ms"])
        merges.append(inf["merge_ms"])
        tots.append(inf["total_ms"])
    chk = verify(st, k, v, a.g, label=a.cand)
    assert st.ngroups == a.g or chk["ngroups"] <= a.g
    row = {"cand": a.cand, "n": a.n, "g": a.g, "p": a.p,
           "block": a.block or (a.n + a.p - 1) // a.p if a.cand == "C"
           else 0,
           "threads": a.threads, "reps": a.reps,
           "part_ms": round(float(np.median(parts)), 2),
           "agg_ms": round(float(np.median(aggs)), 2),
           "merge_ms": round(float(np.median(merges)), 2),
           "total_ms": round(float(np.median(tots)), 2),
           "rows_per_s": int(a.n / max(float(np.median(tots)) / 1000, 1e-9)),
           "tmp_bytes": inf["tmp_bytes"],
           "traffic_bytes": inf["traffic_bytes"],
           "ws_bytes": inf.get("ws_bytes", 0),
           "ngroups": chk["ngroups"], "correct": True}
    print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    sys.exit(main())
