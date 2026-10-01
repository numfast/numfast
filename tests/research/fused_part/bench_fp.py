# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""bench_fp: one cell per invocation (timeout-friendly chunks).
Usage (Git Bash, fork-first PYTHONPATH, explicit system python):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \\
  timeout 600 python -m tests.research.fused_part.bench_fp Fshift --n 10000000 \\
    --g 1000000 --p 16 --threads 1
CAND = A | Fshift|Fscaled|Fdiv|Fmask | Pshift|Pscaled|Pdiv|Pmask.
F* = fused scatter-direct (golden exact vs flat reference).
P* = pid-computation cost alone (numpy pass, round-trip exact).
Golden-verify every cell. Fixed seed 42. Median of reps.
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


def pid_numpy(k, variant, par, p):
    ki = k.astype(np.int64, copy=False)
    if variant == "shift":
        return np.minimum(ki >> np.int64(par["shift"]), np.int64(p - 1))
    if variant == "scaled":
        return (ki * np.int64(p)) // np.int64(par["g"])
    if variant == "div":
        return np.minimum(ki // np.int64(par["ws"]), np.int64(p - 1))
    if variant == "mask":
        return (ki & np.int64(p - 1)).astype(np.int64)
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--g", type=int, default=1_000_000)
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from . import a_flat as A
    from . import f_fused as F
    from .golden import verify, verify_pid
    from .kernels import HAS_NUMBA

    k, v = gen(a.n, a.g)
    gc.collect()

    if a.cand == "A":
        A.run(k[:1024], v[:1024], a.g)
        gc.collect()
        parts, aggs, merges, tots = [], [], [], []
        for _ in range(a.reps):
            gc.collect()
            st, inf = A.run(k, v, a.g)
            parts.append(inf["part_ms"])
            aggs.append(inf["agg_ms"])
            merges.append(inf["merge_ms"])
            tots.append(inf["total_ms"])
        chk = verify(st, k, v, a.g, label=a.cand)
        row = {"cand": a.cand, "n": a.n, "g": a.g, "p": a.p,
               "threads": 1, "reps": a.reps,
               "part_ms": round(float(np.median(parts)), 2),
               "agg_ms": round(float(np.median(aggs)), 2),
               "merge_ms": round(float(np.median(merges)), 2),
               "total_ms": round(float(np.median(tots)), 2),
               "rows_per_s": int(a.n / max(float(np.median(tots)) / 1000, 1e-9)),
               "tmp_bytes": inf["tmp_bytes"],
               "traffic_bytes": inf["traffic_bytes"],
               "ngroups": chk["ngroups"], "correct": True}
    elif a.cand.startswith("F"):
        variant = {"Fshift": "shift", "Fscaled": "scaled",
                   "Fdiv": "div", "Fmask": "mask"}[a.cand]
        if HAS_NUMBA:
            from .kernels import fused_div  # noqa: F401 (compile warmup)
            F.run(k[:1024], v[:1024], a.g, p=a.p, variant=variant,
                  threads=a.threads)
        gc.collect()
        parts, aggs, merges, tots = [], [], [], []
        for _ in range(a.reps):
            gc.collect()
            st, inf = F.run(k, v, a.g, p=a.p, variant=variant,
                            threads=a.threads)
            parts.append(inf["part_ms"])
            aggs.append(inf["agg_ms"])
            merges.append(inf["merge_ms"])
            tots.append(inf["total_ms"])
        chk = verify(st, k, v, a.g, label=a.cand)
        row = {"cand": a.cand, "variant": variant, "n": a.n, "g": a.g,
               "p": a.p, "threads": a.threads, "reps": a.reps,
               "part_ms": round(float(np.median(parts)), 2),
               "agg_ms": round(float(np.median(aggs)), 2),
               "merge_ms": round(float(np.median(merges)), 2),
               "total_ms": round(float(np.median(tots)), 2),
               "rows_per_s": int(a.n / max(float(np.median(tots)) / 1000, 1e-9)),
               "tmp_bytes": inf["tmp_bytes"],
               "traffic_bytes": inf["traffic_bytes"],
               "width": inf["width"],
               "ngroups": chk["ngroups"], "correct": True}
    elif a.cand.startswith("P"):
        variant = {"Pshift": "shift", "Pscaled": "scaled",
                   "Pdiv": "div", "Pmask": "mask"}[a.cand]
        par = F.params(variant, a.g, a.p)
        par["g"] = a.g
        pid_numpy(k[:1024], variant, par, a.p)  # warmup
        gc.collect()
        ts = []
        for _ in range(a.reps):
            gc.collect()
            t0 = time.perf_counter()
            pid = pid_numpy(k, variant, par, a.p)
            ts.append((time.perf_counter() - t0) * 1000.0)
        chk = verify_pid(pid, k, variant, par, a.g, a.p, label=a.cand)
        med = float(np.median(ts))
        row = {"cand": a.cand, "variant": variant, "n": a.n, "g": a.g,
               "p": a.p, "threads": 1, "reps": a.reps,
               "part_ms": round(med, 2), "agg_ms": 0.0, "merge_ms": 0.0,
               "total_ms": round(med, 2),
               "rows_per_s": int(a.n / max(med / 1000, 1e-9)),
               "pids_used": chk["pids_used"], "correct": True}
    else:
        raise ValueError(f"unknown cand {a.cand}")

    print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    sys.exit(main())
