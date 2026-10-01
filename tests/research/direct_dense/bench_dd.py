# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""bench_dd: one cell per invocation (timeout-friendly chunks).
Usage: python -m tests.research.direct_dense.bench_dd CAND --n N --m M
       --pattern uniform|skewed|sorted --mode q1|q4|q5|main [--out row.json]
CAND = A|B|C|Dsoa|Daos|E. Golden-verify every cell (exact). Fixed seed 42.
"""
import argparse
import gc
import json
import sys
import time

import numpy as np


def gen(n, m, pattern, seed=42):
    rng = np.random.default_rng(seed)
    if pattern == "uniform":
        k = rng.integers(0, m, size=n, dtype=np.int64).astype(np.int32)
    elif pattern == "skewed":
        u = rng.random(n)
        k = ((m * u * u).astype(np.int64) % m).astype(np.int32)
    elif pattern == "sorted":
        k = np.sort(rng.integers(0, m, size=n, dtype=np.int64)).astype(np.int32)
    else:
        raise ValueError(pattern)
    v = rng.integers(-1000, 1000, size=n).astype(np.int32)
    return k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--pattern", default="uniform")
    ap.add_argument("--mode", default="main")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from . import a_shift as A
    from . import b_direct as B
    from . import c_numba as C
    from . import d_fused as D
    from . import e_nocopy as E
    from .golden import verify

    fused = a.mode in ("q4",)
    k, v = gen(a.n, a.m, a.pattern)
    gc.collect()
    # warmup outside timers (numba compile)
    kw = dict(m=a.m) if a.cand in ("B", "C") else {}
    if a.cand == "A":
        _s, _ = A.run(k[:1024], v[:1024], fused=fused)
    elif a.cand == "B":
        _s, _ = B.run(k[:1024], v[:1024], m=a.m, fused=fused)
    elif a.cand == "C":
        _s, _ = C.run(k[:1024], v[:1024], m=a.m)
    elif a.cand.startswith("D"):
        _s, _ = D.run(k[:1024], v[:1024], m=a.m,
                      layout="aos" if a.cand == "Daos" else "soa")
    else:
        _s, _ = E.run(k[:1024], v[:1024], fused=fused)
    gc.collect()
    t0 = time.perf_counter()
    if a.cand == "A":
        st, info = A.run(k, v, fused=fused)
    elif a.cand == "B":
        st, info = B.run(k, v, m=a.m, fused=fused)
    elif a.cand == "C":
        st, info = C.run(k, v, m=a.m)
    elif a.cand.startswith("D"):
        st, info = D.run(k, v, m=a.m, layout="aos" if a.cand == "Daos" else "soa")
    else:
        st, info = E.run(k, v, fused=fused)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    chk = verify(st, k, v, fused=fused, label=a.cand)
    row = {"cand": a.cand, "n": a.n, "m": a.m, "pattern": a.pattern,
           "mode": a.mode, "fused": fused, "gi_ms": round(info["gi_ms"], 2),
           "agg_ms": round(info["agg_ms"], 2),
           "total_ms": round(info["total_ms"], 2),
           "wall_ms": round(wall_ms, 2),
           "rows_per_s": int(a.n / max(wall_ms / 1000, 1e-9)),
           "tmp_bytes": info["tmp_bytes"], "traffic_bytes": info["traffic_bytes"],
           "state_bytes": chk["state_bytes"], "ngroups": chk["ngroups"],
           "correct": True}
    print(json.dumps(row))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(row, f)


if __name__ == "__main__":
    sys.exit(main())
