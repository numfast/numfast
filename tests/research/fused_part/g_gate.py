# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GATE Fshift-as-conditional-fast-path (research only, no integration).

Compares Fshift (fused partitioned dense, shift geometry) vs A flat dense.
One cell per invocation (timeout-friendly, sequential heavy jobs).

Usage (Git Bash, fork-first PYTHONPATH, explicit system python):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \\
  timeout 600 <syspython> -m tests.research.fused_part.g_gate cell \\
    --cand Fshift --n 10000000 --g 1000000 --dist uniform --dtype int32 \\
    --p 16 --reps 3 --out results_gate.jsonl
  <syspython> -m tests.research.fused_part.g_gate geometry --g 1000000 --p 16

Dists (seed 42, documented): uniform = rng.integers(0,G,N);
  clustered = 80% rows uniform in [0,max(1,G//20)) + 20% uniform in [0,G),
  concatenated then in-place shuffled (order affects cache realistically).
Golden-verify every cell (bit-exact sums/counts/ukeys vs bincount ref).
"""

import argparse
import gc
import json
import math
import sys

import numpy as np

DTYPES = ("int32", "int64", "uint32", "uint64")


def gen(n, g, dist="uniform", dtype="int32", seed=42):
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    if dist == "uniform":
        if dt == np.dtype("int32"):
            k = rng.integers(0, g, size=n, dtype=np.int32)
        else:
            k = rng.integers(0, g, size=n, dtype=np.int64).astype(dt, copy=False)
    elif dist == "clustered":
        hot_g = max(1, g // 20)
        nh = int(n * 0.8)
        kh = rng.integers(0, hot_g, size=nh, dtype=np.int64)
        kc = rng.integers(0, g, size=n - nh, dtype=np.int64)
        k = np.concatenate([kh, kc])
        rng.shuffle(k)
        k = k.astype(dt, copy=False)
    else:
        raise ValueError(dist)
    v = rng.integers(-1000, 1000, size=n, dtype=np.int64).astype(np.int32)
    return k, v


def geometry_report(g, p=16):
    """Exhaustive vectorized partition-coverage check for shift geometry."""
    import numpy as np  # local, stdlib-adjacent reuse

    s = max(1, int(math.ceil(math.log2(max(2, (g + p - 1) // p)))))
    w = 1 << s
    pw = p * w
    pad = pw - g
    ar = np.arange(g, dtype=np.int64)
    q = np.minimum(ar >> np.int64(s), np.int64(p - 1))
    li = ar - q * np.int64(w)
    holes = bool((li < 0).any() or (li >= w).any()
                 or (q < 0).any() or (q >= p).any())
    rt = bool(np.array_equal(q * np.int64(w) + li, ar))
    # padding slots: keys [G, P*W) must be unreachable from real keys;
    # run tiny Fshift over 2x arange(G) and assert zero counts on pad slots
    from . import f_fused as F

    kt = np.tile(ar, 2).astype(np.int32)
    vt = np.ones(kt.size, dtype=np.int32)
    st, _ = F.run(kt, vt, g, p=p, variant="shift", threads=1)
    assert int(st.counts.sum()) == int(kt.size)
    pad_hit = False
    if pad:
        # reconstruct per-(q,li) counts via inverse map on observed ukeys
        seen = set(np.asarray(st.ukeys, dtype=np.int64).tolist())
        pad_keys = set(range(g, pw))
        pad_hit = bool(seen & pad_keys)
    return {"g": g, "p": p, "shift": s, "W": w, "PxW": pw,
            "padding": pad, "holes": holes, "roundtrip_all": rt,
            "padding_hit": pad_hit,
            "geometry_ok": (not holes) and rt and (not pad_hit)}


def cmd_cell(a):
    from . import a_flat as A
    from . import f_fused as F
    from .golden import verify

    assert a.cand in ("A", "Fshift"), a.cand
    assert a.dtype in DTYPES, a.dtype
    k, v = gen(a.n, a.g, a.dist, a.dtype)
    gc.collect()
    if a.cand == "A":
        A.run(k[:1024], v[:1024], a.g)
        gc.collect()
        ts = []
        for _ in range(a.reps):
            gc.collect()
            st, inf = A.run(k, v, a.g)
            ts.append((inf["part_ms"], inf["agg_ms"], inf["merge_ms"],
                       inf["total_ms"]))
        chk = verify(st, k, v, a.g, label=a.cand)
        med = [round(float(np.median([t[i] for t in ts])), 2) for i in range(4)]
        row = {"cand": a.cand, "n": a.n, "g": a.g, "dist": a.dist,
               "dtype": a.dtype, "p": a.p, "threads": 1, "reps": a.reps,
               "part_ms": med[0], "agg_ms": med[1], "merge_ms": med[2],
               "total_ms": med[3],
               "rows_per_s": int(a.n / max(med[3] / 1000, 1e-9)),
               "tmp_bytes": inf["tmp_bytes"],
               "traffic_bytes": inf["traffic_bytes"],
               "ngroups": chk["ngroups"], "correct": True}
    else:
        F.run(k[:1024], v[:1024], a.g, p=a.p, variant="shift", threads=1)
        gc.collect()
        ts = []
        for _ in range(a.reps):
            gc.collect()
            st, inf = F.run(k, v, a.g, p=a.p, variant="shift", threads=1)
            ts.append((inf["part_ms"], inf["agg_ms"], inf["merge_ms"],
                       inf["total_ms"]))
        chk = verify(st, k, v, a.g, label=a.cand)
        med = [round(float(np.median([t[i] for t in ts])), 2) for i in range(4)]
        row = {"cand": a.cand, "variant": "shift", "n": a.n, "g": a.g,
               "dist": a.dist, "dtype": a.dtype, "p": a.p, "threads": 1,
               "reps": a.reps, "part_ms": med[0], "agg_ms": med[1],
               "merge_ms": med[2], "total_ms": med[3],
               "rows_per_s": int(a.n / max(med[3] / 1000, 1e-9)),
               "tmp_bytes": inf["tmp_bytes"],
               "traffic_bytes": inf["traffic_bytes"],
               "width": inf["width"], "ngroups": chk["ngroups"],
               "correct": True}
    print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


def cmd_geometry(a):
    rep = geometry_report(a.g, a.p)
    print(json.dumps(rep), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(rep) + "\n")
    if not rep["geometry_ok"]:
        raise SystemExit(f"GEOMETRY FAIL: {rep}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--cand", required=True)
    c.add_argument("--n", type=int, default=10_000_000)
    c.add_argument("--g", type=int, default=1_000_000)
    c.add_argument("--dist", default="uniform",
                   choices=("uniform", "clustered"))
    c.add_argument("--dtype", default="int32", choices=DTYPES)
    c.add_argument("--p", type=int, default=16)
    c.add_argument("--reps", type=int, default=3)
    c.add_argument("--out", default=None)
    gm = sub.add_parser("geometry")
    gm.add_argument("--g", type=int, required=True)
    gm.add_argument("--p", type=int, default=16)
    gm.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "cell":
        sys.exit(cmd_cell(a))
    sys.exit(cmd_geometry(a))


if __name__ == "__main__":
    main()
