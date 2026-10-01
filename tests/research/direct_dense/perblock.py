# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""perblock: synthetic multi-block flat whole-op vs per-block strategy.
Blocks: B1 K=100 / B2 K=4000 / B3 K=58000 / B4 K~N (metadata upfront:
rows/unique/min/max/range/density/sorted). Strategies: flat (one run on
concat) vs perblock (local per block + dense-add merge). Extra case:
local K<<global M (B1 keys in 0..100 inside global M=1M code space).
Usage: python -m tests.research.direct_dense.perblock [--out rows.json]
One invocation runs all cells (fast: small N). Golden exact every cell.
"""
import json
import sys
import time

import numpy as np

from . import b_direct as B
from . import c_numba as C
from .golden import verify

SEED = 42
SPECS = [
    ("B1_K100", 200_000, 100),
    ("B2_K4K", 200_000, 4_000),
    ("B3_K58K", 200_000, 58_000),
    ("B4_KeqN", 200_000, 200_000),
]


def block_meta(keys):
    k = np.asarray(keys)
    n = k.size
    mn, mx = int(k.min()), int(k.max())
    span = mx - mn + 1
    g = int(np.unique(k).size)
    srt = bool(np.all(np.diff(k) >= 0)) if n > 1 else True
    return {"rows": n, "unique": g, "min": mn, "max": mx, "range": span,
            "density": round(g / span, 4), "sorted": srt,
            "raw_bytes": int(n * 8)}


def dense_add_merge(states, m):
    t0 = time.perf_counter()
    gs = np.zeros(m, dtype=np.int64)
    gc = np.zeros(m, dtype=np.int64)
    for s in states:
        gs[s.ukeys] += s.sums
        gc[s.ukeys] += s.counts
    mask = gc > 0
    from .contract import State
    st = State(np.flatnonzero(mask).astype(np.int64), gs[mask], gc[mask])
    return st, (time.perf_counter() - t0) * 1000.0


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    rows = []
    # warmup
    _k = np.arange(64, dtype=np.int32)
    _v = np.ones(64, dtype=np.int32)
    B.run(_k % 8, _v, m=8)
    C.run(_k % 8, _v, m=8)
    blocks = {}
    for i, (name, n, kk) in enumerate(SPECS):
        rng = np.random.default_rng(SEED + i)
        k = rng.integers(0, kk, size=n).astype(np.int32)
        v = rng.integers(-1000, 1000, size=n).astype(np.int32)
        blocks[name] = (k, v)
    for name, (k, v) in blocks.items():
        meta = block_meta(k)
        m = int(k.max()) + 1
        t0 = time.perf_counter()
        st_flat, i_flat = B.run(k, v, m=m)
        flat_ms = (time.perf_counter() - t0) * 1000.0
        verify(st_flat, k, v, label=f"flat/{name}")
        # per-block: 4 chunks of 50K, local B + dense-add merge
        parts = [(k[j::4], v[j::4]) for j in range(4)]
        t0 = time.perf_counter()
        Loc = []
        loc_ms = 0.0
        for pk, pv in parts:
            s, info = B.run(pk, pv, m=m)
            loc_ms += info["total_ms"]
            Loc.append(s)
        st_pb, merge_ms = dense_add_merge(Loc, m)
        pb_ms = loc_ms + merge_ms
        verify(st_pb, k, v, label=f"perblock/{name}")
        state_b = sum(s.state_bytes() for s in Loc)
        rows.append({"block": name, "meta": meta, "m": m,
                     "flat_ms": round(flat_ms, 2),
                     "perblock_local_ms": round(loc_ms, 2),
                     "perblock_merge_ms": round(merge_ms, 2),
                     "perblock_total_ms": round(pb_ms, 2),
                     "perblock_state_bytes": state_b,
                     "winner": "flat" if flat_ms <= pb_ms else "perblock"})
    # local K<<global M: B1 keys (0..100) inside M=1M space
    k, v = blocks["B1_K100"]
    meta = block_meta(k)
    M = 1_000_000
    t0 = time.perf_counter()
    stf, _ = B.run(k, v, m=M)
    fms = (time.perf_counter() - t0) * 1000.0
    verify(stf, k, v, label="flat/K<<M")
    parts = [(k[j::4], v[j::4]) for j in range(4)]
    t0 = time.perf_counter()
    Loc = [B.run(pk, pv, m=M)[0] for pk, pv in parts]
    sms, mms = dense_add_merge(Loc, M)
    pms = (time.perf_counter() - t0) * 1000.0
    verify(sms, k, v, label="perblock/K<<M")
    rows.append({"block": "B1_in_M1M", "meta": meta, "m": M,
                 "flat_ms": round(fms, 2),
                 "perblock_total_ms": round(pms, 2),
                 "perblock_merge_ms": round(mms, 2),
                 "note": "local K=100 << global M=1M: dense tmp=16MB, merge O(M)",
                 "winner": "flat" if fms <= pms else "perblock"})
    print(json.dumps(rows, indent=1))
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
