# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""F FUSED scatter-direct: one pass computes pid from the key and updates
the owning partition accumulator. No pid/key/val intermediate buffers.

T=1: single njit call over (P x W) accumulators.
T>1: row-blocks, one (P x W) replica per thread (njit releases GIL),
     merge = replica sum (vectorized) + per-partition compact.
Merge for shift/scaled/div = concat (disjoint ordered ranges).
Merge for mask (strided) = concat + argsort (its honest extra price).
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .contract import State, empty_state, info
from .golden import key_of_vec
from .kernels import (HAS_NUMBA, fused_div, fused_mask, fused_scaled,
                      fused_shift)

NAME = "F_fused_direct"

VARIANTS = ("shift", "scaled", "div", "mask")


def params(variant, g, p):
    if variant == "shift":
        s = max(1, int(math.ceil(math.log2(max(2, (g + p - 1) // p)))))
        return {"shift": s, "width": 1 << s}
    if variant == "scaled":
        lo = (np.arange(p, dtype=np.int64) * g + p - 1) // p
        hi = ((np.arange(p, dtype=np.int64) + 1) * g + p - 1) // p
        return {"lo": np.ascontiguousarray(lo), "hi": np.ascontiguousarray(hi),
                "width": int((hi - lo).max())}
    if variant == "div":
        ws = (g + p - 1) // p
        return {"ws": ws, "width": ws}
    if variant == "mask":
        lb = int(math.log2(p))
        assert (1 << lb) == p, "mask needs P=2**lb"
        return {"lbits": lb, "width": (g + p - 1) // p}
    raise ValueError(variant)


def _call(variant, kk, vv, sums, counts, par, p, g):
    if variant == "shift":
        fused_shift(kk, vv, sums, counts, par["shift"], par["width"], p)
    elif variant == "scaled":
        fused_scaled(kk, vv, sums, counts, par["lo"], p, g)
    elif variant == "div":
        fused_div(kk, vv, sums, counts, par["ws"], p)
    elif variant == "mask":
        fused_mask(kk, vv, sums, counts, par["lbits"], p)
    else:
        raise ValueError(variant)


def run(keys, vals, g, p=16, variant="shift", threads=1):
    if not HAS_NUMBA:
        raise RuntimeError("F needs numba")
    assert variant in VARIANTS, variant
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0.0, 0.0, 0.0, 0, 0,
                                   {"P": p, "variant": variant})
    assert int(k.min()) >= 0 and int(k.max()) < g, "keys out of [0,G)"
    par = params(variant, g, p)
    w = par["width"]
    t0 = time.perf_counter()
    part_ms = (time.perf_counter() - t0) * 1000.0  # pid fused: nothing staged
    t1 = time.perf_counter()
    if threads == 1:
        sums = np.zeros((p, w), dtype=np.int64)
        counts = np.zeros((p, w), dtype=np.int64)
        _call(variant, k, v, sums, counts, par, p, g)
        gs, gc = sums, counts
        rep_bytes = 0
    else:
        bounds = np.linspace(0, n, threads + 1).astype(np.int64)
        sums3 = np.zeros((threads, p, w), dtype=np.int64)
        counts3 = np.zeros((threads, p, w), dtype=np.int64)

        def _blk(t):
            a, b = int(bounds[t]), int(bounds[t + 1])
            _call(variant, k[a:b], v[a:b], sums3[t], counts3[t], par, p, g)

        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(_blk, range(threads)))
        rep_bytes = int(sums3.nbytes + counts3.nbytes)
        gs, gc = None, None  # replica-sum timed as merge below
    agg_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    if threads > 1:
        gs = sums3.sum(axis=0)
        gc = counts3.sum(axis=0)
    uk, ss, cc = [], [], []
    for q in range(p):
        nz = np.flatnonzero(gc[q] > 0)
        if nz.size:
            uk.append(key_of_vec(variant, par,
                                 np.full(nz.shape, q, dtype=np.int64),
                                 nz.astype(np.int64), g, p))
            ss.append(gs[q][nz])
            cc.append(gc[q][nz])
    ukeys = np.concatenate(uk) if uk else np.empty(0, dtype=np.int64)
    sums_f = np.concatenate(ss) if ss else np.empty(0, dtype=np.int64)
    counts_f = np.concatenate(cc) if cc else np.empty(0, dtype=np.int64)
    if variant == "mask" and ukeys.size:
        o = np.argsort(ukeys, kind="stable")
        ukeys, sums_f, counts_f = ukeys[o], sums_f[o], counts_f[o]
    st = State(ukeys, sums_f, counts_f) if ukeys.size else empty_state()
    merge_ms = (time.perf_counter() - t2) * 1000.0
    tmp_b = int(gs.nbytes + gc.nbytes + rep_bytes)
    traffic = int(k.nbytes + v.nbytes + 8 * p * w * 2 + 8 * st.ngroups * 2)
    return st, info(part_ms, agg_ms, merge_ms, tmp_b, traffic,
                    {"P": p, "variant": variant, "threads": threads,
                     "width": w, "ws_bytes": int(8 * w * 2)})
