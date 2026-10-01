# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""C BLOCK-LOCAL control: row-chunks, full-G bincount per block, dense-add merge.

Merge is O(nb*G): the reason it loses at high G. Kept as checkpoint only."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .contract import State, empty_state, info

NAME = "C_blocklocal"


def _agg_block(args):
    kb, vb, g = args
    s = np.bincount(kb, weights=vb, minlength=g).astype(np.int64)
    c = np.bincount(kb, minlength=g).astype(np.int64)
    return s, c


def run(keys, vals, g, p=4, threads=1, block=None):
    k = np.asarray(keys).astype(np.int32, copy=False)
    v = np.asarray(vals).astype(np.int32, copy=False)
    n = k.size
    if n == 0:
        return empty_state(), info(0.0, 0.0, 0.0, 0, 0, {"nb": 0})
    bs = int(block or max(1, (n + p - 1) // p))
    nb = (n + bs - 1) // bs
    t0 = time.perf_counter()
    jobs = [(k[i * bs:(i + 1) * bs].astype(np.int64, copy=False),
             v[i * bs:(i + 1) * bs].astype(np.int64, copy=False), g)
            for i in range(nb)]
    part_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            res = list(ex.map(_agg_block, jobs))
    else:
        res = [_agg_block(j) for j in jobs]
    agg_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    gs = np.zeros(g, dtype=np.int64)
    gc = np.zeros(g, dtype=np.int64)
    for s, c in res:
        gs += s
        gc += c
    mask = gc > 0
    st = State(np.flatnonzero(mask).astype(np.int64), gs[mask], gc[mask])
    merge_ms = (time.perf_counter() - t2) * 1000.0
    tmp_b = int(nb * 8 * g * 2)
    traffic = int(k.nbytes + v.nbytes + nb * 8 * g * 2 + 8 * st.ngroups * 2)
    return st, info(part_ms, agg_ms, merge_ms, tmp_b, traffic,
                    {"nb": nb, "block": bs, "threads": threads,
                     "ws_bytes": int(8 * g * 2)})
