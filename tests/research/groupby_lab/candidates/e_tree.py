# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""E BALANCED REDUCTION TREE: same local as A (full unique per block), merge
via balanced binary tree of golden pair-funnels. Wide rounds merged in a
thread pool (order-preserving); result bit-identical to direct funnel.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..contract import empty_state, merge_states
from .a_full_unique import local

NAME = "E_balanced_tree"


def merge(states, workers=4):
    t0 = time.perf_counter()
    cur = [s for s in states if s.ngroups > 0]
    if not cur:
        return empty_state(), {"merge_ms": 0.0, "rounds": 0, "pair_merges": 0}
    rounds = 0
    pairs = 0
    comps = 0
    while len(cur) > 1:
        idx = [(cur[i], cur[i + 1]) for i in range(0, len(cur) - 1, 2)]
        odd = cur[-1] if len(cur) % 2 else None
        if len(idx) >= 4 and workers and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(lambda p: merge_states(p), idx))
        else:
            res = [merge_states(p) for p in idx]
        pairs += len(idx)
        nxt = []
        for st, info in res:
            comps += info["comparisons"]
            nxt.append(st)
        if odd is not None:
            nxt.append(odd)
        cur = nxt
        rounds += 1
    ms = (time.perf_counter() - t0) * 1000.0
    return cur[0], {"merge_ms": ms, "rounds": rounds, "pair_merges": pairs,
                    "comparisons": comps, "tree": True}
