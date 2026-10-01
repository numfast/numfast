# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research bench: OLD vs F1 (nocopy bincount) vs F2 (numba single-pass).

Synthetic N=10M seed 42, M in (100, 10K, 100K, 1M), shapes Q1/Q3/Q4/Q5-like.
Correctness: exact vs ref_unique every cell. No prod change.
"""
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK / "tests" / "research" / "singlepass"))

import numpy as np
from kernels_sp import (dict_fast_multi, dict_fast_single, dict_loop, f1_nocopy,
                        fused_singlepass, old_mirror, ref_unique)

N = 10_000_000
SEED = 42


def timed(fn, reps=3):
    best = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        dt = (time.perf_counter() - t) * 1000
        best = dt if best is None else min(best, dt)
    return out, best


def check(a, b, label):
    uka, ca, sa = a
    ukb, cb, sb = b
    assert bool((uka == ukb).all()), f"{label}: ukeys differ"
    assert bool((ca == cb).all()), f"{label}: counts differ"
    for x, y in zip(sa, sb):
        assert x.dtype == y.dtype, f"{label}: dtype {x.dtype} vs {y.dtype}"
        assert bool((x == y).all()), f"{label}: sums differ"


def main():
    import kernels_sp as K
    assert K._HAS_NUMBA, "numba required for F2"
    # warmup numba for all sigs on tiny data
    kk = np.array([0, 1, 0], dtype=np.int32)
    vv = np.array([1, 2, 3], dtype=np.int32)
    vf = np.array([1.0, 2.0, 3.0])
    for cols in ([vv], [vf], [vv, vv], [vv, vf], [vv, vv, vv],
                 [vv, vv, vf], [vv, vf, vf], [vf, vf, vf]):
        fused_singlepass(kk, cols, 2)
    print("numba warmup done", flush=True)

    shapes = {
        "Q1-like(1i-sum)": ([("i", np.int32)], [("sum",)]),
        "Q3-like(1i-s+1f-m)": ([("i", np.int32), ("f", np.float64)],
                               [("sum",), ("mean",)]),
        "Q4-like(2i-m+1f-m)": ([("i", np.int32), ("i", np.int32), ("f", np.float64)],
                               [("mean",), ("mean",), ("mean",)]),
        "Q5-like(3i-sum)": ([("i", np.int32)] * 3,
                            [("sum",), ("sum",), ("sum",)]),
    }
    for m in (100, 10_000, 100_000, 1_000_000):
        rng = np.random.default_rng(SEED)
        keys = rng.integers(0, m, size=N).astype(np.int32)
        iv = rng.integers(0, 1000, size=N).astype(np.int32)
        fv = (rng.random(N) * 1000).astype(np.float64)
        pool = {"i": iv, "f": fv}
        print(f"--- M={m} ---", flush=True)
        for name, (kinds, ops) in shapes.items():
            vcols = [pool[k].copy() if False else pool[k] for k, _dt in kinds]
            # note: share buffers across variants (read-only); copies only inside variants
            ref = ref_unique(keys, vcols, m)
            _, t_old = timed(lambda: old_mirror(keys, vcols, m))
            got_f1, t_f1 = timed(lambda: f1_nocopy(keys, vcols, m))
            got_f2, t_f2 = timed(lambda: fused_singlepass(keys, vcols, m))
            check(ref, got_f1, f"F1 {name} M={m}")
            check(ref, got_f2, f"F2 {name} M={m}")
            print(f"{name}: OLD={t_old:.0f} F1={t_f1:.0f} ({t_old/t_f1:.2f}x) "
                  f"F2={t_f2:.0f} ({t_old/t_f2:.2f}x)", flush=True)

    # dict materialization at 100K groups
    rng = np.random.default_rng(SEED)
    ng = 100_000
    uk = np.arange(ng, dtype=np.int64)
    co = np.ones(ng, dtype=np.int64)
    s1 = rng.integers(0, 10**6, size=ng).astype(np.int64)
    s2 = rng.integers(0, 10**6, size=ng).astype(np.int64)
    sf = (rng.random(ng) * 1000).astype(np.float64)
    r_loop, t_loop = timed(lambda: dict_loop(uk, co, [s1, s2, sf],
                                             [("sum",), ("sum",), ("sum",)]))
    r_fast, t_fast = timed(lambda: dict_fast_multi(uk, co, [s1, s2, sf],
                                                   [("sum",), ("sum",), ("sum",)]))
    # normalize key shapes for equality (dict_loop uses ci ints as subkeys too)
    assert r_loop == r_fast, "multi dict mismatch"
    print(f"dict-multi 100K 3col: loop={t_loop:.0f} fast={t_fast:.0f} "
          f"({t_loop/t_fast:.2f}x)", flush=True)
    r1, t1 = timed(lambda: dict_loop(uk, co, [s1], [("sum", "count", "mean")]))
    r2, t2 = timed(lambda: dict_fast_single(uk, co, s1, ("sum", "count", "mean")))
    assert r1 == r2, "single dict mismatch"
    print(f"dict-single 100K: loop={t1:.0f} fast={t2:.0f} ({t1/t2:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
