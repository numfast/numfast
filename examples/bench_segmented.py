#!/usr/bin/env python3
# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Microbench P1/P2 (a NEW file; the existing benchmarks are untouched).

OLD: the naive python-loop / groupby Sort->Scan->Reduce chain.
NEW: direct segmented_reduce / adjacency_slice.
Metrics: correctness max_diff first, then TIME_RATIO, throughput, memory, and a
stage breakdown. Fixed seed 42.

KNOWN FAILURE, measured 2026-10-05 at 597d2ec: this script ends on
`assert fdiff == 0.0`, and the f32 lane prints `max_diff 0.03515625`. The int32
lane matches the naive chain exactly (`max_diff 0.0`). The run aborts on that
assertion, so nothing after it is produced. Recorded rather than fixed: the fix
is either the f32 tolerance or the segmented f32 accumulation, and neither is a
documentation change. See ../examples/README.md.
"""
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Relational.Segmented._lib.adjacency import (  # noqa: E402
    adjacency_flat,
    adjacency_slice,
)
from Relational.Segmented._lib.segmented import segmented_reduce  # noqa: E402

rng = np.random.default_rng(42)
N = 200_000
M = 2_000
cuts = np.sort(rng.integers(0, N + 1, size=M - 1))
bounds = np.unique(np.concatenate(([0], cuts, [N]))).astype(np.uint32)
M = int(bounds.size) - 1
vals_i = rng.integers(-1000, 1000, size=N).astype(np.int32)
vals_f = (rng.random(N).astype(np.float32) * 100).astype(np.float32)


def old_loop_sum_i32(v, b):
    out = np.empty(b.size - 1, dtype=np.int32)
    for i in range(b.size - 1):
        out[i] = np.int32(
            max(-2147483648, min(2147483647, int(v[b[i]:b[i + 1]].sum(
                dtype=np.int64)))))
    return out


def old_groupby_chain(v, b):
    segs = np.repeat(np.arange(b.size - 1), np.diff(b.astype(np.int64)))
    order = np.argsort(segs, kind="stable")
    sv = v[order]
    ss = segs[order]
    ch = np.empty(ss.size, dtype=bool)
    if ss.size == 0:
        return np.zeros(0, dtype=np.int32)
    ch[0] = True
    ch[1:] = ss[1:] != ss[:-1]
    starts = np.flatnonzero(ch)
    sums = np.add.reduceat(sv.astype(np.int64), starts)
    return np.clip(sums, -2147483648, 2147483647).astype(np.int32)


def bench_fn(fn, *a, warm=2, iters=5):
    for _ in range(warm):
        fn(*a)
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*a)
        best = min(best, (time.perf_counter() - t0) * 1000)
    return best, out


def mem_of(fn, *a):
    tracemalloc.start()
    fn(*a)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024


print(f"n={N} m={M} seed=42")
# correctness first
ref = old_loop_sum_i32(vals_i, bounds)
new = segmented_reduce(vals_i, bounds, "sum")
max_diff = float(np.max(np.abs(ref.astype(np.int64)
                               - new.astype(np.int64)))) if M else 0.0
print(f"correctness max_diff i32-sum: {max_diff}")
assert max_diff == 0.0
ref_f = np.add.reduceat(vals_f, bounds[:-1].astype(np.int64))
ref_f[np.diff(bounds.astype(np.int64)) == 0] = 0.0
new_f = segmented_reduce(vals_f, bounds, "sum")
fdiff = float(np.max(np.abs(ref_f - new_f))) if M else 0.0
print(f"correctness max_diff f32-sum: {fdiff}")
assert fdiff == 0.0

t_old, _ = bench_fn(old_loop_sum_i32, vals_i, bounds)
t_chain, _ = bench_fn(old_groupby_chain, vals_i, bounds)
# stage breakdown OLD chain: sort + scan(detect) + reduce
t0 = time.perf_counter()
segs = np.repeat(np.arange(M), np.diff(bounds.astype(np.int64)))
t_repeat = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
order = np.argsort(segs, kind="stable")
t_sort = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
sv = vals_i[order]
ss = segs[order]
ch = np.ones(ss.size, dtype=bool)
ch[0] = True
ch[1:] = ss[1:] != ss[:-1]
st = np.flatnonzero(ch)
t_scan = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
np.add.reduceat(sv.astype(np.int64), st)
t_reduce = (time.perf_counter() - t0) * 1000
# NEW breakdown: validate + dispatch
t0 = time.perf_counter()
np.diff(bounds.astype(np.int64))
t_valid = (time.perf_counter() - t0) * 1000
t_new, _ = bench_fn(segmented_reduce, vals_i, bounds, "sum")
mem_old = mem_of(old_loop_sum_i32, vals_i, bounds)
mem_new = mem_of(segmented_reduce, vals_i, bounds, "sum")
ratio = t_old / t_new if t_new else float("inf")
thr_old = N / (t_old / 1000) / 1e6
thr_new = N / (t_new / 1000) / 1e6
print(f"OLD loop: {t_old:.2f}ms thr={thr_old:.2f}M/s mem={mem_old:.1f}KiB")
print(f"OLD chain Sort>Scan>Reduce: {t_chain:.2f}ms "
      f"[repeat={t_repeat:.2f} sort={t_sort:.2f} scan={t_scan:.2f} "
      f"reduce={t_reduce:.2f}]")
print(f"NEW direct: {t_new:.2f}ms thr={thr_new:.2f}M/s mem={mem_new:.1f}KiB "
      f"[validate~{t_valid:.2f} dispatch={t_new - t_valid:.2f}]")
print(f"TIME_RATIO OLD/NEW: {ratio:.2f}x")

# P2 smoke bench
V, DEG, K = 50_000, 4, 10_000
indptr = (np.arange(V + 1, dtype=np.int64) * DEG).astype(np.uint32)
indices = np.tile(np.arange(DEG, dtype=np.uint32), V)
q = rng.integers(0, V, size=K).astype(np.uint32)
t0 = time.perf_counter()
for _ in range(3):
    bb, ee = adjacency_slice(indptr, indices, q)
t_p2 = (time.perf_counter() - t0) / 3 * 1000
flat = adjacency_flat(indices, bb, ee)
assert int((ee - bb).sum()) == flat.size == K * DEG
print(f"P2 adjacency_slice k={K}: {t_p2:.2f}ms flat={flat.size}")
