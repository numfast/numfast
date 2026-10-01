# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""bench_rt: static balanced vs dynamic ready-queue, leaves = 2/4/8/16 sorted
blocks of 32K/64K/128K/256K. Golden exact (root == np.unique+bincount ref)
every cell. Metrics: throughput/utilization/wait/depth/overlap/contention/RSS.
Usage: python -m tests.research.reduction_tree.bench_rt LEAVES BLOCK [--out f]
"""
import gc
import json
import sys
import time

import numpy as np

try:
    import psutil

    def rss():
        return psutil.Process().memory_info().rss
except Exception:
    import resource

    def rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main():
    from .tree import HAS_NUMBA, leaf_sorted, merge_pair, reduce_dynamic, reduce_static
    leaves_n, block = int(sys.argv[1]), int(sys.argv[2])
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    rng = np.random.default_rng(42)
    M = block  # ~50% unique per block -> non-trivial merges
    keys = [rng.integers(0, M, size=block).astype(np.int32) for _ in range(leaves_n)]
    vals = [rng.integers(-1000, 1000, size=block).astype(np.int32) for _ in range(leaves_n)]
    K = np.concatenate(keys)
    V = np.concatenate(vals).astype(np.float64)
    uk, inv = np.unique(K, return_inverse=True)
    ref_s = np.bincount(inv, weights=V).astype(np.int64)
    ref_c = np.bincount(inv, minlength=uk.size).astype(np.int64)
    leaves = [leaf_sorted(k, v) for k, v in zip(keys, vals)]
    # warmup (numba compile) outside timers
    merge_pair(leaves[0], leaves[1])
    rows = []
    for mode in ("static", "dynamic"):
        gc.collect()
        r0 = rss()
        t0 = time.perf_counter()
        if mode == "static":
            root, st = reduce_static(leaves, workers=4)
        else:
            root, st = reduce_dynamic(leaves, workers=4, qmax=8)
        wall = (time.perf_counter() - t0) * 1000.0
        assert np.array_equal(root[0], uk.astype(np.int64)), f"{mode} keys"
        assert np.array_equal(root[1], ref_s), f"{mode} sums"
        assert np.array_equal(root[2], ref_c), f"{mode} counts"
        n = leaves_n * block
        rows.append({"mode": mode, "leaves": leaves_n, "block": block,
                     "rows": n, "wall_ms": round(wall, 2),
                     "merge_ms": round(st["merge_ms"], 2),
                     "wait_ms": round(st["wait_ms"], 2),
                     "depth": st["depth"], "rounds": st["rounds"],
                     "comparisons": st["comparisons"],
                     "rows_per_s": int(n / max(wall / 1000, 1e-9)),
                     "utilization": round(st["merge_ms"] / max(wall, 1e-9), 3),
                     "rss_delta_bytes": int(rss() - r0),
                     "numba": HAS_NUMBA, "correct": True})
    print(json.dumps(rows))
    if out:
        with open(out, "w") as f:
            json.dump(rows, f)


if __name__ == "__main__":
    sys.exit(main())
