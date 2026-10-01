# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""bench_br: levels 10M->2.5M->600K->120K->20K in ONE reused workspace pair
vs separate buffers per level. Measures allocations (tracemalloc peak),
RSS delta, time. Golden exact (counts.sum()==N, ukeys==reference) every level.
Usage: python -m tests.research.buffer_reuse.bench_br [--out rows.json]
"""
import gc
import json
import sys
import time
import tracemalloc

import numpy as np

LEVELS = [10_000_000, 2_500_000, 600_000, 120_000, 20_000]

try:
    import psutil

    def rss():
        return psutil.Process().memory_info().rss
except Exception:
    import resource

    def rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def synth(n, seed):
    rng = np.random.default_rng(seed)
    k = np.sort(rng.integers(0, n * 2, size=n, dtype=np.int64))
    return k


def reference(k):
    uk, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    return uk, cnt


def run_reused(level_data):
    from .workspace import Workspace, compact_keys_counts_inplace
    t0 = time.perf_counter()
    tracemalloc.start()
    ws_k = Workspace(LEVELS[0])
    ws_c = Workspace(LEVELS[0])
    alloc0 = tracemalloc.get_traced_memory()[1]
    rss0 = rss()
    total_ms = 0.0
    out = []
    for i, k in enumerate(level_data):
        t1 = time.perf_counter()
        uk, cc = compact_keys_counts_inplace(k, ws_k, ws_c)
        total_ms += (time.perf_counter() - t1) * 1000.0
        out.append((uk, cc))
    peak = tracemalloc.get_traced_memory()[1] - alloc0
    tracemalloc.stop()
    wall = (time.perf_counter() - t0) * 1000.0
    return out, {"mode": "reused", "op_ms": round(total_ms, 2),
                 "wall_ms": round(wall, 2),
                 "tracemalloc_peak_bytes": peak,
                 "rss_delta_bytes": int(rss() - rss0), "allocs": 2}


def run_separate(level_data):
    t0 = time.perf_counter()
    tracemalloc.start()
    alloc0 = tracemalloc.get_traced_memory()[1]
    rss0 = rss()
    total_ms = 0.0
    out = []
    for k in level_data:
        t1 = time.perf_counter()
        uk, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
        total_ms += (time.perf_counter() - t1) * 1000.0
        out.append((uk, cnt))
    peak = tracemalloc.get_traced_memory()[1] - alloc0
    tracemalloc.stop()
    wall = (time.perf_counter() - t0) * 1000.0
    n_alloc = 3 * len(level_data)
    return out, {"mode": "separate", "op_ms": round(total_ms, 2),
                 "wall_ms": round(wall, 2),
                 "tracemalloc_peak_bytes": peak,
                 "rss_delta_bytes": int(rss() - rss0), "allocs": n_alloc}


def run_reused_njit(level_data):
    from .workspace import HAS_NUMBA, Workspace, compact_njit
    assert HAS_NUMBA, "njit variant needs numba"
    # warmup outside timers
    _a = np.array([1, 1, 2], dtype=np.int64)
    _b = np.empty(3, dtype=np.int64)
    compact_njit(_a, _b, 3)
    gc.collect()
    t0 = time.perf_counter()
    tracemalloc.start()
    alloc0 = tracemalloc.get_traced_memory()[1]
    ws_k = Workspace(LEVELS[0])
    ws_c = Workspace(LEVELS[0])
    rss0 = rss()
    total_ms = 0.0
    out = []
    for k in level_data:
        n = len(k)
        ws_k.buf[:n] = k
        t1 = time.perf_counter()
        u = int(compact_njit(ws_k.buf, ws_c.buf, n))
        total_ms += (time.perf_counter() - t1) * 1000.0
        ws_k.length, ws_c.length = u, u
        out.append((ws_k.view().copy(), ws_c.view().copy()))
    peak = tracemalloc.get_traced_memory()[1] - alloc0
    tracemalloc.stop()
    wall = (time.perf_counter() - t0) * 1000.0
    return out, {"mode": "reused_njit", "op_ms": round(total_ms, 2),
                 "wall_ms": round(wall, 2),
                 "tracemalloc_peak_bytes": peak,
                 "rss_delta_bytes": int(rss() - rss0), "allocs": 2}


def main():
    from .workspace import impossible_cases
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    gc.collect()
    level_data = [synth(n, 42 + i) for i, n in enumerate(LEVELS)]
    refs = [reference(k) for k in level_data]
    ro, ri = run_reused(level_data)
    gc.collect()
    so, si = run_separate(level_data)
    gc.collect()
    no, ni = run_reused_njit(level_data)
    for i, ((uk, cc), (rk, rc)) in enumerate(zip(ro, refs)):
        assert np.array_equal(uk, rk), f"reused L{i} keys"
        assert np.array_equal(cc, rc), f"reused L{i} counts"
        assert int(cc.sum()) == LEVELS[i], f"reused L{i} sum"
    for i, ((uk, cc), (rk, rc)) in enumerate(zip(so, refs)):
        assert np.array_equal(uk, rk), f"separate L{i} keys"
    for i, ((uk, cc), (rk, rc)) in enumerate(zip(no, refs)):
        assert np.array_equal(uk, rk), f"njit L{i} keys"
        assert np.array_equal(cc, rc), f"njit L{i} counts"
        assert int(cc.sum()) == LEVELS[i], f"njit L{i} sum"
    rows = [ri, ni, si, {"impossible": impossible_cases(), "correct": True,
                         "levels": LEVELS}]
    print(json.dumps(rows, indent=1))
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
