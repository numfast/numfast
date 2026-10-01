# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""bench_10M: full-scheme validation, 10M rows, one algo+pattern per chunk.
Usage: python -m tests.research.groupby_lab.bench_10M ALGO PATTERN [M] [OUT]
Golden-verified whole. Patterns: uniform / skewed. M default 10M.
~80x128K blocks serial local, then one funnel merge.
"""

import json
import sys
import time

import numpy as np

from .blockgen import gen_block
from .candidates import get
from .runners import cpu_sum_ms, merge_all, run_serial, run_threaded
from .verifier import verify

N_TOTAL = 10_000_000
BLOCK = 131_072


def main(argv):
    algo, pattern = argv[0], argv[1]
    M = int(argv[2]) if len(argv) > 2 else 10_000_000
    out = argv[3] if len(argv) > 3 else None
    workers = int(argv[4]) if len(argv) > 4 else 8
    mod = get(algo)
    rng = np.random.default_rng(0)  # JIT warmup, outside timers
    _k = rng.integers(0, 1000, size=1024).astype(np.int32)
    _v = rng.integers(-9, 9, size=1024).astype(np.int32)
    _s, _ = mod.local(_k, _v)
    _s2, _ = mod.local(_k[::-1].copy(), _v)
    mod.merge([_s, _s2])
    nblk = (N_TOTAL + BLOCK - 1) // BLOCK
    t0 = time.perf_counter()
    blocks = []
    for i in range(nblk):
        rng = np.random.default_rng(9000 + i)
        n = BLOCK if (i + 1) * BLOCK <= N_TOTAL else N_TOTAL - i * BLOCK
        k, v = gen_block(n, M, pattern, rng)
        blocks.append({"keys": k, "vals": v})
    gen_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    states, stats, wall = run_threaded(mod, blocks, workers)
    fin, info = merge_all(mod, states)
    el = (time.perf_counter() - t0) * 1000.0
    big_k = np.concatenate([b["keys"] for b in blocks])
    big_v = np.concatenate([b["vals"] for b in blocks])
    chk = verify(fin, big_k, big_v, label=f"{algo}/10M/{pattern}")
    cpu = cpu_sum_ms(stats)
    print(f"ok {algo} 10M/{pattern} M={M}: gen={gen_ms:.0f}ms local_wall={wall:.0f}ms "
          f"cpu={cpu:.0f}ms su={cpu / max(wall, 1e-9):.2f}x merge={info.get('merge_ms', 0):.0f}ms "
          f"total={el:.0f}ms groups={chk['ngroups']} "
          f"thr={N_TOTAL / max(el / 1e3, 1e-9) / 1e6:.2f}M rows/s", flush=True)
    if out:
        with open(out, "w") as f:
            json.dump({"algo": algo, "regime": f"10M/{pattern}", "M": M,
                       "ms": el, "local_wall_ms": wall, "cpu_ms": cpu,
                       "merge_ms": info.get("merge_ms", 0.0),
                       "bytes": chk["state_bytes"], "ngroups": chk["ngroups"],
                       "throughput_rows_s": N_TOTAL / max(el / 1e3, 1e-9)}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
