# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""bench_scale: argv-driven chunks (one regime per invocation, each <3 min).

  threads ALGO BLOCK [NBLK] [OUT]   serial,1,2,4,8,16 workers over NBLK blocks
  dist ALGO PATTERN [N] [OUT]        M sweep (100..10M) serial, fixed N rows
  hier ALGO [FANIN] [OUT]            L0(8x64K mix)->L1->L2 hierarchy
  pipe ALGO [NWORKERS] [QMAX] [OUT]  bounded-queue pipeline over 16x64K

Every chunk golden-verified vs concatenated input. Exit nonzero on failure.
"""

import json
import sys
import time

import numpy as np

from .blockgen import M_SWEEP, gen_block, make_suite
from .candidates import get
from .runners import (cpu_sum_ms, hierarchical, merge_all, pipelined, run_serial,
                      run_threaded)
from .verifier import verify


def _warmup(mod):
    """JIT warmup (numba cache=False compiles per-process). Outside timers."""
    import numpy as np
    rng = np.random.default_rng(0)
    k = rng.integers(0, 1000, size=1024).astype(np.int32)
    v = rng.integers(-9, 9, size=1024).astype(np.int32)
    s, _ = mod.local(k, v)
    s2, _ = mod.local(k[::-1].copy(), v)
    mod.merge([s, s2])


def _blocks(n, M, pattern, count, seed):
    out = []
    for i in range(count):
        rng = np.random.default_rng(seed + i)
        k, v = gen_block(n, M, pattern, rng)
        out.append({"keys": k, "vals": v})
    return out


def _concat(blocks):
    return (np.concatenate([b["keys"] for b in blocks]),
            np.concatenate([b["vals"] for b in blocks]))


def cmd_threads(algo, blockname, nblk=16, out=None):
    mod = get(algo)
    _warmup(mod)
    tpl = next(b for b in make_suite() if b["name"] == blockname)
    blocks = _blocks(tpl["n"], tpl["M"], tpl["pattern"], nblk, seed=1000)
    big_k, big_v = _concat(blocks)
    rows = []
    for mode in ["serial", 1, 2, 4, 8, 16]:
        t0 = time.perf_counter()
        if mode == "serial":
            states, stats, wall = run_serial(mod, blocks)
        else:
            states, stats, wall = run_threaded(mod, blocks, mode)
        fin, info = merge_all(mod, states)
        el = (time.perf_counter() - t0) * 1000.0
        verify(fin, big_k, big_v, label=f"{algo}/T{mode}/{blockname}")
        cpu = cpu_sum_ms(stats)
        rows.append({"algo": algo, "regime": f"T{mode}/{blockname}",
                     "mode": mode, "local_wall_ms": wall, "cpu_ms": cpu,
                     "merge_ms": info.get("merge_ms", 0.0),
                     "ms": el, "bytes": int(fin.state_bytes()),
                     "speedup": cpu / max(wall, 1e-9)})
        print(f"ok {algo} T={mode!s:6s} wall={wall:8.1f}ms cpu={cpu:8.1f}ms "
              f"su={cpu / max(wall, 1e-9):.2f}x merge={info.get('merge_ms', 0):.1f}ms",
              flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=1, default=str)
    return 0


def cmd_dist(algo, pattern, n=65_536, out=None):
    mod = get(algo)
    _warmup(mod)
    rows = []
    for M in M_SWEEP:
        rng = np.random.default_rng(42 + M % 100003)
        k, v = gen_block(n, M, pattern, rng)
        t0 = time.perf_counter()
        st, s = mod.local(k, v)
        fin, info = merge_all(mod, [st])
        el = (time.perf_counter() - t0) * 1000.0
        chk = verify(fin, k, v, label=f"{algo}/M{M}/{pattern}")
        rows.append({"algo": algo, "regime": f"M{M}/{pattern}", "M": M,
                     "pattern": pattern, "ms": el,
                     "local_ms": s.get("local_ms", 0.0),
                     "bytes": chk["state_bytes"], "ngroups": chk["ngroups"],
                     "extra": {kk: vv for kk, vv in s.items() if kk != "local_ms"}})
        print(f"ok {algo} M={M:<10d} {pattern:9s} {el:8.2f}ms "
              f"groups={chk['ngroups']:7d} path={s.get('path', '-')} "
              f"uniq={chk['ngroups'] / max(n, 1):.3f}", flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=1, default=str)
    return 0


def cmd_hier(algo, fanin=4, out=None):
    mod = get(algo)
    _warmup(mod)
    specs = [(65_536, 1_000_000, "uniform"), (65_536, 100_000, "skewed"),
              (65_536, 1_000_000, "clustered"), (65_536, 100_000, "sorted"),
              (65_536, 100, "uniform"), (65_536, 70_000, "unique95"),
              (131_072, 10_000_000, "uniform"), (65_536, 1_000_000, "random")]
    blocks = []
    for i, (n, M, p) in enumerate(specs):
        rng = np.random.default_rng(7 + i)
        k, v = gen_block(n, M, p, rng)
        blocks.append({"keys": k, "vals": v})
    h = hierarchical(mod, blocks, fanin=fanin)
    big_k, big_v = _concat(blocks)
    verify(h["final"], big_k, big_v, label=f"{algo}/hier")
    print(f"ok {algo} hier: raw={h['raw_bytes']} L1={h['l1_bytes']} "
          f"({h['l1_reduction']:+.1%}) L2={h['l2_bytes']} ({h['l2_reduction']:+.1%}) "
          f"final={h['final_bytes']} total={h['total_reduction']:+.1%} "
          f"stages meta={h['l0_meta_ms']:.1f} L1={h['l1_ms']:.1f} "
          f"L2={h['l2_ms']:.1f} F={h['final_ms']:.1f}ms", flush=True)
    if out:
        row = {"algo": algo, "regime": "hier8",
               "ms": h["l1_ms"] + h["l2_ms"] + h["final_ms"],
               "bytes": h["final_bytes"], **{k: v for k, v in h.items()
               if k in ("l1_reduction", "l2_reduction", "total_reduction",
                        "l1_ms", "l2_ms", "final_ms", "raw_bytes",
                        "l1_bytes", "l2_bytes", "final_bytes")}}
        with open(out, "w") as f:
            json.dump(row, f, indent=1, default=str)
    return 0


def cmd_pipe(algo, nworkers=4, qmax=4, out=None):
    mod = get(algo)
    _warmup(mod)
    blocks = _blocks(65_536, 1_000_000, "uniform", 16, seed=5000)
    big_k, big_v = _concat(blocks)
    r = pipelined(mod, blocks, nworkers=nworkers, qmax=qmax)
    verify(r["final"], big_k, big_v, label=f"{algo}/pipe{nworkers}q{qmax}")
    print(f"ok {algo} pipe W={nworkers} Q={qmax}: wall={r['wall_ms']:.1f}ms "
          f"compute={r['compute_sum_ms']:.1f} merge={r['merge_ms']:.1f} "
          f"overlap={r['overlap_ratio']:.1%} idle={r['worker_idle_ms']:.1f} "
          f"put={r['put_ms']:.1f} thr={r['throughput_rows_s'] / 1e6:.2f}M rows/s",
          flush=True)
    if out:
        row = {"algo": algo, "regime": f"pipe{nworkers}q{qmax}",
               "ms": r["wall_ms"], "bytes": int(r["final"].state_bytes()),
               **{k: v for k, v in r.items() if k != "final"}}
        with open(out, "w") as f:
            json.dump(row, f, indent=1, default=str)
    return 0


def main(argv):
    mode = argv[0]
    if mode == "threads":
        return cmd_threads(argv[1], argv[2],
                           int(argv[3]) if len(argv) > 3 else 16,
                           argv[4] if len(argv) > 4 else None)
    if mode == "dist":
        return cmd_dist(argv[1], argv[2],
                        int(argv[3]) if len(argv) > 3 else 65_536,
                        argv[4] if len(argv) > 4 else None)
    if mode == "hier":
        return cmd_hier(argv[1],
                        int(argv[2]) if len(argv) > 2 else 4,
                        argv[3] if len(argv) > 3 else None)
    if mode == "pipe":
        return cmd_pipe(argv[1],
                        int(argv[2]) if len(argv) > 2 else 4,
                        int(argv[3]) if len(argv) > 3 else 4,
                        argv[4] if len(argv) > 4 else None)
    print("usage: threads|dist|hier|pipe ...", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
