# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""bench_micro: A-F x 12-block suite, serial. Golden verify every cell +
multi-block merge verify (D linear / E tree / B-C-F funnel bit-exactness).
Usage: python -m tests.research.groupby_lab.bench_micro [--out results.json]
Fast chunk (<2 min). Exit nonzero on any golden failure.
"""

import json
import sys
import time

import numpy as np

from .blockgen import BLOCK_SUITE, M_SWEEP, PATTERNS, gen_block, make_suite
from .candidates import ALGOS
from .runners import merge_all
from .verifier import verify


def main():
    t_all = time.perf_counter()
    suite = make_suite()
    for _mod in ALGOS.values():  # JIT warmup, outside timers
        _s, _ = _mod.local(suite[0]["keys"][:1024], suite[0]["vals"][:1024])
        _s2, _ = _mod.local(suite[0]["keys"][1024:2048], suite[0]["vals"][1024:2048])
        _mod.merge([_s, _s2])
    rows = []
    fails = 0
    # single-block cells
    for b in suite:
        for name, mod in ALGOS.items():
            t0 = time.perf_counter()
            st, s = mod.local(b["keys"], b["vals"])
            fin, info = merge_all(mod, [st])
            wall = (time.perf_counter() - t0) * 1000.0
            try:
                chk = verify(fin, b["keys"], b["vals"], label=f"{name}/{b['name']}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}/{b['name']}: {e}", flush=True)
                continue
            extra = {k: v for k, v in s.items() if k != "local_ms"}
            if name == "B":
                from .candidates.b_partial import describe as _desc
                extra.update(_desc(b["keys"], s["partial_rows"]))
            rows.append({
                "algo": name, "regime": b["name"], "n": b["n"], "M": b["M"],
                "pattern": b["pattern"], "level": b["level"],
                "ms": wall, "local_ms": s.get("local_ms", 0.0),
                "merge_ms": info.get("merge_ms", 0.0),
                "bytes": chk["state_bytes"], "ngroups": chk["ngroups"],
                "extra": extra,
            })
            print(f"ok {name:1s} {b['name']:24s} {wall:8.2f}ms "
                  f"groups={chk['ngroups']:6d} bytes={chk['state_bytes']}", flush=True)
    # multi-block merge exactness: 4x64K uniform, all algos vs concatenated golden
    rng = np.random.default_rng(42)
    mblocks = []
    for i in range(4):
        k, v = gen_block(65_536, 1_000_000, "uniform", rng)
        mblocks.append((k, v))
    big_k = np.concatenate([k for k, _ in mblocks])
    big_v = np.concatenate([v for _, v in mblocks])
    for name, mod in ALGOS.items():
        parts = [mod.local(k, v)[0] for k, v in mblocks]
        fin, info = merge_all(mod, parts)
        try:
            verify(fin, big_k, big_v, label=f"{name}/merge4")
            print(f"ok {name:1s} merge4 comps={info.get('comparisons')} "
                  f"rounds={info.get('rounds')}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"FAIL {name}/merge4: {e}", flush=True)
    el = time.perf_counter() - t_all
    print(f"micro done: {len(rows)} cells, fails={fails}, elapsed={el:.1f}s", flush=True)
    if "--out" in sys.argv:
        p = sys.argv[sys.argv.index("--out") + 1]
        with open(p, "w") as f:
            json.dump(rows, f, indent=1, default=str)
        print(f"wrote {p}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
