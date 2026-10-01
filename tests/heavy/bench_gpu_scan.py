# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU standalone Scan/Reduction bench (NEW file, bench-only).

Portable WGSL prefix scan (int32/uint32, Hillis-Steele 256 + fixup) and
standalone reduction (sum/min/max/count, exact integer lanes) on RTX 2060
(Vulkan, warm, seed 42). Compares:
  H hybrid scan (GPU block-scan + host W-prefix + GPU fixup, default)
  G resident scan (5 dispatches, zero host round trips, measurement variant)
  R reduce_full vs numpy (context, verbatim single-thread)
Stages per N: full transfer path (H2D+dispatches+D2H) vs transfer-only
(raw H2D+D2H, kernel = full - transfer). Prefix-replacement section: host
W-prefix round trip (what Filter pays today) vs resident GPU prefix
dispatches over W=39063 counts (10M rows) -> verdict for resident graphs.
N-scale 256K/1M/4M/10M. Integrity: exact match before timing (any diff = STOP).
No production code touched by this file; no IR/Planner changes.

Usage (Git Bash, sequential, timeout -- heavy jobs strictly serial):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 python tests/heavy/bench_gpu_scan.py [256K|1M|4M|10M|all]
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

OUT = FORK / "tests" / "heavy" / "bench_gpu_scan.json"
SEED = 42
REPS = 5
WARM = 3

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "nfgpu_scan_bench", str(FORK / "src" / "Drivers" / "GPU" / "_lib"
                            / "gpu.py"))
G = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(G)

NS = {"256K": 256 * 1024, "1M": 1024 * 1024, "4M": 4 * 1024 * 1024,
      "10M": 10 * 1024 * 1024}


def med(fn, reps=REPS, warm=WARM):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def gputime(fn):
    """Dispatch-only estimate: fn runs dispatches on persistent buffers."""
    return med(fn)


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(NS) if want == "all" else [want]
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "reps": REPS, "warm": WARM, "sizes": {},
           "prefix_replace": {}, "notes": (
        "warm RTX2060 Vulkan wgpu-py; full=H2D+dispatch+D2H; "
        "transfer=raw H2D+D2H same bytes; kernel=full-transfer; "
        "pipelines recompile per call (same for H and G, disclosed); "
        "numpy single-thread verbatim; overflow policy wraparound mod 2**32")}
    # device warm once
    G.scan_inclusive(np.ones(1024, dtype=np.int32))
    for nm in names:
        n = NS[nm]
        x = rng.integers(-10 ** 6, 10 ** 6, n, dtype=np.int32)
        # integrity gate: exact before timing
        assert (G.scan_inclusive(x) == G.scan_ref(x, "int32", True)).all()
        assert (G.scan_exclusive(x) == G.scan_ref(x, "int32", False)).all()
        assert (G.scan_exclusive_gpu(x)
                == G.scan_ref(x, "int32", False)).all()
        for op in ("sum", "min", "max", "count"):
            assert G.reduce_full(x, "int32", op) == G.reduce_ref(x, "int32",
                                                                 op)
        row = {"n": n}
        row["full_incl_H_ms"] = med(lambda: G.scan_inclusive(x))
        row["full_excl_H_ms"] = med(lambda: G.scan_exclusive(x))
        row["full_excl_G_ms"] = med(lambda: G.scan_exclusive_gpu(x))
        # transfer-only: raw H2D+D2H of N u32 in + N out (same bytes as scan)
        import wgpu
        dev = G._device()
        u = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC \
            | wgpu.BufferUsage.COPY_DST

        def xfer():
            b = dev.create_buffer_with_data(
                data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
            o = dev.create_buffer(size=n * 4, usage=u)
            _ = bytes(dev.queue.read_buffer(o))
        row["transfer_ms"] = med(xfer)
        row["kernel_incl_H_ms"] = max(
            0.0, row["full_incl_H_ms"] - row["transfer_ms"])
        row["kernel_excl_H_ms"] = max(
            0.0, row["full_excl_H_ms"] - row["transfer_ms"])
        row["kernel_excl_G_ms"] = max(
            0.0, row["full_excl_G_ms"] - row["transfer_ms"])
        for op in ("sum", "min", "max", "count"):
            row[f"reduce_{op}_ms"] = med(
                lambda op=op: G.reduce_full(x, "int32", op))
        t = time.perf_counter()
        _ = np.cumsum(x.astype(np.int64))
        row["numpy_cumsum_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        _ = (int(x.astype(np.int64).sum()), int(x.min()), int(x.max()))
        row["numpy_red_ms"] = (time.perf_counter() - t) * 1000
        row["gbytes_full"] = round(2 * n * 4 / 1e9, 4)
        res["sizes"][nm] = row
        print(f"{nm}: " + json.dumps(row), flush=True)
        gc.collect()
    # prefix-replacement: W block counts as Filter pays them at 10M rows
    n10 = NS["10M"]
    w = (n10 + 256 - 1) // 256
    cnt = rng.integers(0, 257, w, dtype=np.int32)
    assert (G.scan_exclusive(cnt) == G.scan_ref(cnt, "int32", False)).all()
    assert (G.scan_exclusive_gpu(cnt)
            == G.scan_ref(cnt, "int32", False)).all()
    import wgpu
    dev = G._device()
    u = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC \
        | wgpu.BufferUsage.COPY_DST

    def host_prefix_stage():
        b = dev.create_buffer_with_data(
            data=cnt.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST)
        d = bytes(dev.queue.read_buffer(b))
        c = np.frombuffer(d, dtype=np.int32).copy()
        offs = np.zeros(c.size, dtype=np.int64)
        if c.size > 1:
            offs[1:] = np.cumsum(c[:-1])
        o = offs.astype(np.int32)
        _h = dev.create_buffer_with_data(
            data=o.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    pr = {"W": w}
    pr["host_prefix_stage_ms"] = med(host_prefix_stage)
    # resident prefix dispatches only (persistent buffers, no transfers)
    b_c = dev.create_buffer_with_data(
        data=cnt.astype(np.uint32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE)
    mk = u
    b_i = dev.create_buffer(size=w * 4, usage=mk)
    b_p = dev.create_buffer(size=w * 4, usage=mk)
    w2 = (w + 256 - 1) // 256
    b_p2 = dev.create_buffer(size=w2 * 4, usage=mk)
    b_o2 = dev.create_buffer(size=w2 * 4, usage=mk)
    blk = G._SCAN_BLOCK_WGSL.replace("{WG}", "256").replace("{WGM1}", "255")
    e1 = G._SCAN_EXCL1_WGSL.replace("{WG}", "256")
    fx = G._SCAN_FIXUP_WGSL.replace("{WG}", "256")

    def gpu_prefix_stage():
        # level-1 incl of counts in b_i; level-2 reuses b_c as scratch
        # (counts consumed by D1), poff1 lands in b_c; dispatch-only.
        G._chain_dispatch(dev, blk.replace("{N}", str(w)),
                          [(b_c, True), (b_i, False), (b_p, False)], w)
        G._chain_dispatch(dev, blk.replace("{N}", str(w)),
                          [(b_p, True), (b_c, False), (b_p2, False)], w)
        G._chain_dispatch(dev, e1.replace("{N}", str(w2)),
                          [(b_p2, True), (b_o2, False)], 256)
        G._chain_dispatch(dev, fx.replace("{N}", str(w)).replace(
            "{EXCL}", "v = v - x[i];"),
            [(b_p, True), (b_c, False), (b_o2, True)], w)
    # NOTE: result identical to scan_exclusive prefix; timed dispatch-only.
    pr["gpu_prefix_stage_ms"] = med(gpu_prefix_stage)
    pr["full_scan_excl_H_ms"] = med(lambda: G.scan_exclusive(cnt))
    pr["full_scan_excl_G_ms"] = med(lambda: G.scan_exclusive_gpu(cnt))
    res["prefix_replace"] = pr
    print("prefix_replace: " + json.dumps(pr), flush=True)
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
            prev.get("sizes", {}).update(res["sizes"])
            res["sizes"] = prev["sizes"]
        except (ValueError, KeyError):
            pass
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
