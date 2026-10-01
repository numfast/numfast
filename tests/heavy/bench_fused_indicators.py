# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fused single-dispatch vs separate dispatches, N=1M/4M/10M, RTX 2060, warm.

Same 5 indicators (sma20/rsi14/boll_up20/boll_lo20/stoch_k14), same math
(one WGSL generator both modes): fused = upload ONCE -> 1 dispatch -> 5 D2H;
separate = upload ONCE (resident) -> 5 dispatches -> 5 D2H. H2D/D2H bytes
identical -> delta is pure dispatch overhead + L1/L2 input reuse.
Correctness gate: fused vs separate BIT-EXACT at each N (mismatch = STOP),
oracle tolerance on 20k prefix. No Planner/Kernel/SPEC change. No 1B.
No ClickBench/NFS. Seed 42.

Usage (Git Bash, sequential, timeout):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 590 /c/App/numfast/.venv/Scripts/python.exe \
    tests/heavy/bench_fused_indicators.py [--smoke]
"""
import gc
import json
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import importlib.util as _ilu

_fspec = _ilu.spec_from_file_location(
    "nffused", str(FORK / "src" / "Compute" / "Fused" / "_lib" / "fused.py"))
F = _ilu.module_from_spec(_fspec)
_fspec.loader.exec_module(F)

SMOKE = "--smoke" in sys.argv
NS = [20_000] if SMOKE else [1_000_000, 4_000_000, 10_000_000]
REPS = 2 if SMOKE else 5
OUT = FORK / "tests" / "heavy" / "bench_fused_indicators.json"

FIVE = [
    {"name": "sma20", "op": "sma", "params": {"w": 20}},
    {"name": "rsi14", "op": "rsi", "params": {"w": 14}},
    {"name": "bup20", "op": "boll_up", "params": {"w": 20, "k": 2.0}},
    {"name": "blo20", "op": "boll_lo", "params": {"w": 20, "k": 2.0}},
    {"name": "stk14", "op": "stoch_k", "params": {"w": 14}},
]


def _quotes(n, seed=42):
    # Bounded quotes-like series (~[85, 115]): drift-free, finite at any N
    # (a raw random walk explodes under exp() by N=10M).
    rng = np.random.default_rng(seed)
    i = np.arange(n, dtype=np.float64)
    x = (100.0 + 8.0 * np.sin(2.0 * np.pi * i / 256.0)
         + rng.normal(0.0, 1.0, size=n))
    return x.astype(np.float32)


def _bitexact(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return (a.shape == b.shape and a.dtype == b.dtype
            and (a.view(np.uint32) == b.view(np.uint32)).all())


def _oracle_ok(x, outs):
    # Full tolerance check (same contract as fast tests): NaN placement
    # exact, values within max(atol, rtol*|ref|) or 4 ULP. STOP otherwise.
    import struct as _st
    ref = F.fused_oracle(x[:20_000], FIVE)
    worst_rel = 0.0

    def _ord(f):
        (v,) = _st.unpack("<i", _st.pack("<f", float(f)))
        v &= 0xFFFFFFFF
        return v ^ (0xFFFFFFFF if v & 0x80000000 else 0x80000000)

    for g, r, s in zip(outs, ref, FIVE):
        gg = np.asarray(g[:20_000], dtype=np.float64)
        rr = np.asarray(r, dtype=np.float64)
        m = np.isnan(gg) & np.isnan(rr)
        if (np.isnan(gg) ^ np.isnan(rr)).any():
            return False, float("inf"), s["name"] + ":NaN"
        d = np.abs(gg - rr)
        d[m] = 0.0
        worst_rel = max(worst_rel,
                        float((d / np.maximum(np.abs(rr), 1e-30)).max()))
        tol = np.maximum(1e-5, 1e-5 * np.abs(rr))
        bad = np.where(~((d <= tol) | m))[0]
        for i in bad:
            if abs(_ord(gg[i]) - _ord(rr[i])) > 4:
                return False, worst_rel, f"{s['name']}@{i}"
    return True, worst_rel, ""


def _run_all(comp, b_x, n, reps):
    """Best-of-reps (submit_ms, d2h_ms) for one compiled plan on resident x."""
    best = None
    for _ in range(reps):
        res, info = F.execute_compiled(comp, x_buf=b_x)
        key = (info["submit_ms"], info["d2h_ms"])
        if best is None or sum(key) < sum(best[:2]):
            best = (info["submit_ms"], info["d2h_ms"], res)
    return best


def bench_n(n):
    print(f"\n-- N={n} --", flush=True)
    x = _quotes(n)
    t0 = time.perf_counter()
    b_x = F.r_upload(x)
    h2d_ms = (time.perf_counter() - t0) * 1000
    h2d_bytes = n * 4
    print(f"  H2D (shared, both modes): {h2d_ms:.2f} ms "
          f"({h2d_bytes / 1e6:.1f} MB)", flush=True)

    comp_f = F.compile_spec(FIVE)
    comps_s = [F.compile_spec([s]) for s in FIVE]

    # warm (pipeline compile cached; stats prove reuse)
    F.execute_compiled(comp_f, x_buf=b_x)
    for c in comps_s:
        F.execute_compiled(c, x_buf=b_x)
    # second warm to settle clocks
    F.execute_compiled(comp_f, x_buf=b_x)
    for c in comps_s:
        F.execute_compiled(c, x_buf=b_x)
    stats = F._ctx().stats()
    print(f"  pipelines cached: {stats}", flush=True)

    f_sub, f_d2h, f_res = _run_all(comp_f, b_x, n, REPS)
    s_sub = s_d2h = 0.0
    s_res = []
    for c in comps_s:
        a, b, r = _run_all(c, b_x, n, REPS)
        s_sub += a
        s_d2h += b
        s_res.append(r[0])

    # correctness gate: bit-exact fused vs separate (same generator)
    for f, s, spec in zip(f_res, s_res, FIVE):
        if not _bitexact(f, s):
            raise SystemExit(f"STOP: {spec['name']} fused != separate "
                             f"(bits differ, N={n})")
    ok, worst, where = _oracle_ok(x, f_res)
    if not ok:
        raise SystemExit(f"STOP: oracle mismatch at {where}, N={n}")
    print(f"  correctness: fused==separate bit-exact; "
          f"oracle 20k-prefix worst-rel-diff={worst:.3g}", flush=True)

    d2h_bytes = n * 4 * len(FIVE)
    nominal_gb = (h2d_bytes + d2h_bytes) / 1e9
    f_e2e = h2d_ms + f_sub + f_d2h
    s_e2e = h2d_ms + s_sub + s_d2h
    row = {"n": n, "h2d_ms": h2d_ms, "h2d_bytes": h2d_bytes,
           "fused": {"dispatches": 1, "submit_ms": f_sub, "d2h_ms": f_d2h,
                     "e2e_ms": f_e2e, "d2h_bytes": d2h_bytes,
                     "nominal_gbps": nominal_gb / max(f_sub + f_d2h, 1e-9)
                     * 1000},
           "separate": {"dispatches": len(FIVE), "submit_ms": s_sub,
                        "d2h_ms": s_d2h, "e2e_ms": s_e2e,
                        "d2h_bytes": d2h_bytes,
                        "nominal_gbps": nominal_gb / max(s_sub + s_d2h, 1e-9)
                        * 1000},
           "speedup_e2e": s_e2e / f_e2e,
           "speedup_kernel": (s_sub + s_d2h) / max(f_sub + f_d2h, 1e-9),
           "oracle_20k_worst_rel": worst}
    print(f"  fused   : 1 dispatch submit={f_sub:.2f}ms d2h={f_d2h:.2f}ms "
          f"e2e={f_e2e:.2f}ms gbps={row['fused']['nominal_gbps']:.1f}",
          flush=True)
    print(f"  separate: 5 dispatches submit={s_sub:.2f}ms d2h={s_d2h:.2f}ms "
          f"e2e={s_e2e:.2f}ms gbps={row['separate']['nominal_gbps']:.1f}",
          flush=True)
    print(f"  speedup: e2e x{row['speedup_e2e']:.2f}, "
          f"kernel x{row['speedup_kernel']:.2f}", flush=True)
    del b_x, x, f_res, s_res
    gc.collect()
    return row


def bench_scale(n=10_000_000):
    # Dispatch-overhead scaling law at fixed N: K=1/5/8 outputs.
    # Separate submits grow with K, fused stays 1 dispatch.
    print(f"\n-- K-scaling N={n} --", flush=True)
    x = _quotes(n)
    b_x = F.r_upload(x)
    specs = {
        1: FIVE[:1],
        5: FIVE,
        8: (FIVE + [{"name": "rsum20", "op": "rsum", "params": {"w": 20}},
                    {"name": "mom5", "op": "mom", "params": {"lag": 5}},
                    {"name": "dbl", "op": "axpb",
                     "params": {"a": 2.0, "b": 1.0}}]),
    }
    rows = []
    for k, spec in specs.items():
        comp_f = F.compile_spec(spec)
        comps_s = [F.compile_spec([s]) for s in spec]
        F.execute_compiled(comp_f, x_buf=b_x)
        for c in comps_s:
            F.execute_compiled(c, x_buf=b_x)
        f_sub, f_d2h, f_res = _run_all(comp_f, b_x, n, REPS)
        s_sub = s_d2h = 0.0
        for c, s, f in zip(comps_s, spec, f_res):
            a, b, r = _run_all(c, b_x, n, REPS)
            s_sub += a
            s_d2h += b
            assert _bitexact(f, r[0]), f"STOP: scale K={k} {s['name']}"
        rows.append({"k": k, "fused_submit_ms": f_sub,
                     "separate_submit_ms": s_sub,
                     "submit_saved_ms": s_sub - f_sub,
                     "fused_dispatches": 1, "separate_dispatches": k})
        print(f"  K={k}: fused submit={f_sub:.2f}ms (1 dsp) vs separate "
              f"submit={s_sub:.2f}ms ({k} dsp) -> saved {s_sub - f_sub:.2f}ms",
              flush=True)
    del b_x, x
    gc.collect()
    return rows


def main():
    # Pay device + first-compile cost ONCE, outside per-N H2D timing.
    x0 = _quotes(1024)
    F.execute_compiled(F.compile_spec(FIVE[:1]), x=x0)
    del x0
    gc.collect()
    t0 = time.perf_counter()
    rows = [bench_n(n) for n in NS]
    out = {"device": "RTX 2060", "seed": 42, "reps": REPS,
           "outputs": [s["name"] for s in FIVE],
           "elapsed_s": 0.0, "rows": rows}
    if not SMOKE:
        out["k_scaling"] = bench_scale()
        out["elapsed_s"] = time.perf_counter() - t0
    else:
        out["elapsed_s"] = time.perf_counter() - t0
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT} elapsed={out['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
