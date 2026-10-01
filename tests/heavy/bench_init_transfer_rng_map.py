# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Init/transfer/rng/map measurement (bench-only, NEW file, no Planner change).

Seed 42, stage breakdown, best-of-3 warm, cold via fresh subprocesses.
States never mixed: every point labeled cold_process / warm_process /
already_resident / H2D / D2H. Writes dataset candidate JSON only;
NEVER touches calibration.toml. GroupBy-cardinality / sort-nonpow2 excluded.

Usage (Git Bash, sequential):
  timeout 550 /c/App/numfast/.venv/Scripts/python tests/heavy/bench_init_transfer_rng_map.py
Worker (internal, spawned by main):
  .../bench_init_transfer_rng_map.py --worker <device_init|cold_rng|cold_map> ...
"""
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from builder import MAIN  # noqa: E402

SEED = 42
REPS = 3
WARM = 2
NS = (10_000, 100_000, 1_000_000, 10_000_000)
BYTES = (1 << 20, 10 << 20, 100 << 20, 1 << 30)  # 1MB/10MB/100MB/1GB
CANDIDATE = FORK / "calibration_dataset_candidate_init_transfer_rng_map.json"
PY = sys.executable


def fit_line(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n = len(xs)
    if n == 1:
        return 0.0, float(ys[0])
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return a, my - a * mx


def r2(xs, ys):
    n = len(xs)
    if n < 2:
        return 1.0
    my = sum(ys) / n
    tot = sum((y - my) ** 2 for y in ys)
    if tot == 0.0:
        return 1.0
    a, b = fit_line(xs, ys)
    res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - res / tot


def best3(fn):
    for _ in range(WARM):
        fn()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    return min(ts), sorted(ts)[len(ts) // 2], ts


def build():
    kernel = MAIN["build"](str(FORK))
    return kernel.alias


def run_graph(a, jobs, backend):
    g = a["compile"](jobs)
    return a["cpu_execute"](g["nodes"]) if backend == "cpu" else a["gpu_execute"](g["nodes"])


def jobs_rng(a, n):
    return [a["ir_rng_fill_i32"]("r", int(n), 42, lo=0, hi=100)], "r"


def jobs_map(a, dtype, n):
    rng = np.random.default_rng(SEED)
    if dtype == "int32":
        v = rng.integers(0, 100, int(n), dtype=np.int32)
    else:
        v = (rng.random(int(n)) * 100).astype(np.float32)
    return ([a["ir_series"]("x", v, dtype),
             a["ir_map"]("m", "x", fn="mul", value=2)], "m", v)


# ---- workers (fresh cold processes; stdout single JSON line) ----
def worker():
    kind = sys.argv[2]
    a = build()
    if kind == "device_init":
        jobs0, _ = jobs_rng(a, 10_000)
        t0 = time.perf_counter()
        run_graph(a, jobs0, "gpu")
        first = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        run_graph(a, jobs0, "gpu")
        second = (time.perf_counter() - t0) * 1000.0
        rng = np.random.default_rng(SEED)
        v = rng.integers(0, 100, 10_000, dtype=np.int32)
        cmp_jobs = ([a["ir_series"]("x", v),
                     a["ir_compare"]("m", "x", 50, ">")], "m")[0]
        t0 = time.perf_counter()
        run_graph(a, cmp_jobs, "gpu")
        other = (time.perf_counter() - t0) * 1000.0
        print(json.dumps({"first_ms": first, "second_same_ms": second,
                          "first_other_op_ms": other}))
    elif kind == "cold_rng":
        backend, n = sys.argv[3], int(sys.argv[4])
        jobs, _ = jobs_rng(a, n)
        t0 = time.perf_counter()
        bufs = run_graph(a, jobs, backend)
        ms = (time.perf_counter() - t0) * 1000.0
        print(json.dumps({"cold_ms": ms, "size": int(np.asarray(bufs["r"]).size)}))
    elif kind == "cold_map":
        dtype, n = sys.argv[3], int(sys.argv[4])
        jobs, _, _ = jobs_map(a, dtype, n)
        t0 = time.perf_counter()
        bufs = run_graph(a, jobs, "cpu")
        ms = (time.perf_counter() - t0) * 1000.0
        print(json.dumps({"cold_ms": ms, "size": int(np.asarray(bufs["m"]).size)}))
    return 0


def cold_worker(args, reps=REPS):
    vals = []
    for _ in range(reps):
        p = subprocess.run([PY, str(Path(__file__))] + args, capture_output=True,
                           text=True, timeout=300)
        if p.returncode != 0:
            raise RuntimeError(f"worker {args} failed: {p.stderr[-2000:]}")
        vals.append(json.loads(p.stdout.strip().splitlines()[-1]))
    return vals


def main():
    t_all = time.perf_counter()
    a = build()
    raw = {"device_init": {}, "transfer": {}, "resident": {}, "rng_fill_i32": {},
           "map": {}}
    stages = []

    # 1. device_init: 5 cold procs, median; warm = in-process 2nd call
    dev_runs = cold_worker(["--worker", "device_init"], reps=5)
    f = sorted(r["first_ms"] for r in dev_runs)
    s = sorted(r["second_same_ms"] for r in dev_runs)
    o = sorted(r["first_other_op_ms"] for r in dev_runs)
    raw["device_init"] = {
        "state": "cold_process vs warm_process, RTX 2060 Vulkan, seed 42",
        "cold_first_runs_ms": sorted(r["first_ms"] for r in dev_runs),
        "cold_first_median_ms": f[2],
        "warm_second_median_ms": s[2],
        "compile_only_other_op_median_ms": o[2],
        "device_init_one_time_median_ms": max(0.0, f[2] - o[2]),
        "note": "cold_first=device request+compile(rng)+exec@10K; "
                "other=compile(compare)+exec@10K device-warm; "
                "init~=cold_first-other (same-N approx)"}
    stages.append(f"device_init cold_first_med={f[2]:.2f}ms warm_2nd_med={s[2]:.2f}ms "
                  f"compile_only_med={o[2]:.2f}ms init~={max(0.0, f[2]-o[2]):.2f}ms")

    # GPU lazy import for transfers/resident (importlib, bench-only pattern)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "nfgpu_meas", str(FORK / "src" / "Drivers" / "GPU" / "_lib" / "gpu.py"))
    G = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(G)

    # 2. H2D/D2H curves (pure transfer, explicit state labels)
    for direction in ("h2d", "d2h"):
        raw["transfer"][direction] = {}
        for nb in BYTES:
            n = nb // 4
            rng = np.random.default_rng(SEED)
            arr = rng.integers(0, 100, n, dtype=np.int32)
            if direction == "h2d":
                best, med, runs = best3(lambda: G.r_upload(arr))
                stages.append(f"H2D {nb//2**20}MB best={best:.2f}ms "
                              f"{nb/best/1e6:.2f}GB/s")
            else:
                buf = G.r_upload(arr)
                best, med, runs = best3(lambda: G.r_download(buf, np.int32))
                got = G.r_download(buf, np.int32)
                if not bool((got == arr).all()):
                    raise RuntimeError(f"D2H integrity STOP @{nb}B")
                stages.append(f"D2H {nb//2**20}MB best={best:.2f}ms "
                              f"{nb/best/1e6:.2f}GB/s")
                del buf
            raw["transfer"][direction][str(nb)] = {
                "state": "CPU->GPU" if direction == "h2d" else "GPU->CPU",
                "bytes": nb, "best_ms": best, "median_ms": med, "runs_ms": runs}
            gc.collect()

    # already-resident: upload once, dispatch-only (zero transfer)
    n_res = 10_000_000
    rng = np.random.default_rng(SEED)
    ab = G.r_upload(rng.integers(0, 2, n_res, dtype=np.int32).astype(np.int32))
    bb = G.r_upload(rng.integers(0, 2, n_res, dtype=np.int32).astype(np.int32))
    G.r_mask(ab, bb, "and", n_res)
    best, med, runs = best3(lambda: G.r_mask(ab, bb, "and", n_res))
    raw["resident"] = {"mask_and@10M": {
        "state": "already-resident (no H2D/D2H inside)", "n": n_res,
        "best_ms": best, "median_ms": med, "runs_ms": runs}}
    stages.append(f"resident mask_and@10M best={best:.2f}ms (dispatch only)")
    del ab, bb
    gc.collect()

    # 3. rng_fill_i32 CPU+GPU x N, warm in-proc + cold via workers
    for backend in ("cpu", "gpu"):
        raw["rng_fill_i32"][backend] = {}
        for n in NS:
            jobs, out = jobs_rng(a, n)
            if backend == "gpu":  # integrity once per N before timing
                cb = run_graph(a, jobs, "cpu")
                gb = run_graph(a, jobs, "gpu")
                if not bool((np.asarray(cb[out]) == np.asarray(gb[out])).all()):
                    raise RuntimeError(f"rng integrity STOP n={n}")
            best, med, runs = best3(lambda: run_graph(a, jobs, backend))
            colds = cold_worker(["--worker", "cold_rng", backend, str(n)])
            cm = sorted(c["cold_ms"] for c in colds)[1]
            raw["rng_fill_i32"][backend][str(n)] = {
                "warm_state": "warm_process", "warm_best_ms": best,
                "warm_median_ms": med, "warm_runs_ms": runs,
                "cold_state": "cold_process (1st call, fresh proc)",
                "cold_runs_ms": [c["cold_ms"] for c in colds],
                "cold_median_ms": cm}
            stages.append(f"rng_fill_i32 {backend} n={n} warm_best={best:.2f}ms "
                          f"cold_med={cm:.2f}ms")

    # 4. map CPU x dtype x N (warm+cold); GPU attempt -> explicit error evidence
    for dtype in ("int32", "float32"):
        raw["map"][dtype] = {"cpu": {}}
        for n in NS:
            jobs, out, v = jobs_map(a, dtype, n)
            cb = run_graph(a, jobs, "cpu")
            if not np.allclose(np.asarray(cb[out]),
                               np.asarray(v) * 2, rtol=1e-5, atol=1e-5):
                raise RuntimeError(f"map integrity STOP {dtype} n={n}")
            best, med, runs = best3(lambda: run_graph(a, jobs, "cpu"))
            colds = cold_worker(["--worker", "cold_map", dtype, str(n)])
            cm = sorted(c["cold_ms"] for c in colds)[1]
            raw["map"][dtype]["cpu"][str(n)] = {
                "warm_state": "warm_process", "warm_best_ms": best,
                "warm_median_ms": med, "warm_runs_ms": runs,
                "cold_state": "cold_process (1st call, fresh proc)",
                "cold_runs_ms": [c["cold_ms"] for c in colds],
                "cold_median_ms": cm}
            stages.append(f"map cpu {dtype} n={n} warm_best={best:.2f}ms "
                          f"cold_med={cm:.2f}ms")
    try:
        jobs, _, _ = jobs_map(a, "int32", 10_000)
        run_graph(a, jobs, "gpu")
        raw["map"]["gpu"] = {"state": "warm_process", "status": "unexpected-success"}
        stages.append("map gpu UNEXPECTED success @10K")
    except Exception as e:  # noqa: BLE001 -- error text is the evidence
        raw["map"]["gpu"] = {"state": "warm_process",
                             "status": "explicit-cpu-only-error",
                             "error": str(e)[-500:]}
        stages.append(f"map gpu explicit-error: {str(e)[-160:]}")

    # fits (warm=best-of-3; cold=median-of-3; transfer fit on all 4 + 1GB holdout check)
    cost, metrics, validation = {}, {}, {}
    cost["device_init_one_time_ms"] = raw["device_init"]["device_init_one_time_median_ms"]
    for direction in ("h2d", "d2h"):
        xs = list(BYTES)
        ys = [raw["transfer"][direction][str(b)]["best_ms"] for b in xs]
        fa, fb = fit_line(xs, ys)
        cost[f"transfer_{direction}_a_ms_per_byte"] = fa
        cost[f"transfer_{direction}_b_ms"] = fb
        metrics[f"transfer_{direction}_r2"] = r2(xs, ys)
        ha, hb = fit_line(xs[:3], ys[:3])  # hold out 1GB
        pred = ha * xs[3] + hb
        validation[f"transfer_{direction}@1GB_holdout"] = {
            "predicted_ms": pred, "measured_ms": ys[3],
            "abs_err_ms": pred - ys[3],
            "rel_err": (pred - ys[3]) / ys[3] if ys[3] else 0.0}
    for backend in ("cpu", "gpu"):
        for st, fld in (("warm", "warm_best_ms"), ("cold", "cold_median_ms")):
            ys = [raw["rng_fill_i32"][backend][str(n)][fld] for n in NS]
            fa, fb = fit_line(list(NS), ys)
            cost[f"rng_fill_i32_{backend}_{st}_a"] = fa
            cost[f"rng_fill_i32_{backend}_{st}_b"] = fb
            metrics[f"rng_fill_i32_{backend}_{st}_r2"] = r2(list(NS), ys)
        gaps = [raw["rng_fill_i32"][backend][str(n)]["cold_median_ms"]
                - raw["rng_fill_i32"][backend][str(n)]["warm_best_ms"] for n in NS]
        gaps.sort()
        cost[f"rng_fill_i32_{backend}_compile_ms"] = max(0.0, gaps[1])
    for dtype in ("int32", "float32"):
        for st, fld in (("warm", "warm_best_ms"), ("cold", "cold_median_ms")):
            ys = [raw["map"][dtype]["cpu"][str(n)][fld] for n in NS]
            fa, fb = fit_line(list(NS), ys)
            cost[f"map_{dtype}_cpu_{st}_a"] = fa
            cost[f"map_{dtype}_cpu_{st}_b"] = fb
            metrics[f"map_{dtype}_cpu_{st}_r2"] = r2(list(NS), ys)

    cap = {}
    try:
        cap = a.get("gpu_capability", lambda: {})()
    except Exception:  # noqa: BLE001
        pass
    import platform as _pf
    dataset = {"profile": {
        "model_version": "calibrated_v1", "profile_version": 2,
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": "measured:seed42", "quick": False,
        "hardware": {"cpu": _pf.processor() or "unknown",
                     "platform": _pf.platform(),
                     "python": _pf.python_version(), "backend": "webgpu",
                     "gpu_device": "NVIDIA GeForce RTX 2060",
                     "gpu_backend": "Vulkan",
                     "capability_note": str(cap.get("note", ""))[:300]},
        "cost": cost, "metrics": metrics,
        "measurements": {"Ns": list(NS), "bytes": list(BYTES), "reps": REPS,
                         "warmup": WARM, "seed": SEED,
                         "note": "candidate only: init+transfer+rng+map; "
                                 "groupby-card/sort-nonpow2 excluded; "
                                 "NOT for calibration.toml"}},
        "raw": raw, "validation": validation}
    CANDIDATE.write_text(json.dumps(dataset, indent=1), encoding="utf-8")
    el = time.perf_counter() - t_all
    for line in stages:
        print("  stage " + line)
    print(f"dataset={CANDIDATE} elapsed={el:.1f}s")
    print("NOT covered: groupby-cardinality, sort-nonpow2 (next step); "
          "calibration.toml untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(worker() if "--worker" in sys.argv else main())
