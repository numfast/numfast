# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pass 4A: real select_backend(auto) for rng_fill_i32, NOT manual choice.

Seed 42, stage breakdown, sequential. For each N in (1M, 10M, 50M):
  S1 CPU-resident (residence_from=host) x warm/cold
  S2 GPU-resident (residence_from=resident) x warm/cold
  S3 transfer accounting (H2D/D2H slots + execution_info crossings)
  S4 cold vs warm (device_state hint + cold-subprocess actuals)
For each point: predicted (cost_estimate), actual CPU/GPU (warm best-of-3
explicit + cold via fresh subprocess), chosen (auto), regret, H2D/D2H,
cold/warm, residence, execution_info fully.
Writes bench_pass4a_auto_rng.json next to this file. Touches nothing else.

Usage (Git Bash, sequential):
  timeout 550 /c/App/numfast/.venv/Scripts/python tests/heavy/bench_pass4a_auto_rng.py
Worker (internal, spawned by main):
  .../bench_pass4a_auto_rng.py --worker cold <cpu|gpu> <n>
"""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
SEED = 42
REPS = 3
WARMUP = 1
NS = (1_000_000, 10_000_000, 16_000_000)
# 16M = largest single-dispatch RNG under backend max_dispatch.x=65535
# (ceil(16M/256)=62500 workgroups); 50M needs chunked path which
# chunk_plan does not split on dispatch (out of scope: Planner frozen).
OUT = Path(__file__).resolve().parent / "bench_pass4a_auto_rng.json"
PY = sys.executable


def _boot():
    sys.path.insert(0, str(FORK / "src"))
    import numfast as nf  # noqa: E402
    k = nf.get_kernel()
    return nf, k.alias


def _jobs(a, n):
    return [a["ir_rng_fill_i32"]("r", int(n), SEED, 0, 0, 0, 100)]


def _best3(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(ts)
    return s[0], s[len(s) // 2], ts


def _worker():
    mode, backend, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
    assert mode == "cold"
    _, a = _boot()
    jobs = _jobs(a, n)
    g = a["compile"](jobs)
    t0 = time.perf_counter()
    # explicit backend via evaluate (Runtime contract; chunk_plan owns splits)
    res = a["evaluate"](g, backend, n)
    ms = (time.perf_counter() - t0) * 1000.0
    import numpy as _np
    print(json.dumps({"cold_ms": ms, "size": int(_np.asarray(res["result"]).size),
                      "head": [int(x) for x in _np.asarray(res["result"][:4])]}))
    return 0


def _cold_once(backend, n):
    p = subprocess.run([PY, str(Path(__file__)), "--worker", "cold",
                        backend, str(n)], capture_output=True, text=True,
                       timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"cold worker {backend}@{n} failed: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    t_all = time.perf_counter()
    import numpy as np
    _, a = _boot()
    cal = a["calibrate_info"]()
    stages = [f"calibrate hooks={[(h['hook'], h['status']) for h in cal['hooks']]}"]
    # GPU warmup in-process (pays init once, discarded, reported)
    t0 = time.perf_counter()
    a["evaluate"](a["compile"](_jobs(a, 1024)), "gpu", 1024)
    stages.append(f"device warmup 1x rng@1K gpu {(time.perf_counter()-t0)*1000:.1f}ms (discarded)")

    points = []
    for n in NS:
        # warm actuals: explicit both backends via evaluate (Runtime contract),
        # best-of-3 (integrity: CPU==GPU)
        rc = a["evaluate"](a["compile"](_jobs(a, n)), "cpu", n)
        rg = a["evaluate"](a["compile"](_jobs(a, n)), "gpu", n)
        if not bool((np.asarray(rc["result"]) == np.asarray(rg["result"])).all()):
            raise RuntimeError(f"integrity STOP rng n={n}: CPU != GPU")
        del rc, rg
        gc.collect()
        cpu_best, cpu_med, cpu_runs = _best3(
            lambda: a["evaluate"](a["compile"](_jobs(a, n)), "cpu", n))
        gpu_best, gpu_med, gpu_runs = _best3(
            lambda: a["evaluate"](a["compile"](_jobs(a, n)), "gpu", n))
        stages.append(f"actual-warm n={n} cpu_best={cpu_best:.2f}ms "
                      f"gpu_best={gpu_best:.2f}ms speedup={cpu_best/gpu_best:.2f}x")
        # cold actuals: fresh subprocess first-call (init penalty inside)
        cold_cpu = _cold_once("cpu", n)["cold_ms"]
        cold_gpu = _cold_once("gpu", n)["cold_ms"]
        stages.append(f"actual-cold n={n} cpu={cold_cpu:.2f}ms gpu={cold_gpu:.2f}ms")

        for residence in ("host", "resident"):
            for state in ("warm", "cold"):
                hints = {"residence_from": residence, "device_state": state}
                gg = a["compile"](_jobs(a, n))
                sel = a["select_backend"](gg, n, None, dict(hints))
                ev = a["evaluate"](gg, "auto", n, dict(hints))
                info = ev["execution_info"]
                ce = sel.get("cost_estimate") or {}
                # regret on warm actuals (planner predicts warm steady-state;
                # cold regret reported separately below)
                best_min = min(cpu_best, gpu_best)
                chosen = info["actual"]
                actual_chosen = cpu_best if chosen == "cpu" else gpu_best
                regret = actual_chosen - best_min
                # cold regret: what if this state were first-call?
                cold_chosen = cold_cpu if chosen == "cpu" else cold_gpu
                cold_regret = cold_chosen - min(cold_cpu, cold_gpu)
                est = sel.get("estimate")
                points.append({
                    "n": n, "scenario": (
                        "S1-cpu-resident" if residence == "host"
                        else "S2-gpu-resident"),
                    "hints": hints,
                    "predicted_cpu_ms": ce.get("cpu"),
                    "predicted_gpu_ms": ce.get("gpu"),
                    "chosen_select": sel.get("backend"),
                    "chosen_actual": chosen,
                    "reason": sel.get("reason"),
                    "gpu_eligible": sel.get("gpu_eligible"),
                    "gpu_blockers": sel.get("gpu_blockers"),
                    "coverage": sel.get("coverage"),
                    "profile": sel.get("profile"),
                    "estimate": est,
                    "actual_cpu_warm_best_ms": cpu_best,
                    "actual_cpu_warm_med_ms": cpu_med,
                    "actual_cpu_warm_runs_ms": cpu_runs,
                    "actual_gpu_warm_best_ms": gpu_best,
                    "actual_gpu_warm_med_ms": gpu_med,
                    "actual_gpu_warm_runs_ms": gpu_runs,
                    "actual_cpu_cold_ms": cold_cpu,
                    "actual_gpu_cold_ms": cold_gpu,
                    "regret_warm_ms": regret,
                    "regret_cold_ms": cold_regret,
                    "speedup_warm_cpu_over_gpu": cpu_best / gpu_best
                    if gpu_best else None,
                    "execution_info": {
                        "requested": info.get("requested"),
                        "actual": info.get("actual"),
                        "reason": info.get("reason"),
                        "dispatches": info.get("dispatches"),
                        "h2d": info.get("h2d"),
                        "d2h": info.get("d2h"),
                        "num_chunks": info.get("num_chunks"),
                        "placement": info.get("placement"),
                        "residence": info.get("residence"),
                        "cost_estimate": info.get("cost_estimate"),
                        "profile": info.get("profile"),
                        "gpu_eligible": info.get("gpu_eligible"),
                        "gpu_blockers": info.get("gpu_blockers"),
                        "coverage": info.get("coverage"),
                        "estimate": info.get("estimate"),
                    },
                })
                stages.append(
                    f"auto n={n} {residence}/{state}: "
                    f"pred=({ce.get('cpu')},{ce.get('gpu')}) "
                    f"chosen={chosen} regret_warm={regret:.2f}ms "
                    f"regret_cold={cold_regret:.2f}ms "
                    f"cross={info.get('residence', {}).get('crossings')} "
                    f"h2d/d2h={info.get('h2d')}/{info.get('d2h')}")
        gc.collect()

    el = time.perf_counter() - t_all
    OUT.write_text(json.dumps({"seed": SEED, "Ns": list(NS), "calibrate": cal,
                               "points": points, "stages": stages,
                               "elapsed_s": el}, indent=1), encoding="utf-8")
    for s in stages:
        print("  stage " + s)
    print(f"wrote={OUT} elapsed={el:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker() if "--worker" in sys.argv else main())
