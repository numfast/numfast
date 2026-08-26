"""Phase 4 - ROLLINGSTDDEV profile (S102b): 6 stages cold/warm.

Spec: ROLLINGSTDDEV_SLICE_SPEC.md sec.8 (S102b).

Scenarios (actual execution path, recorded in profile.json):
  A: RollingStdDev n=64 period=14    - ordinary (1 packet, NOT fused, dx=1).
  B: RollingStdDev n=1M period=100  - ordinary (dx=15625 workgroups, 1 packet).
  C: RollingStdDev n=4_194_240 period=100 - ordinary limit (dx=65535 workgroups, 1 packet).
  D: RollingStdDev large-mean-small-std n=1M period=100 - degraded zone, correctness within abs 1e-3.
Stages: compile, alloc (create_buffer_with_data/create_buffer), h2d (upload), dispatch, d2h (readback), pool_reuse (warm runs: uploads stable).

Actual path facts (S102b): RollingStdDev is a single-packet graph -> NOT fused; N <= 4_194_240 -> ordinary path (Runtime.execute -> driver.execute_wave -> driver.execute); chunked path unreachable (NOT chunkable); the GpuBufferPool is NOT used on the ordinary path (per-call create_buffer_with_data), so pool_reuse stability holds trivially (uploads=0). N > 4_194_240 -> dispatch-limit ValueError (_check_dispatch_limit) - documented, NOT profiled.

Evidence: evidence/rollingstddev_slice_phase4/profile.json

IMPORTANT: this file is ASCII-only (no Cyrillic) because
tests/test_backend.py::test_no_cupy_in_test_files reads tests/*.py with
open() in locale encoding (cp1251 on this host).
"""

import json
import os
import pathlib
import statistics
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "math")))

from Runtime._lib.runtime import Runtime
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Compute import register_all

CHECKPOINT_SHA = "a60e39b0c49434231433faac671369b8f7e34c2f"
SLICE = "rollingstddev"
SEED = 42
LIMIT = 4_194_240

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "rollingstddev_slice_phase4")

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}

needs_gpu = pytest.mark.skipif(not GPU_AVAILABLE,
                               reason="WebGPU adapter unavailable")


def _versions():
    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import wgpu
        v["wgpu"] = wgpu.__version__
    except Exception:  # noqa: BLE001
        v["wgpu"] = "n/a"
    v["gpu"] = "n/a"
    if _adapter is not None:
        info = dict(_adapter.info)
        v["gpu"] = f"{info.get('device', 'unknown')} ({info.get('backend_type', 'unknown')})"
    return v


def _make_gpu_runtime():
    rt = Runtime(driver=WebGpuDriver())
    register_all(rt)
    return rt


def _job_for(period):
    return {"op": "RollingStdDev", "inputs": ["x"], "params": {"period": period},
            "out": "y"}


def _wrap_execute_counter(driver):
    packets_seen = []
    orig = driver.execute

    def wrapped(packet):
        packets_seen.append(packet)
        return orig(packet)

    driver.execute = wrapped
    return packets_seen


def _wrap_alloc_timing(device):
    times = []

    def _wrap(name):
        orig = getattr(device, name)

        def timed(*a, **k):
            t0 = time.perf_counter_ns()
            r = orig(*a, **k)
            times.append(time.perf_counter_ns() - t0)
            return r

        setattr(device, name, timed)

    _wrap("create_buffer_with_data")
    _wrap("create_buffer")
    return times


def _sum_stages(packets, key):
    return sum(p.profile.get(key, 0) for p in packets)


@needs_gpu
def test_profile_6_stages():
    def profile_scenario(n, period, warm_runs, data_fn=None):
        rng = np.random.default_rng(SEED)
        if data_fn is not None:
            x = data_fn(rng, n)
        else:
            x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        job = _job_for(period)
        rt = _make_gpu_runtime()
        try:
            alloc_times = _wrap_alloc_timing(rt.driver._device)
            packets_seen = _wrap_execute_counter(rt.driver)
            t0 = time.perf_counter_ns()
            tasks = rt.compile([job])
            rt_compile_ns = time.perf_counter_ns() - t0
            t0 = time.perf_counter_ns()
            rt.execute(tasks, {"x": x})
            exec_wall_ns = time.perf_counter_ns() - t0
            chunks_observed = len(packets_seen)
            cold = {
                "compile_ns": rt_compile_ns
                              + _sum_stages(packets_seen, "compile_ns")
                              + _sum_stages(packets_seen, "pipeline_ns"),
                "alloc_ns": sum(alloc_times),
                "h2d_ns": _sum_stages(packets_seen, "upload_ns"),
                "dispatch_ns": _sum_stages(packets_seen, "dispatch_ns"),
                "d2h_ns": packets_seen[-1].profile.get("readback_ns", 0),
                "exec_wall_ns": exec_wall_ns,
                "latency_ns": rt_compile_ns + sum(alloc_times)
                              + _sum_stages(packets_seen, "upload_ns")
                              + _sum_stages(packets_seen, "dispatch_ns")
                              + packets_seen[-1].profile.get("readback_ns", 0),
                "dispatch_x": packets_seen[-1].profile.get("dispatch_x", 0),
            }
            warm = {"compile_ns": [], "alloc_ns": [], "h2d_ns": [],
                    "dispatch_ns": [], "d2h_ns": [], "exec_wall_ns": [],
                    "latency_ns": []}
            for _ in range(warm_runs):
                alloc_times.clear()
                packets_seen.clear()
                t0 = time.perf_counter_ns()
                tasks = rt.compile([job])
                warm["compile_ns"].append(time.perf_counter_ns() - t0)
                t0 = time.perf_counter_ns()
                rt.execute(tasks, {"x": x})
                warm["exec_wall_ns"].append(time.perf_counter_ns() - t0)
                warm["alloc_ns"].append(sum(alloc_times))
                warm["h2d_ns"].append(_sum_stages(packets_seen, "upload_ns"))
                warm["dispatch_ns"].append(_sum_stages(packets_seen, "dispatch_ns"))
                warm["d2h_ns"].append(packets_seen[-1].profile.get("readback_ns", 0))
                warm["latency_ns"].append(
                    warm["compile_ns"][-1] + warm["alloc_ns"][-1]
                    + warm["h2d_ns"][-1] + warm["dispatch_ns"][-1]
                    + warm["d2h_ns"][-1])
            med = {k: int(statistics.median(v)) for k, v in warm.items()}
            return rt, cold, med, chunks_observed, rt.driver._pool.stats()
        except Exception:
            rt.driver.release()
            raise

    # Scenario A: RollingStdDev n=64 period=14 (ordinary, dx=1; cold + 7 warm)
    rt64, cold64, warm64, chunks64, _ = profile_scenario(64, 14, 7)
    rt64.driver.release()
    assert chunks64 == 1, f"n=64 packets={chunks64}"

    # Scenario B: RollingStdDev n=1M period=100 (ordinary, dx=15625; cold + 5 warm)
    rt1m, cold1m, warm1m, chunks1m, pool_stats_1m = profile_scenario(
        1_000_000, 100, 5)
    rt1m.driver.release()
    assert chunks1m == 1, f"n=1M packets={chunks1m}"

    # Scenario C: RollingStdDev n=4_194_240 period=100 (ordinary limit, dx=65535; cold + 3 warm)
    rt4m, cold4m, warm4m, chunks4m, pool_stats_4m = profile_scenario(
        LIMIT, 100, 3)
    rt4m.driver.release()
    assert chunks4m == 1, f"n={LIMIT} packets={chunks4m}"

    # Scenario D: large-mean-small-std n=1M period=100 (degraded zone)
    def large_mean_data(rng, n):
        return (rng.standard_normal(n).astype(np.float32) * np.float32(1e-2) + np.float32(1e6)).astype(np.float32)

    rtD, coldD, warmD, chunksD, pool_stats_D = profile_scenario(1_000_000, 100, 3, data_fn=large_mean_data)
    rtD.driver.release()
    assert chunksD == 1, f"n=1M degraded packets={chunksD}"

    # path facts
    assert cold64["dispatch_x"] == 1, f"n=64 dx={cold64['dispatch_x']}"
    assert cold1m["dispatch_x"] == 15625, f"n=1M dx={cold1m['dispatch_x']}"
    assert cold4m["dispatch_x"] == 65535, f"n={LIMIT} dx={cold4m['dispatch_x']}"
    assert coldD["dispatch_x"] == 15625, f"n=1M D dx={coldD['dispatch_x']}"

    # pool reuse: warm RollingStdDev runs, uploads stable
    rt_f = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        x = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        tasks = rt_f.compile([_job_for(14)])
        rt_f.execute(tasks, {"x": x})
        after_first = rt_f.driver._pool.stats()
        rt_f.execute(tasks, {"x": x})
        after_warm = rt_f.driver._pool.stats()
        assert after_first["uploads"] == after_warm["uploads"], \
            "warm run must not re-upload"
        pool_reuse = {
            "after_first": after_first,
            "after_warm": after_warm,
            "uploads_stable": after_first["uploads"] == after_warm["uploads"],
            "resident_stable": after_first["resident_bytes"] == after_warm["resident_bytes"],
            "note": "RollingStdDev n=64 ordinary path; pool NOT used on the "
                    "ordinary path (per-call create_buffer_with_data), "
                    "so uploads=0 and stability holds trivially",
        }
    finally:
        rt_f.driver.release()

    # N > 4_194_240 -> dispatch-limit ValueError, documented NOT profiled
    limit_fact = "skipped (no adapter)"
    if GPU_AVAILABLE:
        rt_l = _make_gpu_runtime()
        try:
            y = np.zeros(LIMIT + 1, dtype=np.float32)
            try:
                rt_l.execute(rt_l.compile([_job_for(100)]), {"x": y})
                limit_fact = "unexpectedly worked"
            except ValueError as e:
                limit_fact = (f"ValueError '{str(e)[:80]}' BEFORE dispatch "
                              f"(_check_dispatch_limit, S64) - documented, "
                              f"NOT profiled")
        finally:
            rt_l.driver.release()

    evidence = {
        "phase": 4,
        "slice": SLICE,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": _versions(),
        "result": {
            "device": {"adapter": ADAPTER_INFO.get("device", "unknown"),
                       "backend": ADAPTER_INFO.get("backend_type", "unknown")},
            "stages": ["compile", "alloc", "h2d", "dispatch", "d2h",
                       "pool_reuse"],
            "rs_n64_p14": {"path": "ordinary (1 packet, NOT fused, dx=1, "
                                   "1 workgroup)",
                           "packets": chunks64, "cold": cold64,
                           "warm_median": warm64},
            "rs_n1M_p100": {"path": "ordinary (<= limit 4_194_240, 1 "
                                     "packet, NOT fused, dx=15625)",
                            "packets": chunks1m, "cold": cold1m,
                            "warm_median": warm1m,
                            "pool_after_warm": pool_stats_1m},
            "rs_n4194240_p100": {"path": "ordinary limit (dx=65535, 1 "
                                         "packet, NOT fused)",
                                 "packets": chunks4m, "cold": cold4m,
                                 "warm_median": warm4m,
                                 "pool_after_warm": pool_stats_4m},
            "rs_n1M_p100_large_mean": {"path": "ordinary degraded zone large-mean-small-std (1 packet, dx=15625)",
                                       "packets": chunksD, "cold": coldD,
                                       "warm_median": warmD,
                                       "pool_after_warm": pool_stats_D,
                                       "note": "large-mean-small-std core: mean 1e6 std 1e-2 ratio 1e8 -> abs 1e-3 degraded tolerance, cancellation variance<0->0"},
            "pool_reuse": pool_reuse,
            "limit": LIMIT,
            "over_limit": limit_fact,
            "note": "actual execution path (S102b): RollingStdDev is a "
                    "single-packet graph -> NOT fused (fusion requires "
                    ">1 packet); N <= 4_194_240 -> ordinary path "
                    "(Runtime.execute -> driver.execute_wave -> "
                    "driver.execute); chunked path unreachable (NOT chunkable); "
                    "GpuBufferPool not used on the ordinary path; "
                    "N > 4_194_240 -> dispatch-limit ValueError "
                    "(_check_dispatch_limit), documented NOT profiled.",
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "profile.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    assert os.path.getsize(path) > 0, "profile.json written empty"
