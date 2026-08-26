"""Phase 4 - MAP profile (S44): 6 stages cold/warm.

Spec: MAP_SLICE_SPEC.md sec.8 (S44).

Scenarios (actual execution path per S40, recorded in chunking.json):
  Map n=64         - ordinary (1 packet, NOT fused: single-packet graph).
  Map n=1M         - ordinary (<= dispatch limit, 1 packet, NOT fused).
  Map n=4_200_000  - chunked (2 chunks via ExecutionScheduler
                     _execute_chunked, per-chunk execute_wave).
Stages: compile, alloc (create_buffer_with_data/create_buffer), h2d
(upload), dispatch, d2h (readback), pool_reuse (warm runs: uploads stable).
Same instrumentation as Phase 2 (execution_phase2/instrumentation.json).

Evidence: evidence/map_slice_phase4/profile.json

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

CHECKPOINT_SHA = "91fe56e"
SLICE = "map"
SEED = 42
MAP_JOBS = [{"op": "Map", "inputs": ["data"], "params": {"func": 0}}]
LIMIT = 4_194_240

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "map_slice_phase4")

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}

needs_gpu = pytest.mark.skipif(not GPU_AVAILABLE,
                               reason="WebGPU adapter unavailable")


def _make_gpu_runtime():
    rt = Runtime(driver=WebGpuDriver())
    register_all(rt)
    return rt


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
    def profile_scenario(n, warm_runs, expected_path, expected_packets):
        """Map profile: rt.execute path (ordinary or chunked).

        Returns (cold, warm_median, chunks_observed, pool_stats).
        """
        x = (np.random.default_rng(SEED).standard_normal(n) * 5 + 30).astype(np.float32)
        rt = _make_gpu_runtime()
        try:
            alloc_times = _wrap_alloc_timing(rt.driver._device)
            packets_seen = _wrap_execute_counter(rt.driver)
            t0 = time.perf_counter_ns()
            tasks = rt.compile(MAP_JOBS)
            rt_compile_ns = time.perf_counter_ns() - t0
            t0 = time.perf_counter_ns()
            rt.execute(tasks, {"data": x})
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
            }
            warm = {"compile_ns": [], "alloc_ns": [], "h2d_ns": [],
                    "dispatch_ns": [], "d2h_ns": [], "exec_wall_ns": [],
                    "latency_ns": []}
            for _ in range(warm_runs):
                alloc_times.clear()
                packets_seen.clear()
                t0 = time.perf_counter_ns()
                tasks = rt.compile(MAP_JOBS)
                warm["compile_ns"].append(time.perf_counter_ns() - t0)
                t0 = time.perf_counter_ns()
                rt.execute(tasks, {"data": x})
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

    # Scenario A: Map n=64 (ordinary, 1 packet, cold + 7 warm)
    rt64, cold64, warm64, chunks64, _ = profile_scenario(64, 7, "ordinary", 1)
    rt64.driver.release()
    assert chunks64 == 1, f"n=64 chunks={chunks64}"

    # Scenario B: Map n=1M (ordinary, <= limit, 1 packet, cold + 5 warm)
    rt1m, cold1m, warm1m, chunks1m, pool_stats_1m = profile_scenario(
        1_000_000, 5, "ordinary", 1)
    rt1m.driver.release()
    assert chunks1m == 1, f"n=1M chunks={chunks1m}"

    # Scenario C: Map n=4_200_000 (chunked, 2 chunks, cold + 3 warm)
    rt4m, cold4m, warm4m, chunks4m, pool_stats_4m = profile_scenario(
        4_200_000, 3, "chunked", 2)
    rt4m.driver.release()
    assert chunks4m == 2, f"n=4_200_000 chunks={chunks4m}"

    # pool reuse: warm Map runs, uploads stable
    rt_f = _make_gpu_runtime()
    try:
        x = (np.random.default_rng(SEED).standard_normal(64) * 5 + 30).astype(np.float32)
        tasks = rt_f.compile(MAP_JOBS)
        rt_f.execute(tasks, {"data": x})
        after_first = rt_f.driver._pool.stats()
        rt_f.execute(tasks, {"data": x})  # warm: mapping reuse, no upload
        after_warm = rt_f.driver._pool.stats()
        assert after_first["uploads"] == after_warm["uploads"], \
            "warm run must not re-upload"
        pool_reuse = {
            "after_first": after_first,
            "after_warm": after_warm,
            "uploads_stable": after_first["uploads"] == after_warm["uploads"],
            "resident_stable": after_first["resident_bytes"] == after_warm["resident_bytes"],
            "note": "Map n=64 ordinary path; pool reuse measured on warm runs",
        }
    finally:
        rt_f.driver.release()

    evidence = {
        "phase": 4,
        "slice": SLICE,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "wgpu": wgpu.__version__ if GPU_AVAILABLE else "n/a",
        },
        "result": {
            "device": {"adapter": ADAPTER_INFO.get("device", "unknown"),
                       "backend": ADAPTER_INFO.get("backend_type", "unknown")},
            "stages": ["compile", "alloc", "h2d", "dispatch", "d2h",
                       "pool_reuse"],
            "map_n64": {"path": "ordinary (1 packet, NOT fused)",
                        "chunks": chunks64, "cold": cold64,
                        "warm_median": warm64},
            "map_n1M": {"path": "ordinary (<= limit 4_194_240, 1 packet, "
                                "NOT fused)",
                        "chunks": chunks1m, "cold": cold1m,
                        "warm_median": warm1m,
                        "pool_after_warm": pool_stats_1m},
            "map_n4200000": {"path": "chunked (ExecutionScheduler "
                                     "_execute_chunked, 2 chunks, "
                                     "per-chunk execute_wave)",
                             "chunks": chunks4m, "cold": cold4m,
                             "warm_median": warm4m,
                             "pool_after_warm": pool_stats_4m},
            "pool_reuse": pool_reuse,
            "limit": LIMIT,
            "note": "actual execution path (S40): Map is a single-packet "
                    "graph -> NOT fused (fusion requires >1 packet); "
                    "N <= 4_194_240 -> ordinary path; N > limit -> chunked "
                    "path. Recorded in chunking.json.",
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "profile.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    assert os.path.getsize(path) > 0, "profile.json written empty"