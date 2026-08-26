"""Phase 4 - SCAN profile (S33): 6 stages cold/warm AFTER the Q1 fix.

Spec: SCAN_SLICE_SPEC.md sec.8 (S33).

Scenarios: ScanLocal n=64 (ordinary single-packet path) and Scan chain
(Local+Totals+Final) n=1M (fused path - rt.execute routes multi-packet
chains through ExecutionScheduler -> execute_fused with WebGpuDriver).
Stages: compile, alloc (create_buffer_with_data/create_buffer), h2d
(upload), dispatch, d2h (readback), pool_reuse (fused chain warm runs:
uploads stable). Same instrumentation as Phase 2
(execution_phase2/instrumentation.json).

Before (reference): execution_phase2/C.json fused readback_ns n64=9,966,800,
n1M=10,958,100; live pre-fix warm median n64=2,567,200, n1M=10,017,500.

Evidence: evidence/scan_slice_phase4/profile.json

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
from Runtime._lib.planner import planner
from Runtime._lib.optimizer import optimize_graph
from Runtime._lib.builder import builder
from Compute import register_all

CHECKPOINT_SHA = "7be0094"
SEED = 42
SCANLOCAL_JOBS = [{"op": "ScanLocal", "inputs": ["data"], "params": {"op": 0},
                   "out": ["scan", "_bsum"]}]
SCAN_CHAIN_JOBS = [
    {"op": "ScanLocal", "inputs": ["data"], "params": {"op": 0},
     "out": ["scan_local", "block_sum"]},
    {"op": "ScanTotals", "inputs": ["block_sum"], "params": {"op": 0},
     "out": "block_prefix"},
    {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"],
     "params": {"op": 0}, "out": "scan"},
]

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "scan_slice_phase4")

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


def _fused_build(rt, source_data):
    rt.driver.kernel_table = rt.kernel_table
    tasks = rt.compile(SCAN_CHAIN_JOBS)
    graph = planner(tasks, rt.kernel_table)
    graph = optimize_graph(graph, level=rt.optimizer_level)
    return builder(graph, source_data, rt.driver, rt.kernel_table)


def _sum_stages(packets, key):
    return sum(p.profile.get(key, 0) for p in packets)


@needs_gpu
def test_profile_6_stages():
    def profile_ordinary(jobs, n, warm_runs):
        """Single-packet path (ScanLocal): rt.execute -> ordinary execute()."""
        x = (np.random.default_rng(SEED).standard_normal(n) * 5 + 30).astype(np.float32)
        rt = _make_gpu_runtime()
        try:
            alloc_times = _wrap_alloc_timing(rt.driver._device)
            packets_seen = _wrap_execute_counter(rt.driver)
            t0 = time.perf_counter_ns()
            tasks = rt.compile(jobs)
            rt_compile_ns = time.perf_counter_ns() - t0
            t0 = time.perf_counter_ns()
            rt.execute(tasks, {"data": x})
            exec_wall_ns = time.perf_counter_ns() - t0
            last = packets_seen[-1]
            cold = {
                "compile_ns": rt_compile_ns + last.profile.get("compile_ns", 0)
                              + last.profile.get("pipeline_ns", 0),
                "alloc_ns": sum(alloc_times),
                "h2d_ns": last.profile.get("upload_ns", 0),
                "dispatch_ns": last.profile.get("dispatch_ns", 0),
                "d2h_ns": last.profile.get("readback_ns", 0),
                "exec_wall_ns": exec_wall_ns,
                "latency_ns": rt_compile_ns + sum(alloc_times)
                              + last.profile.get("upload_ns", 0)
                              + last.profile.get("dispatch_ns", 0)
                              + last.profile.get("readback_ns", 0),
            }
            warm = {"compile_ns": [], "alloc_ns": [], "h2d_ns": [],
                    "dispatch_ns": [], "d2h_ns": [], "exec_wall_ns": [],
                    "latency_ns": []}
            for _ in range(warm_runs):
                alloc_times.clear()
                packets_seen.clear()
                t0 = time.perf_counter_ns()
                tasks = rt.compile(jobs)
                warm["compile_ns"].append(time.perf_counter_ns() - t0)
                t0 = time.perf_counter_ns()
                rt.execute(tasks, {"data": x})
                warm["exec_wall_ns"].append(time.perf_counter_ns() - t0)
                p = packets_seen[-1]
                warm["alloc_ns"].append(sum(alloc_times))
                warm["h2d_ns"].append(p.profile.get("upload_ns", 0))
                warm["dispatch_ns"].append(p.profile.get("dispatch_ns", 0))
                warm["d2h_ns"].append(p.profile.get("readback_ns", 0))
                warm["latency_ns"].append(
                    warm["compile_ns"][-1] + warm["alloc_ns"][-1]
                    + warm["h2d_ns"][-1] + warm["dispatch_ns"][-1]
                    + warm["d2h_ns"][-1])
            med = {k: int(statistics.median(v)) for k, v in warm.items()}
            return rt, cold, med
        except Exception:
            rt.driver.release()
            raise

    def profile_fused(n, warm_runs):
        """Chain path: planner+optimizer+builder -> execute_fused (pool)."""
        x = (np.random.default_rng(SEED).standard_normal(n) * 5 + 30).astype(np.float32)
        rt = _make_gpu_runtime()
        try:
            alloc_times = _wrap_alloc_timing(rt.driver._device)
            t0 = time.perf_counter_ns()
            tasks = rt.compile(SCAN_CHAIN_JOBS)
            rt_compile_ns = time.perf_counter_ns() - t0

            alloc_times.clear()
            packets = _fused_build(rt, {"data": x})
            t0 = time.perf_counter_ns()
            rt.driver.execute_fused(packets)
            exec_wall_ns = time.perf_counter_ns() - t0
            cold = {
                "compile_ns": rt_compile_ns
                              + _sum_stages(packets, "compile_ns")
                              + _sum_stages(packets, "pipeline_ns"),
                "alloc_ns": sum(alloc_times),
                "h2d_ns": _sum_stages(packets, "upload_ns"),
                "dispatch_ns": _sum_stages(packets, "dispatch_ns"),
                "d2h_ns": packets[-1].profile.get("readback_ns", 0),
                "exec_wall_ns": exec_wall_ns,
                "latency_ns": rt_compile_ns + sum(alloc_times)
                              + _sum_stages(packets, "upload_ns")
                              + _sum_stages(packets, "dispatch_ns")
                              + packets[-1].profile.get("readback_ns", 0),
            }
            warm = {"compile_ns": [], "alloc_ns": [], "h2d_ns": [],
                    "dispatch_ns": [], "d2h_ns": [], "exec_wall_ns": [],
                    "latency_ns": []}
            for _ in range(warm_runs):
                alloc_times.clear()
                t0 = time.perf_counter_ns()
                tasks = rt.compile(SCAN_CHAIN_JOBS)
                warm["compile_ns"].append(time.perf_counter_ns() - t0)
                packets = _fused_build(rt, {"data": x})
                t0 = time.perf_counter_ns()
                rt.driver.execute_fused(packets)
                warm["exec_wall_ns"].append(time.perf_counter_ns() - t0)
                warm["alloc_ns"].append(sum(alloc_times))
                warm["h2d_ns"].append(_sum_stages(packets, "upload_ns"))
                warm["dispatch_ns"].append(_sum_stages(packets, "dispatch_ns"))
                warm["d2h_ns"].append(packets[-1].profile.get("readback_ns", 0))
                warm["latency_ns"].append(
                    warm["compile_ns"][-1] + warm["alloc_ns"][-1]
                    + warm["h2d_ns"][-1] + warm["dispatch_ns"][-1]
                    + warm["d2h_ns"][-1])
            med = {k: int(statistics.median(v)) for k, v in warm.items()}
            return rt, cold, med, rt.driver._pool.stats()
        except Exception:
            rt.driver.release()
            raise

    # Scenario A: ScanLocal n=64 (ordinary, cold + 7 warm)
    rt64, cold64, warm64 = profile_ordinary(SCANLOCAL_JOBS, 64, 7)
    rt64.driver.release()
    # Scenario B: Scan chain n=1M (fused, cold + 5 warm)
    rt1m, cold1m, warm1m, pool_stats = profile_fused(1_000_000, 5)
    rt1m.driver.release()

    # pool reuse: fused 3-packet chain, warm uploads stable
    rt_f = _make_gpu_runtime()
    try:
        x = (np.random.default_rng(SEED).standard_normal(64) * 5 + 30).astype(np.float32)
        packets = _fused_build(rt_f, {"data": x})
        rt_f.driver.execute_fused(packets)
        after_first = rt_f.driver._pool.stats()
        rt_f.driver.execute_fused(packets)  # warm: mapping reuse, no upload
        after_warm = rt_f.driver._pool.stats()
        assert after_first["uploads"] == after_warm["uploads"], \
            "warm fused run must not re-upload"
        pool_reuse = {
            "after_first": after_first,
            "after_warm": after_warm,
            "uploads_stable": after_first["uploads"] == after_warm["uploads"],
            "resident_stable": after_first["resident_bytes"] == after_warm["resident_bytes"],
            "note": "fused Scan chain n=64; pool reuse measured on warm runs",
        }
    finally:
        rt_f.driver.release()

    evidence = {
        "phase": 4,
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
            "scanlocal_n64": {"path": "ordinary (1 packet)", "cold": cold64,
                              "warm_median": warm64},
            "chain_n1M": {"path": "fused (3 packets, execute_fused)",
                          "cold": cold1m, "warm_median": warm1m,
                          "pool_after_warm": pool_stats},
            "pool_reuse": pool_reuse,
            "readback_before_ref": {
                "c_json_fused_n64": 9966800,
                "c_json_fused_n1M": 10958100,
                "live_pre_fix_warm_n64": 2567200,
                "live_pre_fix_warm_n1M": 10017500,
                "note": "from execution_phase2/C.json + pre-fix warm measure "
                        "on checkpoint 7be0094; after-fix values are the "
                        "d2h_ns medians above (ScanLocal n64 / chain n1M "
                        "final packet); exact fused before/after pair in "
                        "q1_fix.json",
            },
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "profile.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    assert os.path.getsize(path) > 0, "profile.json written empty"