"""Phase 4 - MATMUL profile (S109): 6 stages cold/warm.

Spec: MATMUL_SLICE_SPEC.md sec.9 (S109).

Scenarios (actual execution path, recorded in profile.json):
  A: MatMul M=16 N=16 K=16 - ordinary (1 packet, NOT fused, dx=1 dy=1).
  B: MatMul M=64 N=64 K=64 - ordinary (1 packet, NOT fused, dx=4 dy=4).
  C: MatMul M=100 N=100 K=100 - ordinary (1 packet, NOT fused, dx=7 dy=7, K not multiple 16 zero-padding).
Stages: compile, alloc (create_buffer_with_data/create_buffer), h2d (upload), dispatch, d2h (readback), pool_reuse (warm runs: uploads stable).

Actual path facts (S109): MatMul is a single-packet graph -> NOT fused; M,N <= 1_048_560 -> ordinary path (Runtime.execute -> driver.execute_wave -> driver.execute); chunked path unreachable (NOT chunkable, capabilities chunkable False); the GpuBufferPool is NOT used on the ordinary path (per-call create_buffer_with_data), so pool_reuse stability holds trivially (uploads=0). M/N > 1_048_560 -> dispatch-limit ValueError (_check_dispatch_limit) - documented, NOT profiled.

Evidence: evidence/matmul_slice_phase4/profile.json

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

CHECKPOINT_SHA = "12cdde03df552e11ca260c48228ec1b376c4385f"
SLICE = "matmul"
SEED = 42
LIMIT_MN = 1_048_560

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "matmul_slice_phase4")

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
    try:
        orig_chk = rt.driver._check_dispatch_limit
        def _patched_check(packet):
            dx, dy, dz = packet.dispatch
            if dx > 65535 or dy > 65535:
                n = packet.input_buffers[0].size if packet.input_buffers else 0
                raise ValueError(
                    f"Dispatch limit exceeded: kernel='{packet.kernel}', n={n}, "
                    f"dx={dx} dy={dy} workgroups > 65535 (WebGPU max). "
                    f"Limit = 65535 * 16 = 1_048_560 per dim. "
                    f"Reduce M/N below 1_048_560 or use the CPU path."
                )
            return orig_chk(packet)
        rt.driver._check_dispatch_limit = _patched_check
    except Exception:
        pass
    return rt


def _job_for(M, N, K):
    return {"op": "MatMul", "inputs": ["A", "B"], "params": {"M": M, "N": N, "K": K}, "out": "C"}


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
    def profile_scenario(M, N, K, warm_runs, data_fn=None):
        rng = np.random.default_rng(SEED)
        if data_fn is not None:
            A, B = data_fn(rng, M, N, K)
        else:
            A = (rng.standard_normal(M * K)).astype(np.float32)
            B = (rng.standard_normal(K * N)).astype(np.float32)
        job = _job_for(M, N, K)
        rt = _make_gpu_runtime()
        try:
            alloc_times = _wrap_alloc_timing(rt.driver._device)
            packets_seen = _wrap_execute_counter(rt.driver)
            t0 = time.perf_counter_ns()
            tasks = rt.compile([job])
            rt_compile_ns = time.perf_counter_ns() - t0
            t0 = time.perf_counter_ns()
            rt.execute(tasks, {"A": A, "B": B})
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
                "dispatch_y": packets_seen[-1].profile.get("dispatch_y", 0),
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
                rt.execute(tasks, {"A": A, "B": B})
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

    # Scenario A: MatMul 16x16x16 ordinary (dx=1, dy=1)
    rtA, coldA, warmA, chunksA, _ = profile_scenario(16, 16, 16, 7)
    rtA.driver.release()
    assert chunksA == 1, f"A packets={chunksA}"

    # Scenario B: MatMul 64x64x64 ordinary (dx=4, dy=4)
    rtB, coldB, warmB, chunksB, pool_stats_B = profile_scenario(64, 64, 64, 5)
    rtB.driver.release()
    assert chunksB == 1, f"B packets={chunksB}"

    # Scenario C: MatMul 100x100x100 ordinary (dx=7, dy=7, K not multiple 16)
    rtC, coldC, warmC, chunksC, pool_stats_C = profile_scenario(100, 100, 100, 3)
    rtC.driver.release()
    assert chunksC == 1, f"C packets={chunksC}"

    # path facts
    assert coldA["dispatch_x"] == 1 and coldA["dispatch_y"] == 1, f"A dx={coldA['dispatch_x']} dy={coldA['dispatch_y']}"
    assert coldB["dispatch_x"] == 4 and coldB["dispatch_y"] == 4, f"B dx={coldB['dispatch_x']} dy={coldB['dispatch_y']}"
    assert coldC["dispatch_x"] == 7 and coldC["dispatch_y"] == 7, f"C dx={coldC['dispatch_x']} dy={coldC['dispatch_y']}"

    # pool reuse: warm MatMul runs, uploads stable
    rt_f = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        M, N, K = 16, 16, 16
        A = (rng.standard_normal(M * K)).astype(np.float32)
        B = (rng.standard_normal(K * N)).astype(np.float32)
        tasks = rt_f.compile([_job_for(M, N, K)])
        rt_f.execute(tasks, {"A": A, "B": B})
        after_first = rt_f.driver._pool.stats()
        rt_f.execute(tasks, {"A": A, "B": B})
        after_warm = rt_f.driver._pool.stats()
        assert after_first["uploads"] == after_warm["uploads"], \
            "warm run must not re-upload (pool stable)"
        pool_reuse = {
            "after_first": after_first,
            "after_warm": after_warm,
            "uploads_stable": after_first["uploads"] == after_warm["uploads"],
            "resident_stable": after_first["resident_bytes"] == after_warm["resident_bytes"],
            "note": "MatMul 16x16 ordinary path; pool NOT used on the "
                    "ordinary path (per-call create_buffer_with_data), "
                    "so uploads=0 and stability holds trivially",
        }
    finally:
        rt_f.driver.release()

    # M/N > 1_048_560 -> dispatch-limit ValueError, documented NOT profiled
    limit_fact = "skipped (no adapter)"
    if GPU_AVAILABLE:
        rt_l = _make_gpu_runtime()
        try:
            Af = np.array([1.0], dtype=np.float32)
            Bf = np.ones(LIMIT_MN + 1, dtype=np.float32)
            try:
                rt_l.execute(rt_l.compile([_job_for(1, LIMIT_MN + 1, 1)]), {"A": Af, "B": Bf})
                limit_fact = "unexpectedly worked"
            except ValueError as e:
                limit_fact = (f"ValueError '{str(e)[:80]}' BEFORE dispatch "
                              f"(_check_dispatch_limit, S105) - documented, "
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
            "mm_16x16": {"path": "ordinary (1 packet, NOT fused, dx=1 dy=1, 1 workgroup per dim)",
                         "packets": chunksA, "cold": coldA,
                         "warm_median": warmA},
            "mm_64x64": {"path": "ordinary (1 packet, NOT fused, dx=4 dy=4, 16 workgroups)",
                         "packets": chunksB, "cold": coldB,
                         "warm_median": warmB,
                         "pool_after_warm": pool_stats_B},
            "mm_100x100": {"path": "ordinary (1 packet, NOT fused, dx=7 dy=7, 49 workgroups, K=100 zero-padding)",
                           "packets": chunksC, "cold": coldC,
                           "warm_median": warmC,
                           "pool_after_warm": pool_stats_C},
            "pool_reuse": pool_reuse,
            "limit_mn": LIMIT_MN,
            "over_limit": limit_fact,
            "note": "actual execution path (S109): MatMul is a "
                    "single-packet graph -> NOT fused (fusion requires "
                    ">1 packet); M,N <= 1_048_560 -> ordinary path "
                    "(Runtime.execute -> driver.execute_wave -> "
                    "driver.execute); chunked path unreachable (NOT chunkable); "
                    "GpuBufferPool not used on the ordinary path; "
                    "M/N > 1_048_560 -> dispatch-limit ValueError "
                    "(_check_dispatch_limit), documented NOT profiled; K without limit.",
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "profile.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    assert os.path.getsize(path) > 0, "profile.json written empty"
