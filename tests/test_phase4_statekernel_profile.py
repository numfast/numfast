"""Phase 4 - STATEKERNEL profile (S90): 6 stages cold/warm.

Spec: STATEKERNEL_SLICE_SPEC.md sec.8 (S90).

Scenarios (actual execution path, recorded in profile.json):
  A: serial mode 0, N=64, 1 packet, dispatch=(1,1,1), dx=1 (fix applied).
  B: serial mode 0, N=1M (fix dx=1), 1 packet, dispatch=(1,1,1) -
     before fix dx=ceil(1M/64)=15625 O(dx*n), after dx=1 O(n).
  C: scan mode 0, chunk=256, N=1M, 3 packets (Local->Totals->Final),
     nblocks=ceil(1M/256)=3907, dx=15625 over-dispatch, guard AS-IS,
     fused via Runtime.execute (3 sequential packets, not chunked).
  D: mode 2 ST, N=1K, 1 packet, dispatch=(1,1,1), dx=1.

Stages: compile, alloc (create_buffer_with_data/create_buffer), h2d
(upload), dispatch, d2h (readback), pool_reuse (warm runs: uploads stable).

Evidence: evidence/statekernel_slice_phase4/profile.json

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

CHECKPOINT_SHA = "08575d7d8a57ec5f1de0fae974bbeb76bf650f07"
SLICE = "statekernel"
SEED = 42

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "statekernel_slice_phase4")

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


def _job_single_mode0():
    return {"op": "StateKernel_Single", "inputs": ["x"],
            "params": {"mode": 0, "a": 2.0 / 15.0, "b": 13.0 / 15.0},
            "out": "y"}


def _job_st():
    return {"op": "StateKernel_ST", "inputs": ["price", "upper", "lower"],
            "params": {"mode": 2}, "out": "y"}


def _scan_jobs_mode0(chunk=256):
    return [
        {"op": "StateKernelScanLocal", "inputs": ["data"],
         "params": {"mode": 0, "a": 2.0 / 15.0, "b": 13.0 / 15.0, "chunk": chunk},
         "out": ["skl_tmp", "skl_last"]},
        {"op": "StateKernelScanTotals", "inputs": ["skl_last"],
         "params": {"mode": 0, "a": 2.0 / 15.0, "b": 13.0 / 15.0, "chunk": chunk},
         "out": "skl_sp"},
        {"op": "StateKernelScanFinal", "inputs": ["skl_tmp", "skl_sp"],
         "params": {"mode": 0, "a": 2.0 / 15.0, "b": 13.0 / 15.0, "chunk": chunk},
         "out": "skl_out"},
    ]


def _wrap_execute_counter(driver):
    packets_seen = []
    orig_exec = driver.execute
    orig_fused = getattr(driver, "execute_fused", None)

    def wrapped_exec(packet):
        packets_seen.append(packet)
        return orig_exec(packet)

    driver.execute = wrapped_exec
    if orig_fused is not None:
        def wrapped_fused(packets):
            for p in packets:
                packets_seen.append(p)
            return orig_fused(packets)

        driver.execute_fused = wrapped_fused
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
    def profile_scenario(jobs, data, warm_runs):
        rt = _make_gpu_runtime()
        try:
            alloc_times = _wrap_alloc_timing(rt.driver._device)
            packets_seen = _wrap_execute_counter(rt.driver)
            t0 = time.perf_counter_ns()
            tasks = rt.compile(jobs)
            rt_compile_ns = time.perf_counter_ns() - t0
            t0 = time.perf_counter_ns()
            rt.execute(tasks, data)
            exec_wall_ns = time.perf_counter_ns() - t0
            chunks_observed = len(packets_seen)
            cold = {
                "compile_ns": rt_compile_ns
                              + _sum_stages(packets_seen, "compile_ns")
                              + _sum_stages(packets_seen, "pipeline_ns"),
                "alloc_ns": sum(alloc_times),
                "h2d_ns": _sum_stages(packets_seen, "upload_ns"),
                "dispatch_ns": _sum_stages(packets_seen, "dispatch_ns"),
                "d2h_ns": packets_seen[-1].profile.get("readback_ns", 0) if packets_seen else 0,
                "exec_wall_ns": exec_wall_ns,
                "latency_ns": rt_compile_ns + sum(alloc_times)
                              + _sum_stages(packets_seen, "upload_ns")
                              + _sum_stages(packets_seen, "dispatch_ns")
                              + (packets_seen[-1].profile.get("readback_ns", 0) if packets_seen else 0),
                "dispatch_x": packets_seen[0].profile.get("dispatch_x", 0) if packets_seen else 0,
                "dispatch_x_all": [p.profile.get("dispatch_x", 0) for p in packets_seen],
                "packets": chunks_observed,
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
                rt.execute(tasks, data)
                warm["exec_wall_ns"].append(time.perf_counter_ns() - t0)
                warm["alloc_ns"].append(sum(alloc_times))
                warm["h2d_ns"].append(_sum_stages(packets_seen, "upload_ns"))
                warm["dispatch_ns"].append(_sum_stages(packets_seen, "dispatch_ns"))
                warm["d2h_ns"].append(packets_seen[-1].profile.get("readback_ns", 0) if packets_seen else 0)
                warm["latency_ns"].append(
                    warm["compile_ns"][-1] + warm["alloc_ns"][-1]
                    + warm["h2d_ns"][-1] + warm["dispatch_ns"][-1]
                    + warm["d2h_ns"][-1])
            med = {k: int(statistics.median(v)) for k, v in warm.items()}
            return rt, cold, med, chunks_observed, rt.driver._pool.stats(), packets_seen
        except Exception:
            rt.driver.release()
            raise

    rng = np.random.default_rng(SEED)

    # Scenario A: serial mode 0, n=64, 1 packet, dx=1
    x64 = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
    rt_a, cold_a, warm_a, packets_a, pool_a, pkts_a = profile_scenario([_job_single_mode0()], {"x": x64}, 7)
    rt_a.driver.release()
    assert packets_a == 1, f"A n=64 packets={packets_a}"
    assert cold_a["dispatch_x"] == 1, f"A n=64 dx={cold_a['dispatch_x']}"

    # Scenario B: serial mode 0, n=1M, 1 packet, dx=1 (fix demonstrates 15625->1)
    x1m = (rng.standard_normal(1_000_000) * 5 + 30).astype(np.float32)
    rt_b, cold_b, warm_b, packets_b, pool_b, pkts_b = profile_scenario([_job_single_mode0()], {"x": x1m}, 5)
    rt_b.driver.release()
    assert packets_b == 1, f"B n=1M packets={packets_b}"
    assert cold_b["dispatch_x"] == 1, f"B n=1M dx={cold_b['dispatch_x']} expected 1 after fix (before fix 15625)"

    # Scenario C: scan mode 0, chunk=256, n=1M, 3 packets Local->Totals->Final
    n_c = 1_000_000
    chunk_c = 256
    nblocks_c = (n_c + chunk_c - 1) // chunk_c
    assert nblocks_c == 3907, f"nblocks={nblocks_c}"
    x_scan = (rng.standard_normal(n_c) * 5 + 30).astype(np.float32)
    rt_c, cold_c, warm_c, packets_c, pool_c, pkts_c = profile_scenario(_scan_jobs_mode0(chunk_c), {"data": x_scan}, 3)
    rt_c.driver.release()
    assert packets_c == 3, f"C scan packets={packets_c} expected 3 (Local/Totals/Final)"
    # scan over-dispatch: Local and Final dx=ceil(1M/64)=15625, Totals dx=ceil(3907/64)=62 (or 61)
    # check nblocks via output size if available
    # At least first packet should have dx=15625 (over-dispatch AS-IS)
    assert pkts_c[0].profile.get("dispatch_x", 0) == 15625 or pkts_c[0].profile.get("dispatch_x", 0) == 15625, f"C Local dx={pkts_c[0].profile.get('dispatch_x')}"

    # Scenario D: mode 2 ST, n=1000, 1 packet, dx=1
    rng_d = np.random.default_rng(SEED + 1)
    close = (rng_d.standard_normal(1000) * 5 + 30).astype(np.float32)
    upper = (rng_d.standard_normal(1000) * 5 + 35).astype(np.float32)
    lower = (rng_d.standard_normal(1000) * 5 + 25).astype(np.float32)
    rt_d, cold_d, warm_d, packets_d, pool_d, pkts_d = profile_scenario([_job_st()], {"price": close, "upper": upper, "lower": lower}, 5)
    rt_d.driver.release()
    assert packets_d == 1, f"D mode2 packets={packets_d}"
    assert cold_d["dispatch_x"] == 1, f"D mode2 dx={cold_d['dispatch_x']}"

    # pool reuse: warm runs uploads stable
    rt_f = _make_gpu_runtime()
    try:
        rng_f = np.random.default_rng(SEED)
        xf = (rng_f.standard_normal(64) * 5 + 30).astype(np.float32)
        tasks = rt_f.compile([_job_single_mode0()])
        rt_f.execute(tasks, {"x": xf})
        after_first = rt_f.driver._pool.stats()
        rt_f.execute(tasks, {"x": xf})
        after_warm = rt_f.driver._pool.stats()
        assert after_first["uploads"] == after_warm["uploads"], "warm run must not re-upload"
        pool_reuse = {
            "after_first": after_first,
            "after_warm": after_warm,
            "uploads_stable": after_first["uploads"] == after_warm["uploads"],
            "resident_stable": after_first["resident_bytes"] == after_warm["resident_bytes"],
            "note": "StateKernel n=64 serial path; GpuBufferPool not used on ordinary path (per-call create_buffer_with_data), so uploads=0 and stability holds trivially",
        }
    finally:
        rt_f.driver.release()

    evidence = {
        "phase": 4,
        "slice": SLICE,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": _versions(),
        "result": {
            "device": {"adapter": ADAPTER_INFO.get("device", "unknown"),
                       "backend": ADAPTER_INFO.get("backend_type", "unknown")},
            "stages": ["compile", "alloc", "h2d", "dispatch", "d2h", "pool_reuse"],
            "A_serial_n64_mode0": {"path": "serial mode0 (1 packet, dispatch=(1,1,1), dx=1)",
                                   "n": 64, "mode": 0, "packets": packets_a, "cold": cold_a, "warm_median": warm_a,
                                   "pool_after_warm": pool_a},
            "B_serial_n1M_mode0": {"path": "serial mode0 n=1M (1 packet, dispatch=(1,1,1), dx=1; before fix dx=15625 O(dx*n) -> after O(n))",
                                   "n": 1_000_000, "mode": 0, "packets": packets_b, "cold": cold_b, "warm_median": warm_b,
                                   "pool_after_warm": pool_b,
                                   "dispatch_fix": "dx=15625 -> 1, S87 plan.dispatch=(1,1,1)"},
            "C_scan_n1M_chunk256_mode0": {"path": "scan mode0 chunk=256 n=1M (3 packets Local->Totals->Final, nblocks=3907, fused via Runtime.execute, dx=15625 over-dispatch guard AS-IS)",
                                          "n": n_c, "chunk": chunk_c, "nblocks": nblocks_c, "mode": 0, "packets": packets_c, "cold": cold_c, "warm_median": warm_c,
                                          "pool_after_warm": pool_c,
                                          "dispatch_x_all": cold_c["dispatch_x_all"]},
            "D_mode2_n1000": {"path": "mode2 SuperTrend n=1000 (1 packet, dispatch=(1,1,1), dx=1)",
                              "n": 1000, "mode": 2, "packets": packets_d, "cold": cold_d, "warm_median": warm_d,
                              "pool_after_warm": pool_d},
            "pool_reuse": pool_reuse,
            "note": "S90: 6 stages cold/warm (compile/alloc/h2d/dispatch/d2h/pool_reuse); A serial n64 dx=1, B serial n1M dx=1 (fix 15625->1), C scan n1M 3 packets nblocks=3907 fused, D mode2 n1K dx=1; serial WGSL unchanged (workgroup_size(1) correct for 1 invocation), scan over-dispatch AS-IS (guard tid>=nblocks / id.x==0).",
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "profile.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    assert os.path.getsize(path) > 0, "profile.json written empty"
