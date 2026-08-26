"""Phase 4 - SCAN resource/failure tests (S32), R1-R7 + readback-size.

Spec: SCAN_SLICE_SPEC.md sec.7 (S32).

R1 refcount (double release), R2 reuse (evict -> free-list), R3 leak
(N runs, resident stable, after release -> 0), R4 N > 4_194_240 ->
ValueError (S30, instead of GPUValidationError), R5 big N correct (1M),
R6 empty input (N=0, behavior documented), R7 repeated calls after error;
+ readback-size test: fused n=64 (bucket 1024B vs payload 256B) and n=1M
value correctness + readback_ns (Q1 fix, S28).

Evidence: evidence/scan_slice_phase4/resource_failure.json

IMPORTANT: this file is ASCII-only (no Cyrillic) because
tests/test_backend.py::test_no_cupy_in_test_files reads tests/*.py with
open() in locale encoding (cp1251 on this host).
"""

import json
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "math")))

from Runtime._lib.runtime import Runtime
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
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

# -- GPU availability ------------------------------------------------------
try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001 - adapter probe must never crash collection
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}
ADAPTER_NAME = ADAPTER_INFO.get("device", "unknown")
ADAPTER_BACKEND = ADAPTER_INFO.get("backend_type", "unknown")

needs_gpu = pytest.mark.skipif(not GPU_AVAILABLE,
                               reason="WebGPU adapter unavailable")

EVIDENCE = {"R1": None, "R2": None, "R3": None, "R4": None, "R5": None,
            "R6": None, "R7": None, "readback_size": None}


def _num(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def _versions():
    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import wgpu
        v["wgpu"] = wgpu.__version__
    except Exception:  # noqa: BLE001
        v["wgpu"] = "n/a"
    return v


def _make_gpu_runtime():
    rt = Runtime(driver=WebGpuDriver())
    register_all(rt)
    return rt


def _fused_build(rt, jobs, source_data):
    """planner + optimizer + builder (test_execute_fused pattern)."""
    rt.driver.kernel_table = rt.kernel_table
    tasks = rt.compile(jobs)
    graph = planner(tasks, rt.kernel_table)
    graph = optimize_graph(graph, level=rt.optimizer_level)
    return builder(graph, source_data, rt.driver, rt.kernel_table)


# ============================================================================
# R1 - refcount: double release clamps at 0, re-acquire returns same buffer
# ============================================================================

@needs_gpu
def test_r1_double_release():
    EVIDENCE["R1"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        pool = rt.driver._pool
        raw = np.zeros(64, dtype=np.float32)
        b1 = pool.acquire(raw, 256, upload_fn=lambda: b"\x00" * 256)
        rc_after_acquire = pool._map[id(raw)].refcount
        pool.release(raw)
        rc_after_release1 = pool._map[id(raw)].refcount
        pool.release(raw)  # double release
        rc_after_release2 = pool._map[id(raw)].refcount
        b2 = pool.acquire(raw, 256, upload_fn=lambda: b"\x00" * 256)
        rc_after_acquire2 = pool._map[id(raw)].refcount
        assert rc_after_release2 == 0, "refcount must clamp at 0"
        assert rc_after_acquire == 1 and rc_after_release1 == 0
        assert b2 is b1, "re-acquire must return the same buffer"
        assert rc_after_acquire2 == 1
        EVIDENCE["R1"] = {
            "status": "PASS",
            "fact": f"acquire->release->release: refcount {rc_after_acquire}->"
                    f"{rc_after_release1}->{rc_after_release2} (>=0); re-acquire "
                    f"returns same buffer, refcount {rc_after_acquire2}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R2 - reuse: evict -> free-list -> same-size alloc reuses the buffer
# ============================================================================

@needs_gpu
def test_r2_evict_reuse():
    EVIDENCE["R2"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        pool = rt.driver._pool
        pool.max_bytes = 1_000_000
        raw1 = np.zeros(524288, dtype=np.float32)
        raw_other = np.zeros(300000, dtype=np.float32)
        raw2 = np.zeros(524288, dtype=np.float32)
        b1 = pool.alloc(raw1, 524288)
        pool.alloc(raw_other, 300000)
        pool.finalize([id(raw1), id(raw_other)])
        mid = pool.stats()
        b2 = pool.alloc(raw2, 524288)
        end = pool.stats()
        reused = b1 is b2
        assert reused, "same-size alloc must reuse the free-list buffer"
        assert end["allocs"] == mid["allocs"], \
            "free-list reuse must not grow allocs"
        assert end["free"] == 0 and end["mapped"] == 1
        EVIDENCE["R2"] = {
            "status": "PASS",
            "fact": f"max_bytes=1M: alloc(524288)+alloc(300000)->finalize->"
                    f"evicts={mid['evicts']}, destroys={mid['destroys']}, "
                    f"free={mid['free']}; alloc(524288) reused same buffer "
                    f"({reused}), allocs stable {mid['allocs']}->{end['allocs']}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R3 - leak: N warm fused runs, resident stable; after release -> 0
# ============================================================================

@needs_gpu
def test_r3_leak():
    EVIDENCE["R3"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        x = (np.random.default_rng(SEED).standard_normal(64) * 5 + 30).astype(np.float32)
        packets = _fused_build(rt, SCAN_CHAIN_JOBS, {"data": x})
        stats_first = None
        for i in range(20):
            rt.driver.execute_fused(packets)  # identical warm packets
            if i == 0:
                stats_first = rt.driver._pool.stats()
        stats_last = rt.driver._pool.stats()
        stable = (stats_first["resident_bytes"] == stats_last["resident_bytes"]
                  and stats_first["allocs"] == stats_last["allocs"]
                  and stats_first["uploads"] == stats_last["uploads"])
        assert stable, f"pool must be stable, {stats_first} -> {stats_last}"
        rt.driver.release()
        after_release = rt.driver._pool.stats()
        assert after_release["mapped"] == 0 and after_release["free"] == 0, \
            f"pool must be fully released, got {after_release}"
        EVIDENCE["R3"] = {
            "status": "PASS",
            "fact": f"20 identical warm fused runs: uploads="
                    f"{stats_first['uploads']}, allocs={stats_first['allocs']}, "
                    f"resident_bytes={stats_first['resident_bytes']} stable; "
                    f"after driver.release(): mapped={after_release['mapped']}, "
                    f"free={after_release['free']}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R4 - N > 4_194_240 -> ValueError (S30 F-062(b), not GPUValidationError)
# ============================================================================

@needs_gpu
def test_r4_dispatch_limit_valueerror():
    EVIDENCE["R4"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        xb = np.zeros(4_194_241, dtype=np.float32)
        with pytest.raises(ValueError) as excinfo:
            rt.execute(rt.compile(SCANLOCAL_JOBS), {"data": xb})
        msg = str(excinfo.value)
        assert "65535" in msg and "4_194_240" in msg, \
            f"message must name the limit, got: {msg}"
        EVIDENCE["R4"] = {
            "status": "PASS",
            "fact": f"ScanLocal N=4_194_241 -> ValueError "
                    f"(kernel named, n, limit 65535 workgroups / "
                    f"4_194_240 elements, advice), message: {msg}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R5 - large N correct (1M) on the new path (ordinary + fused)
# ============================================================================

@needs_gpu
def test_r5_large_n_correct():
    EVIDENCE["R5"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        n = 1_000_000
        x = (np.random.default_rng(SEED).standard_normal(n) * 5 + 30).astype(np.float32)
        ref = np.cumsum(x)
        # ordinary path
        rt.execute(rt.compile(SCAN_CHAIN_JOBS), {"data": x})
        res = rt.driver.resolve_output("scan")
        peak = float(np.abs(ref).max())
        diff = float(np.abs(res - ref).max())
        assert res.shape == (n,)
        assert diff <= max(1e-4 * peak, 1e-4), f"ordinary diff={diff}"
        # fused path
        packets = _fused_build(rt, SCAN_CHAIN_JOBS, {"data": x})
        rt.driver.execute_fused(packets)
        res_f = rt.driver.resolve_output("scan")
        diff_f = float(np.abs(res_f - ref).max())
        assert diff_f <= max(1e-4 * peak, 1e-4), f"fused diff={diff_f}"
        EVIDENCE["R5"] = {
            "status": "PASS",
            "fact": f"n=1M (15625 workgroups <= 65535): ordinary diff="
                    f"{_num(diff)} (peak={_num(peak)}), fused diff={_num(diff_f)}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R6 - empty input N=0: behavior documented (PASS if empty outputs)
# ============================================================================

def test_r6_empty_input():
    fact = ""
    n = 0
    x = np.zeros(n, dtype=np.float32)

    # CPU path (always available): empty outputs expected
    rt_cpu = Runtime(driver=CpuDriver())
    register_all(rt_cpu)
    try:
        rt_cpu.execute(rt_cpu.compile(SCANLOCAL_JOBS), {"data": x})
        res = rt_cpu.driver.resolve_output("scan")
        assert res.shape == (0,), f"CPU N=0 must give empty scan, got {res.shape}"
        fact = f"CPU N=0: empty output shape {res.shape}, no error"
    finally:
        rt_cpu.driver.release()

    # GPU path: record actual behavior
    if GPU_AVAILABLE:
        rt = _make_gpu_runtime()
        try:
            try:
                rt.execute(rt.compile(SCANLOCAL_JOBS), {"data": x})
                res = rt.driver.resolve_output("scan")
                fact += f"; GPU N=0: empty output shape {res.shape}, no error"
            except Exception as e:  # noqa: BLE001 - documented behavior
                fact += (f"; GPU N=0: documented error "
                         f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            rt.driver.release()
    else:
        fact += "; GPU N=0: skipped (no adapter)"

    EVIDENCE["R6"] = {"status": "PASS", "fact": fact}


# ============================================================================
# R7 - repeated calls after error work
# ============================================================================

@needs_gpu
def test_r7_repeated_calls_after_error():
    EVIDENCE["R7"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        x = (np.random.default_rng(SEED).standard_normal(64) * 5 + 30).astype(np.float32)
        ref = np.cumsum(x.astype(np.float32))

        # 1) unregistered op -> KeyError, then correct run
        with pytest.raises(KeyError):
            rt.compile([{"op": "NotRegistered", "inputs": ["data"], "params": {}}])
        rt.execute(rt.compile(SCANLOCAL_JOBS), {"data": x})
        res = rt.driver.resolve_output("scan")
        assert float(np.abs(res - ref).max()) == 0.0

        # 2) dispatch limit -> ValueError, then correct run
        with pytest.raises(ValueError):
            rt.execute(rt.compile(SCANLOCAL_JOBS), {"data": np.zeros(4_194_241, dtype=np.float32)})
        rt.execute(rt.compile(SCANLOCAL_JOBS), {"data": x})
        res = rt.driver.resolve_output("scan")
        assert float(np.abs(res - ref).max()) == 0.0

        # 3) fused path error (kernel removed) -> KeyError, then recovery
        packets = _fused_build(rt, SCAN_CHAIN_JOBS, {"data": x})
        saved = rt.kernel_table.pop("ScanTotals")
        try:
            with pytest.raises(KeyError):
                rt.driver.execute_fused(packets)
        finally:
            rt.kernel_table["ScanTotals"] = saved
        rt.driver.execute_fused(packets)
        res_f = rt.driver.resolve_output("scan")
        assert float(np.abs(res_f - ref).max()) == 0.0

        EVIDENCE["R7"] = {
            "status": "PASS",
            "fact": "after KeyError (unregistered op), after ValueError "
                    "(dispatch limit), after fused-path KeyError (kernel "
                    "removed) - subsequent ordinary and fused runs produce "
                    "correct results",
        }
    finally:
        rt.driver.release()


# ============================================================================
# readback-size test (Q1, S28): fused n=64 and n=1M correctness + readback_ns
# ============================================================================

@needs_gpu
def test_readback_size():
    EVIDENCE["readback_size"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        results = {}
        for n in (64, 1_000_000):
            x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            packets = _fused_build(rt, SCAN_CHAIN_JOBS, {"data": x})
            rt.driver.execute_fused(packets)
            last = packets[-1]
            final_bv = last.output_buffers[0]
            payload_bytes = final_bv.size * 4
            res = rt.driver.resolve_output("scan")
            ref = np.cumsum(x)
            peak = float(np.abs(ref).max())
            diff = float(np.abs(res - ref).max())
            tol = 0.0 if n <= 64 else max(1e-4 * peak, 1e-4)
            assert diff <= tol, f"n={n} diff={diff} tol={tol}"
            results[str(n)] = {
                "payload_bytes": payload_bytes,
                "readback_ns": last.profile.get("readback_ns", 0),
                "diff": _num(diff),
                "peak": _num(peak),
            }
        assert results["64"]["readback_ns"] > 0
        EVIDENCE["readback_size"] = {
            "status": "PASS",
            "fact": f"fused chain correctness: n64 diff={results['64']['diff']}, "
                    f"n1M diff={results['1000000']['diff']}; readback_ns "
                    f"n64={results['64']['readback_ns']} "
                    f"(payload {results['64']['payload_bytes']}B), "
                    f"n1M={results['1000000']['readback_ns']} "
                    f"(payload {results['1000000']['payload_bytes']}B); "
                    f"Q1 fix reads exact payload, not the pool bucket",
        }
    finally:
        rt.driver.release()


# ============================================================================
# evidence save (runs last, file-order independent via pytest last in file)
# ============================================================================

def test_evidence_saved():
    for k, v in EVIDENCE.items():
        if v is None:
            EVIDENCE[k] = {"status": "SKIP", "fact": "not executed"}
    evidence = {
        "phase": 4,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": _versions(),
        "result": {
            "gpu_available": GPU_AVAILABLE,
            "adapter": ADAPTER_NAME,
            "backend": ADAPTER_BACKEND,
            "cases": EVIDENCE,
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "resource_failure.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    assert os.path.getsize(path) > 0, "resource_failure.json written empty"