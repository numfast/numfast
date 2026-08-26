"""Phase 4 - MAPBINARY resource/failure tests (S56), R1-R7 + readback-size.

Spec: MAPBINARY_SLICE_SPEC.md sec.10 (S56).

R1 refcount (double release clamps at 0, re-acquire returns same buffer).
R2 reuse (evict -> free-list -> same-size alloc reuses the buffer).
R3 leak (20 warm runs: uploads/allocs/resident stable; after release -> 0).
R4-MapBinary: unknown op (99) -> ValueError BEFORE dispatch (descriptor
validation S49, GPU never submitted - execute/execute_wave never called).
R5 large N (4_200_000 / 10_000_000): chunking correct (S40 proof,
chunk count via execute_wave calls, results within contract tolerance).
R6 empty input N=0: CPU empty output without error; GPU N=0 documented
(dispatch 0 -> ValueError from wgpu-py readback/mapped-size).
R7 repeated calls after error (unknown op, empty input, a-len-1 guard
scenario) - pool not broken, subsequent runs correct.
+ readback-size: MapBinary n=64 / n=1M correctness (D2H by exact size).

Evidence: evidence/mapbinary_slice_phase4/resource_failure.json

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
from Compute import register_all

CHECKPOINT_SHA = "2be93f4"
SLICE = "mapbinary"
SEED = 42
MB_JOBS = [{"op": "MapBinary", "inputs": ["a", "b"],
            "params": {"op": 0}, "out": "y"}]
LIMIT = 4_194_240

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "mapbinary_slice_phase4")

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


def _tol(peak):
    return max(1e-6 * peak, 1e-6)


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


def _add_ref(a, b):
    return a.astype(np.float64) + b.astype(np.float64)


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
# R3 - leak: 20 warm MapBinary runs, pool stable; after release -> 0
# ============================================================================

@needs_gpu
def test_r3_leak():
    EVIDENCE["R3"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        x = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        y = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        tasks = rt.compile(MB_JOBS)
        stats_first = None
        for i in range(20):
            rt.execute(tasks, {"a": x, "b": y})
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
            "fact": f"20 identical warm MapBinary runs: uploads="
                    f"{stats_first['uploads']}, allocs={stats_first['allocs']}, "
                    f"resident_bytes={stats_first['resident_bytes']} stable; "
                    f"after driver.release(): mapped={after_release['mapped']}, "
                    f"free={after_release['free']}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R4-MapBinary - unknown op -> ValueError BEFORE dispatch (descriptor S49)
# ============================================================================

@needs_gpu
def test_r4_op_validation_before_dispatch():
    EVIDENCE["R4"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        calls = {"execute": 0, "execute_wave": 0}
        orig_exec = rt.driver.execute
        orig_wave = rt.driver.execute_wave

        def wrap_exec(packet):
            calls["execute"] += 1
            return orig_exec(packet)

        def wrap_wave(wave):
            calls["execute_wave"] += 1
            return orig_wave(wave)

        rt.driver.execute = wrap_exec
        rt.driver.execute_wave = wrap_wave

        with pytest.raises(ValueError) as excinfo:
            rt.compile([{"op": "MapBinary", "inputs": ["a", "b"],
                         "params": {"op": 99}}])
        msg = str(excinfo.value)
        assert "Unknown MapBinary op code: 99" in msg, msg
        assert "expected 0..6" in msg, msg
        assert calls["execute"] == 0 and calls["execute_wave"] == 0, \
            "op validation must fail BEFORE any dispatch"
        EVIDENCE["R4"] = {
            "status": "PASS",
            "fact": f"op=99 -> ValueError at compile (descriptor.describe(), "
                    f"S49): {msg}; GPU never submitted: execute="
                    f"{calls['execute']}, execute_wave={calls['execute_wave']}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R5 - large N (4_200_000 / 10_000_000): chunking correct (S40 proof)
# ============================================================================

@needs_gpu
def test_r5_large_n_chunking():
    EVIDENCE["R5"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        limit = rt.driver.max_dispatch_elements()
        assert limit == LIMIT, f"driver limit {limit} != {LIMIT}"
        results = {}
        for n, expected_chunks in ((4_200_000, 2), (10_000_000, 3)):
            waves = []
            orig_wave = rt.driver.execute_wave

            def wrap_wave(wave, _orig=orig_wave, _waves=waves):
                _waves.append(wave)
                return _orig(wave)

            rt.driver.execute_wave = wrap_wave
            rng = np.random.default_rng(SEED)
            a = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            b = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            ref = _add_ref(a, b)
            rt.execute(rt.compile(MB_JOBS), {"a": a, "b": b})
            res = rt.driver.resolve_output("y")
            assert res is not None, "y should exist"
            assert res.shape == (n,), f"n={n} shape {res.shape}"
            diff = float(np.abs(res - ref).max())
            assert diff <= _tol(float(np.abs(ref).max())), \
                f"n={n} diff={diff:.3e}"
            chunks = len(waves)
            assert chunks == expected_chunks, \
                f"n={n}: chunks={chunks} expected {expected_chunks}"
            sizes = [min(limit, n - i * limit) for i in range(chunks)]
            results[str(n)] = {
                "expected_chunks": expected_chunks,
                "chunks": chunks,
                "chunk_sizes": sizes,
                "max_abs_diff_vs_oracle": _num(diff),
            }
        # CPU reference cross-check on n=4_200_000 (parity vs CPU path)
        rt_cpu = Runtime(driver=CpuDriver())
        register_all(rt_cpu)
        try:
            rng = np.random.default_rng(SEED)
            a = (rng.standard_normal(4_200_000) * 5 + 30).astype(np.float32)
            b = (rng.standard_normal(4_200_000) * 5 + 30).astype(np.float32)
            rt_cpu.execute(rt_cpu.compile(MB_JOBS), {"a": a, "b": b})
            res_cpu = rt_cpu.driver.resolve_output("y")
            ref = _add_ref(a, b)
            diff_cpu = float(np.abs(res_cpu - ref).max())
            assert diff_cpu <= _tol(float(np.abs(ref).max())), \
                f"cpu n=4_200_000 diff={diff_cpu:.3e}"
            results["4200000"]["cpu_max_abs_diff_vs_oracle"] = _num(diff_cpu)
        finally:
            rt_cpu.driver.release()
        EVIDENCE["R5"] = {
            "status": "PASS",
            "fact": f"chunking proven: n=4_200_000 -> {results['4200000']['chunks']} "
                    f"chunks {results['4200000']['chunk_sizes']} "
                    f"(GPU diff={results['4200000']['max_abs_diff_vs_oracle']:.3e}, "
                    f"CPU diff={results['4200000']['cpu_max_abs_diff_vs_oracle']:.3e}); "
                    f"n=10_000_000 -> {results['10000000']['chunks']} chunks "
                    f"{results['10000000']['chunk_sizes']} "
                    f"(diff={results['10000000']['max_abs_diff_vs_oracle']:.3e}); "
                    f"limit={limit}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R6 - empty input N=0: CPU empty output, GPU documented behavior
# ============================================================================

def test_r6_empty_input():
    fact = ""
    n = 0
    x = np.zeros(n, dtype=np.float32)

    # CPU path (always available): empty output expected
    rt_cpu = Runtime(driver=CpuDriver())
    register_all(rt_cpu)
    try:
        rt_cpu.execute(rt_cpu.compile(MB_JOBS), {"a": x, "b": x})
        res = rt_cpu.driver.resolve_output("y")
        assert res is not None and res.shape == (0,), \
            f"CPU N=0 must give empty output, got {None if res is None else res.shape}"
        fact = f"CPU N=0: empty output shape {res.shape}, no error"
    finally:
        rt_cpu.driver.release()

    # GPU path: record actual behavior (dispatch 0, documented)
    if GPU_AVAILABLE:
        rt = _make_gpu_runtime()
        try:
            try:
                rt.execute(rt.compile(MB_JOBS), {"a": x, "b": x})
                res = rt.driver.resolve_output("y")
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
# R7 - repeated calls after error work (pool not broken)
# ============================================================================

@needs_gpu
def test_r7_repeated_calls_after_error():
    EVIDENCE["R7"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        x = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        y = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        ref = _add_ref(x, y)

        def run_ok():
            rt.execute(rt.compile(MB_JOBS), {"a": x, "b": y})
            res = rt.driver.resolve_output("y")
            assert res is not None
            assert float(np.abs(res - ref).max()) <= _tol(
                float(np.abs(ref).max()))

        # 1) unknown op -> ValueError at compile, then correct run
        with pytest.raises(ValueError):
            rt.compile([{"op": "MapBinary", "inputs": ["a", "b"],
                         "params": {"op": 99}}])
        run_ok()

        # 2) empty input N=0, then correct run
        try:
            rt.execute(rt.compile(MB_JOBS),
                       {"a": np.zeros(0, np.float32),
                        "b": np.zeros(0, np.float32)})
        except Exception:  # noqa: BLE001 - documented behavior, pool survives
            pass
        run_ok()

        # 3) a-len-1 scenario (broadcast of 'a' not supported -> out [1],
        #    no error on the ordinary path), then correct run
        rt.execute(rt.compile([{"op": "MapBinary", "inputs": ["a", "b"],
                                "params": {"op": 0}, "out": "y1"}]),
                   {"a": np.array([2.0], np.float32), "b": y})
        res1 = rt.driver.resolve_output("y1")
        assert res1 is not None and res1.shape == (1,), \
            f"a-len-1 out shape {res1.shape}"
        run_ok()

        # 4) unregistered op -> KeyError, then correct run
        with pytest.raises(KeyError):
            rt.compile([{"op": "NotRegistered", "inputs": ["a"], "params": {}}])
        run_ok()

        EVIDENCE["R7"] = {
            "status": "PASS",
            "fact": "after ValueError (unknown op, at compile), after "
                    "empty-input N=0, after a-len-1 broadcast-a scenario "
                    "(out [1], no error), after KeyError (unregistered op) - "
                    "subsequent MapBinary runs produce correct results "
                    "(pool not broken)",
        }
    finally:
        rt.driver.release()


# ============================================================================
# readback-size: MapBinary n=64 / n=1M correctness (D2H by exact size)
# ============================================================================

@needs_gpu
def test_readback_size():
    EVIDENCE["readback_size"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        results = {}
        for n in (64, 1_000_000):
            a = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            b = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            rt.execute(rt.compile(MB_JOBS), {"a": a, "b": b})
            res = rt.driver.resolve_output("y")
            assert res is not None and res.shape == (n,), \
                f"n={n} shape {res.shape}"
            ref = _add_ref(a, b)
            diff = float(np.abs(res - ref).max())
            assert diff <= _tol(float(np.abs(ref).max())), \
                f"n={n} diff={diff:.3e}"
            results[str(n)] = {"shape": res.shape, "diff": _num(diff)}
        EVIDENCE["readback_size"] = {
            "status": "PASS",
            "fact": f"MapBinary ordinary path correctness: n64 shape="
                    f"{results['64']['shape']} diff={results['64']['diff']:.3e}, "
                    f"n1M shape={results['1000000']['shape']} "
                    f"diff={results['1000000']['diff']:.3e}; D2H readback by "
                    f"exact payload size (Q1 fix, S28)",
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
        "slice": SLICE,
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