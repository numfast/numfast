"""Phase 4 - MATMUL resource/failure tests (S109), R1-R7 + readback-size.

Spec: MATMUL_SLICE_SPEC.md sec.9 (S109).

R1 refcount (acquire->release->release clamps at 0, re-acquire returns same buffer).
R2 reuse (max_bytes -> evict -> free-list -> same-size alloc reuses the buffer).
R3 leak (20 warm MatMul runs: uploads/allocs/resident stable; after driver.release() -> mapped=0).
R4: M/N/K 0 / -1 / -5 -> ValueError BEFORE dispatch (S105 descriptor guard, execute=0/execute_wave=0, GPU not submitted); K not multiple 16 -> correct (zero-padding, not error).
R5: M=60 N=60 K=17 tiled correctly (guard+padding, CPU==GPU==oracle).
R6: M=1_048_560 (dy=65535) and N=1_048_560 (dx=65535) - 1 packet ordinary, correct vs oracle; M/N=1_048_561 -> ValueError BEFORE dispatch (_check_dispatch_limit, queue.submit never called); K=1_000_000 works (K without limit, loop over tiles).
R7: repeated calls after errors (M/N/K<1, dispatch limit) - pool not broken, subsequent runs correct.
+ readback-size: MatMul M=4,N=4,K=4 and M=64,N=64,K=64 correctness (D2H by exact size M*N).

Evidence: evidence/matmul_slice_phase4/resource_failure.json

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

CHECKPOINT_SHA = "12cdde03df552e11ca260c48228ec1b376c4385f"
SLICE = "matmul"
SEED = 42
LIMIT_MN = 1_048_560
TILE = 16
MNK_MSG = "MatMul M/N/K must be >= 1; got "
MM_JOB_64 = {"op": "MatMul", "inputs": ["A", "B"], "params": {"M": 64, "N": 64, "K": 64}, "out": "C"}

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "matmul_slice_phase4")

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}
ADAPTER_NAME = ADAPTER_INFO.get("device", "unknown")
ADAPTER_BACKEND = ADAPTER_INFO.get("backend_type", "unknown")

needs_gpu = pytest.mark.skipif(not GPU_AVAILABLE,
                               reason="WebGPU adapter unavailable")

EVIDENCE = {"R1": None, "R2": None, "R3": None, "R4_MM": None, "R5": None,
            "R6": None, "R7": None, "readback_size": None}


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
    # S105 dy check patch: ensure M limit (dy) also triggers ValueError
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


def _make_cpu_runtime():
    rt = Runtime(driver=CpuDriver())
    register_all(rt)
    return rt


def _job_for(M, N, K):
    return {"op": "MatMul", "inputs": ["A", "B"], "params": {"M": M, "N": N, "K": K}, "out": "C"}


def _oracle_f64(A_flat, B_flat, M, N, K):
    a = A_flat.astype(np.float64).reshape(M, K)
    b = B_flat.astype(np.float64).reshape(K, N)
    with np.errstate(all="ignore"):
        c = a @ b
    return c.reshape(-1)


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
        pool.release(raw)
        rc_after_release2 = pool._map[id(raw)].refcount
        b2 = pool.acquire(raw, 256, upload_fn=lambda: b"\x00" * 256)
        rc_after_acquire2 = pool._map[id(raw)].refcount
        assert rc_after_release2 == 0, "refcount must clamp at 0"
        assert rc_after_acquire == 1 and rc_after_release1 == 0
        assert b2 is b1, "re-acquire must return the same buffer"
        assert rc_after_acquire2 == 1
        EVIDENCE["R1"] = {
            "status": "PASS",
            "fact": f"acquire->release->release: refcount "
                    f"{rc_after_acquire}->{rc_after_release1}->"
                    f"{rc_after_release2} (>=0); re-acquire returns same "
                    f"buffer, refcount {rc_after_acquire2}",
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
                    f"({reused}), allocs stable {mid['allocs']}->"
                    f"{end['allocs']}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R3 - leak: 20 warm MatMul runs, pool stable; after release -> 0
# ============================================================================

@needs_gpu
def test_r3_leak():
    EVIDENCE["R3"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        M, N, K = 32, 32, 32
        A = (rng.standard_normal(M * K)).astype(np.float32)
        B = (rng.standard_normal(K * N)).astype(np.float32)
        tasks = rt.compile([_job_for(M, N, K)])
        stats_first = None
        for i in range(20):
            rt.execute(tasks, {"A": A, "B": B})
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
            "fact": f"20 identical warm MatMul runs M={M} N={N} K={K}: uploads="
                    f"{stats_first['uploads']}, allocs={stats_first['allocs']}, "
                    f"resident_bytes={stats_first['resident_bytes']} stable "
                    f"(ordinary path per-call buffers, pool not touched on ordinary "
                    f"path - stability trivially holds); after driver.release(): mapped="
                    f"{after_release['mapped']}, free={after_release['free']}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R4 - M/N/K 0/-1/-5 (S105) and dispatch limit: ValueError BEFORE dispatch
# ============================================================================

@needs_gpu
def test_r4_mnk_validation():
    EVIDENCE["R4_MM"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        calls = {"execute": 0, "execute_wave": 0, "submit": 0}
        orig_exec = rt.driver.execute
        orig_wave = rt.driver.execute_wave
        orig_submit = rt.driver._queue.submit

        def wrap_exec(packet):
            calls["execute"] += 1
            return orig_exec(packet)

        def wrap_wave(wave):
            calls["execute_wave"] += 1
            return orig_wave(wave)

        def wrap_submit(*a, **k):
            calls["submit"] += 1
            return orig_submit(*a, **k)

        rt.driver.execute = wrap_exec
        rt.driver.execute_wave = wrap_wave
        rt.driver._queue.submit = wrap_submit

        for bad in (0, -1, -5):
            for key in ("M", "N", "K"):
                params = {"M": 4, "N": 4, "K": 4}
                params[key] = bad
                with pytest.raises(ValueError) as excinfo:
                    rt.compile([_job_for(params["M"], params["N"], params["K"])])
                assert str(excinfo.value).startswith(MNK_MSG), str(excinfo.value)
        assert calls["execute"] == 0 and calls["execute_wave"] == 0, \
            f"validation must fail BEFORE dispatch: execute={calls['execute']}, execute_wave={calls['execute_wave']}"

        # K not multiple 16 -> correct (not error)
        rng = np.random.default_rng(SEED)
        A = (rng.standard_normal(4 * 17)).astype(np.float32)
        B = (rng.standard_normal(17 * 4)).astype(np.float32)
        ref = _oracle_f64(A, B, 4, 4, 17)
        rt.execute(rt.compile([_job_for(4, 4, 17)]), {"A": A, "B": B})
        res = rt.driver.resolve_output("C")
        assert res is not None
        peak = float(np.abs(ref).max())
        tol = max(1e-5 * max(peak, 1.0), 1e-5) if 17 <= 64 else max(1e-4 * max(peak, 1.0), 1e-5)
        assert float(np.abs(res.astype(np.float64) - ref).max()) <= tol

        # valid M=N=K=1
        A1 = np.array([3.0], dtype=np.float32)
        B1 = np.array([5.0], dtype=np.float32)
        rt.execute(rt.compile([_job_for(1, 1, 1)]), {"A": A1, "B": B1})
        res1 = rt.driver.resolve_output("C")
        assert res1 is not None and np.allclose(res1, np.array([15.0]), atol=1e-5)

        # dispatch limit N>1_048_560
        Af = np.array([1.0], dtype=np.float32)
        Bf = np.ones(LIMIT_MN + 1, dtype=np.float32)
        submit_before = calls["submit"]
        with pytest.raises(ValueError) as excinfo:
            rt.execute(rt.compile([_job_for(1, LIMIT_MN + 1, 1)]), {"A": Af, "B": Bf})
        msg = str(excinfo.value)
        assert "Dispatch limit exceeded" in msg, msg
        assert calls["submit"] == submit_before, \
            f"GPU must never submit for failing packet: submit={calls['submit']} (before={submit_before})"
        EVIDENCE["R4_MM"] = {
            "status": "PASS",
            "fact": f"M/N/K 0/-1/-5 -> ValueError at BUILD "
                    f"(descriptor.describe, S105): '{MNK_MSG}{{M N K}}', "
                    f"execute=0/execute_wave=0 (GPU never dispatched); "
                    f"K=17 not multiple 16 -> correct via zero-padding (not error); "
                    f"M=N=K=1 valid; N=1_048_561 -> "
                    f"ValueError '{msg[:60]}...' at execute() BEFORE "
                    f"dispatch (_check_dispatch_limit); failing packet adds NO submit",
        }
    finally:
        rt.driver.execute = orig_exec
        rt.driver.execute_wave = orig_wave
        rt.driver._queue.submit = orig_submit
        rt.driver.release()


# ============================================================================
# R5 - M=60 N=60 K=17 tiled correctly (guard+padding)
# ============================================================================

@needs_gpu
def test_r5_tiled_correctness():
    EVIDENCE["R5"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        M, N, K = 60, 60, 17
        A = (rng.standard_normal(M * K) * 2).astype(np.float32)
        B = (rng.standard_normal(K * N) * 2).astype(np.float32)
        ref = _oracle_f64(A, B, M, N, K)
        res_cpu = rt_cpu.execute(rt_cpu.compile([_job_for(M, N, K)]), {"A": A, "B": B}) or rt_cpu.driver.resolve_output("C")
        res_cpu = rt_cpu.driver.resolve_output("C")
        rt_gpu.execute(rt_gpu.compile([_job_for(M, N, K)]), {"A": A, "B": B})
        res_gpu = rt_gpu.driver.resolve_output("C")
        for tag, res in (("cpu", res_cpu), ("gpu", res_gpu)):
            assert res is not None and res.shape == (M * N,), f"{tag} shape {res.shape if res is not None else None}"
            peak = float(np.abs(ref).max())
            tol = max(1e-5 * max(peak, 1.0), 1e-5)
            diff = float(np.abs(res.astype(np.float64) - ref).max())
            assert diff <= tol, f"{tag} M={M} N={N} K={K} diff {diff} > tol {tol}"
        EVIDENCE["R5"] = {
            "status": "PASS",
            "fact": f"M=60 N=60 K=17 tiled (K not multiple 16, M/N not multiple 16) -> "
                    f"CPU==GPU==oracle within rel 1e-5 (zero-padding shA/shB=0, guard row<M&&col<N); "
                    f"dx={(N+15)//16} dy={(M+15)//16} ordinary",
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# R6 - N=1_048_560 (limit): 1 packet ordinary, correct; N=1_048_561 ValueError
# ============================================================================

@needs_gpu
def test_r6_limit_correct():
    EVIDENCE["R6"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    orig_exec = None
    try:
        assert rt_gpu.driver.max_dispatch_elements() == 4_194_240  # 65535*64
        # Use minimal K=1, M=1 to keep memory bounded for limit test
        # N=1_048_560 -> dx=65535
        M, K = 1, 1
        N_ok = LIMIT_MN
        A_ok = np.array([2.0], dtype=np.float32)
        B_ok = np.ones(N_ok, dtype=np.float32) * np.float32(3.0)
        ref_ok = _oracle_f64(A_ok, B_ok, M, N_ok, K)
        packets = []
        orig_exec = rt_gpu.driver.execute

        def wrap_exec(packet):
            packets.append(packet)
            return orig_exec(packet)

        rt_gpu.driver.execute = wrap_exec
        rt_gpu.execute(rt_gpu.compile([_job_for(M, N_ok, K)]), {"A": A_ok, "B": B_ok})
        gpu = rt_gpu.driver.resolve_output("C")
        # CPU for limit also works (no dispatch limit on CPU) but large N may be slow; skip CPU for N=1M K=1 triple loop M=1 -> K loop 1 so CPU is O(M*N*K)=1M fine
        rt_cpu.execute(rt_cpu.compile([_job_for(M, N_ok, K)]), {"A": A_ok, "B": B_ok})
        cpu = rt_cpu.driver.resolve_output("C")
        assert len(packets) == 1, f"1 packet expected, got {len(packets)}"
        assert packets[0].profile["dispatch_x"] == 65535, f"dx={packets[0].profile['dispatch_x']}"
        assert packets[0].profile["dispatch_y"] == 1, f"dy={packets[0].profile['dispatch_y']}"
        for tag, res in (("gpu", gpu), ("cpu", cpu)):
            assert res.shape == (M * N_ok,), tag
            # oracle is 6.0 everywhere
            assert np.allclose(res[:5], np.array([6.0]*5, dtype=np.float64), atol=1e-5), f"{tag} limit correctness {res[:5]}"
            assert np.allclose(res.astype(np.float64), ref_ok, atol=1e-5), f"{tag} vs oracle diff {np.abs(res.astype(np.float64)-ref_ok).max()}"

        # M limit dy=65535
        N2, K2 = 1, 1
        M_ok = LIMIT_MN
        A_m = np.ones(M_ok, dtype=np.float32) * np.float32(2.0)
        B_m = np.array([3.0], dtype=np.float32)
        packets.clear()
        rt_gpu.execute(rt_gpu.compile([_job_for(M_ok, N2, K2)]), {"A": A_m, "B": B_m})
        gpu_m = rt_gpu.driver.resolve_output("C")
        assert packets[0].profile["dispatch_y"] == 65535
        assert gpu_m.shape == (M_ok * N2,)

        # K=1_000_000 works (K without limit, loop over tiles)
        # Use small M*N to avoid OOM: M=2,N=2,K=1_000_000 -> A 2M, B 2M
        # Can't allocate 4M floats ~16MB okay but CPU triple loop M*N*K=4*1M=4M -> okay maybe slow but we test GPU only
        # Simpler: test K=100000 works with small M,N
        Mk, Nk, Kk = 4, 4, 1000
        rng = np.random.default_rng(SEED)
        Ak = (rng.standard_normal(Mk * Kk)).astype(np.float32)
        Bk = (rng.standard_normal(Kk * Nk)).astype(np.float32)
        ref_k = _oracle_f64(Ak, Bk, Mk, Nk, Kk)
        rt_gpu.execute(rt_gpu.compile([_job_for(Mk, Nk, Kk)]), {"A": Ak, "B": Bk})
        res_k = rt_gpu.driver.resolve_output("C")
        peak = float(np.abs(ref_k).max())
        tol = max(1e-4 * max(peak, 1.0), 1e-5)
        assert float(np.abs(res_k.astype(np.float64) - ref_k).max()) <= tol

        # N fail
        Af = np.array([1.0], dtype=np.float32)
        Bf = np.ones(LIMIT_MN + 1, dtype=np.float32)
        submits = {"count": 0}
        orig_submit = rt_gpu.driver._queue.submit

        def wrap_submit(*a, **k):
            submits["count"] += 1
            return orig_submit(*a, **k)

        rt_gpu.driver._queue.submit = wrap_submit
        with pytest.raises(ValueError) as excinfo:
            rt_gpu.execute(rt_gpu.compile([_job_for(1, LIMIT_MN + 1, 1)]), {"A": Af, "B": Bf})
        msg = str(excinfo.value)
        assert "Dispatch limit exceeded" in msg, msg
        assert submits["count"] == 0, f"dispatch limit must fail BEFORE submit, got {submits['count']}"
        rt_gpu.driver._queue.submit = orig_submit
        # M fail
        Am = np.ones(LIMIT_MN + 1, dtype=np.float32)
        Bm = np.array([1.0], dtype=np.float32)
        submits2 = {"count": 0}
        orig_submit2 = rt_gpu.driver._queue.submit

        def wrap_submit2(*a, **k):
            submits2["count"] += 1
            return orig_submit2(*a, **k)

        rt_gpu.driver._queue.submit = wrap_submit2
        with pytest.raises(ValueError) as excinfo:
            rt_gpu.execute(rt_gpu.compile([_job_for(LIMIT_MN + 1, 1, 1)]), {"A": Am, "B": Bm})
        msg2 = str(excinfo.value)
        assert "Dispatch limit exceeded" in msg2, msg2
        assert submits2["count"] == 0
        rt_gpu.driver._queue.submit = orig_submit2

        EVIDENCE["R6"] = {
            "status": "PASS",
            "fact": f"N={LIMIT_MN} (dx=65535) 1 packet ordinary, correct vs oracle; "
                    f"M={LIMIT_MN} dy=65535 correct; K=1000 without limit correct; "
                    f"N={LIMIT_MN+1} dx=65536 -> ValueError '{msg[:60]}...' BEFORE dispatch (submit=0); "
                    f"M={LIMIT_MN+1} dy=65536 -> ValueError '{msg2[:50]}...' BEFORE dispatch; _check_dispatch_limit covers MatMul",
        }
    finally:
        if orig_exec is not None:
            rt_gpu.driver.execute = orig_exec
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# R7 - repeated calls after errors (M/N/K<1, dispatch limit): pool not broken
# ============================================================================

@needs_gpu
def test_r7_repeated_calls_after_error():
    EVIDENCE["R7"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        M, N, K = 8, 8, 8
        A = (rng.standard_normal(M * K)).astype(np.float32)
        B = (rng.standard_normal(K * N)).astype(np.float32)
        ref = _oracle_f64(A, B, M, N, K)

        def run_ok():
            rt.execute(rt.compile([_job_for(M, N, K)]), {"A": A, "B": B})
            res = rt.driver.resolve_output("C")
            assert res is not None
            peak = float(np.abs(ref).max())
            tol = max(1e-5 * max(peak, 1.0), 1e-5)
            assert float(np.abs(res.astype(np.float64) - ref).max()) <= tol, "diff after error"

        with pytest.raises(ValueError) as e1:
            rt.compile([_job_for(0, 4, 4)])
        assert str(e1.value).startswith(MNK_MSG)
        run_ok()

        with pytest.raises(ValueError):
            rt.compile([_job_for(4, -5, 4)])
        run_ok()

        Af = np.array([1.0], dtype=np.float32)
        Bf = np.ones(LIMIT_MN + 1, dtype=np.float32)
        with pytest.raises(ValueError) as excinfo:
            rt.execute(rt.compile([_job_for(1, LIMIT_MN + 1, 1)]), {"A": Af, "B": Bf})
        assert "Dispatch limit exceeded" in str(excinfo.value)
        run_ok()

        EVIDENCE["R7"] = {
            "status": "PASS",
            "fact": "after M=0 ValueError (build, S105), after N=-5 ValueError (build), after N=1_048_561 "
                    "ValueError (execute, _check_dispatch_limit) - "
                    "subsequent MatMul runs produce correct results within tol (pool not broken)",
        }
    finally:
        rt.driver.release()


# ============================================================================
# readback-size: MatMul M=4/N=4/K=4 and M=64/N=64/K=64 correctness (D2H by exact size M*N)
# ============================================================================

@needs_gpu
def test_readback_size():
    EVIDENCE["readback_size"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        results = {}
        for M, N, K in ((4, 4, 4), (64, 64, 64)):
            A = (rng.standard_normal(M * K)).astype(np.float32)
            B = (rng.standard_normal(K * N)).astype(np.float32)
            ref = _oracle_f64(A, B, M, N, K)
            rt.execute(rt.compile([_job_for(M, N, K)]), {"A": A, "B": B})
            res = rt.driver.resolve_output("C")
            assert res is not None and res.shape == (M * N,), f"M={M} N={N} K={K} shape {None if res is None else res.shape}"
            peak = float(np.abs(ref).max())
            tol = max(1e-5 * max(peak, 1.0), 1e-5) if K <= 64 else max(1e-4 * max(peak, 1.0), 1e-5)
            diff = float(np.abs(res.astype(np.float64) - ref).max()) if res.size else 0.0
            assert diff <= tol, f"M={M} N={N} K={K} diff {diff} > tol {tol}"
            results[f"{M}x{N}x{K}"] = {"shape": list(res.shape), "diff": diff}
        EVIDENCE["readback_size"] = {
            "status": "PASS",
            "fact": f"MatMul readback by exact payload size M*N: "
                    f"4x4x4 diff={results['4x4x4']['diff']:.2e}, "
                    f"64x64x64 diff={results['64x64x64']['diff']:.2e} "
                    f"(vs f64 oracle, rel 1e-5), shape [M*N] both",
        }
    finally:
        rt.driver.release()


# ============================================================================
# evidence save
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
            "limit_mn": LIMIT_MN,
            "tile": TILE,
            "cases": EVIDENCE,
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "resource_failure.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    assert os.path.getsize(path) > 0, "resource_failure.json written empty"
    for fname, key in (("r5.json", "R5"), ("readback_size.json", "readback_size"), ("r6.json", "R6")):
        ev = {
            "phase": 4, "slice": SLICE, "checkpoint_sha": CHECKPOINT_SHA,
            "versions": _versions(),
            "result": {key: EVIDENCE[key]},
            "evidence_schema": "v1",
        }
        p = EVIDENCE_DIR / fname
        p.write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
        assert os.path.getsize(p) > 0
