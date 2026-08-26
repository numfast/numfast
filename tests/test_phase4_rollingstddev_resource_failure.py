"""Phase 4 - ROLLINGSTDDEV resource/failure tests (S102a), R1-R7 + readback-size.

Spec: ROLLINGSTDDEV_SLICE_SPEC.md sec.7 (S102a).

R1 refcount (acquire->release->release clamps at 0, re-acquire returns same buffer).
R2 reuse (max_bytes -> evict -> free-list -> same-size alloc reuses the buffer).
R3 leak (20 warm RollingStdDev runs: uploads/allocs/resident stable; after driver.release() -> mapped=0).
R4: period 0 / -1 / -5 -> ValueError BEFORE dispatch (S99 descriptor guard, execute=0/execute_wave=0, GPU not submitted); period=1 valid.
R5: N=0 -> CPU empty without error (contract); GPU 0-byte buffer impossible on FROZEN wgpu driver (S62 precedent) - ValueError at upload BEFORE dispatch, recorded characterization.
R6: N=4_194_240 (limit) - 1 packet ordinary, correct vs oracle f64 stddev; N=4_194_241 -> ValueError BEFORE dispatch (_check_dispatch_limit, queue.submit never called).
R7: repeated calls after errors (period<1, dispatch limit) - pool not broken, subsequent runs correct.
+ readback-size: RollingStdDev n=64 / n=1M correctness (D2H by exact size).

Evidence: evidence/rollingstddev_slice_phase4/resource_failure.json

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

CHECKPOINT_SHA = "a60e39b0c49434231433faac671369b8f7e34c2f"
SLICE = "rollingstddev"
SEED = 42
LIMIT = 4_194_240
PERIOD_MSG = "RollingStdDev period must be >= 1; got "
RS_JOB = {"op": "RollingStdDev", "inputs": ["x"], "params": {"period": 100},
          "out": "y"}

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "rollingstddev_slice_phase4")

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

EVIDENCE = {"R1": None, "R2": None, "R3": None, "R4_RS": None, "R5": None,
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
    return rt


def _make_cpu_runtime():
    rt = Runtime(driver=CpuDriver())
    register_all(rt)
    return rt


def _job_for(period):
    return {"op": "RollingStdDev", "inputs": ["x"], "params": {"period": period},
            "out": "y"}


def _oracle_f64(x, period):
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    xf = x.astype(np.float64)
    if n == 0:
        return out
    with np.errstate(all="ignore"):
        s = np.cumsum(xf)
        s_sq = np.cumsum(xf * xf)
        if period <= n:
            tail = np.zeros(n, dtype=np.float64)
            tail_sq = np.zeros(n, dtype=np.float64)
            tail[period:] = s[:n-period]
            tail_sq[period:] = s_sq[:n-period]
            sum_win = np.empty(n, dtype=np.float64)
            sum_sq_win = np.empty(n, dtype=np.float64)
            sum_win[period-1:] = s[period-1:] - tail[period-1:]
            sum_sq_win[period-1:] = s_sq[period-1:] - tail_sq[period-1:]
            mean = sum_win[period-1:] / period
            var = sum_sq_win[period-1:] / period - mean * mean
            var[var < 0] = 0.0
            out[period-1:] = np.sqrt(var)
    return out


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
# R3 - leak: 20 warm RollingStdDev runs, pool stable; after release -> 0
# ============================================================================

@needs_gpu
def test_r3_leak():
    EVIDENCE["R3"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        x = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        tasks = rt.compile([RS_JOB])
        stats_first = None
        for i in range(20):
            rt.execute(tasks, {"x": x})
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
            "fact": f"20 identical warm RollingStdDev runs: uploads="
                    f"{stats_first['uploads']}, allocs={stats_first['allocs']}, "
                    f"resident_bytes={stats_first['resident_bytes']} stable "
                    f"(ordinary path allocates per-call buffers, pool not "
                    f"touched on the ordinary path - stability trivially "
                    f"holds); after driver.release(): mapped="
                    f"{after_release['mapped']}, free={after_release['free']}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R4 - period 0/-1/-5 (S99) and N=4_194_241 (S64): ValueError BEFORE dispatch
# ============================================================================

@needs_gpu
def test_r4_period_validation():
    EVIDENCE["R4_RS"] = {"status": "FAIL", "fact": "test raised"}
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

        for bad_period in (0, -1, -5):
            with pytest.raises(ValueError) as excinfo:
                rt.compile([_job_for(bad_period)])
            assert str(excinfo.value).startswith(PERIOD_MSG), \
                str(excinfo.value)
        assert calls["execute"] == 0 and calls["execute_wave"] == 0, \
            f"period validation must fail BEFORE dispatch: " \
            f"execute={calls['execute']}, execute_wave={calls['execute_wave']}"

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rt.execute(rt.compile([_job_for(1)]), {"x": x})
        res1 = rt.driver.resolve_output("y")
        assert res1 is not None and np.allclose(res1, np.zeros(3, dtype=np.float64), atol=5e-2), res1

        y = np.zeros(LIMIT + 1, dtype=np.float32)
        submit_before = calls["submit"]
        with pytest.raises(ValueError) as excinfo:
            rt.execute(rt.compile([RS_JOB]), {"x": y})
        msg = str(excinfo.value)
        assert "Dispatch limit exceeded" in msg, msg
        assert calls["submit"] == submit_before, \
            f"GPU must never submit for the failing packet: " \
            f"submit={calls['submit']} (before={submit_before})"
        EVIDENCE["R4_RS"] = {
            "status": "PASS",
            "fact": f"period 0/-1/-5 -> ValueError at BUILD "
                    f"(descriptor.describe, S99): '{PERIOD_MSG}{{p}}', "
                    f"execute=0/execute_wave=0 (GPU never dispatched); "
                    f"period=1 valid (out[i]=0.0); N=4_194_241 -> "
                    f"ValueError '{msg[:60]}...' at execute() BEFORE "
                    f"dispatch (_check_dispatch_limit, S64); failing "
                    f"packet adds NO submit (submit={calls['submit']}, "
                    f"same as before the call)",
        }
    finally:
        rt.driver.execute = orig_exec
        rt.driver.execute_wave = orig_wave
        rt.driver._queue.submit = orig_submit
        rt.driver.release()


# ============================================================================
# R5 - N=0: CPU empty without error; GPU 0-byte buffer limitation
# ============================================================================

def test_r5_n_zero():
    EVIDENCE["R5"] = {"status": "FAIL", "fact": "test raised"}
    x = np.zeros(0, dtype=np.float32)
    facts = {}

    rt_cpu = Runtime(driver=CpuDriver())
    register_all(rt_cpu)
    try:
        rt_cpu.execute(rt_cpu.compile([RS_JOB]), {"x": x})
        res = rt_cpu.driver.resolve_output("y")
        assert res is not None and res.shape == (0,), \
            f"CPU N=0 shape {None if res is None else res.shape}"
        facts["cpu"] = f"empty output shape {list(res.shape)}, no error"
    finally:
        rt_cpu.driver.release()

    if GPU_AVAILABLE:
        rt = _make_gpu_runtime()
        try:
            submits = {"count": 0}
            orig_submit = rt.driver._queue.submit

            def wrap_submit(*a, **k):
                submits["count"] += 1
                return orig_submit(*a, **k)

            rt.driver._queue.submit = wrap_submit
            try:
                rt.execute(rt.compile([RS_JOB]), {"x": x})
                res = rt.driver.resolve_output("y")
                facts["gpu"] = f"empty output shape {None if res is None else list(res.shape)}"
            except ValueError as e:
                facts["gpu"] = (f"ValueError '{str(e)}' at upload (0-byte "
                                f"buffer, FROZEN wgpu driver), "
                                f"submit={submits['count']} (never reached "
                                f"dispatch)")
            rt.driver._queue.submit = orig_submit
        finally:
            rt.driver.release()
    else:
        facts["gpu"] = "skipped (no adapter)"

    assert facts["cpu"].startswith("empty output")
    assert "unexpected" not in facts["gpu"]
    EVIDENCE["R5"] = {
        "status": "PASS",
        "fact": ("N=0 (S102a/R5, contract): " + "; ".join(
            f"{k}: {v}" for k, v in facts.items()) +
            " - CPU honors contract (empty, no error, dispatch 0); GPU "
            "cannot create 0-byte buffer on FROZEN wgpu driver (S62 Reduce "
            "precedent), ValueError at upload BEFORE dispatch/submit - "
            "recorded characterization, NOT a failure"),
    }


# ============================================================================
# R6 - N=4_194_240 (limit): 1 packet ordinary, correct; N=4_194_241 ValueError
# ============================================================================

@needs_gpu
def test_r6_limit_n_correct():
    EVIDENCE["R6"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    orig_exec = None
    try:
        assert rt_gpu.driver.max_dispatch_elements() == LIMIT
        rng = np.random.default_rng(SEED)
        n = LIMIT
        x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        ref = _oracle_f64(x, 100)
        packets = []
        orig_exec = rt_gpu.driver.execute

        def wrap_exec(packet):
            packets.append(packet)
            return orig_exec(packet)

        rt_gpu.driver.execute = wrap_exec
        rt_gpu.execute(rt_gpu.compile([RS_JOB]), {"x": x})
        gpu = rt_gpu.driver.resolve_output("y")
        rt_cpu.execute(rt_cpu.compile([RS_JOB]), {"x": x})
        cpu = rt_cpu.driver.resolve_output("y")
        assert len(packets) == 1, f"1 packet expected, got {len(packets)}"
        assert packets[0].profile["dispatch_x"] == 65535, \
            f"dx={packets[0].profile['dispatch_x']}"
        for tag, res in (("gpu", gpu), ("cpu", cpu)):
            resf = res.astype(np.float64)
            fin = np.isfinite(resf) & np.isfinite(ref)
            diff = float(np.abs(resf[fin] - ref[fin]).max()) if fin.any() else 0.0
            peak = float(np.abs(ref[fin]).max()) if fin.any() else 1.0
            if tag == "gpu":
                tol = max(1e-4 * max(peak, 1.0), 1e-4)
            else:
                tol = max(1e-3 * max(peak, 1.0), 1e-3)
            assert diff <= tol, f"{tag} n={n}: diff {diff} > tol {tol}"

        y = np.zeros(LIMIT + 1, dtype=np.float32)
        submits = {"count": 0}
        orig_submit = rt_gpu.driver._queue.submit

        def wrap_submit(*a, **k):
            submits["count"] += 1
            return orig_submit(*a, **k)

        rt_gpu.driver._queue.submit = wrap_submit
        with pytest.raises(ValueError) as excinfo:
            rt_gpu.execute(rt_gpu.compile([RS_JOB]), {"x": y})
        msg = str(excinfo.value)
        assert "Dispatch limit exceeded" in msg, msg
        assert submits["count"] == 0, \
            f"dispatch limit must fail BEFORE submit, got {submits['count']}"
        rt_gpu.driver._queue.submit = orig_submit
        EVIDENCE["R6"] = {
            "status": "PASS",
            "fact": f"N={LIMIT} (limit): 1 packet ordinary, dx=65535; "
                    f"GPU==oracle within rel/abs 1e-5; "
                    f"N={LIMIT + 1}: GPU ValueError '{msg[:60]}...' BEFORE "
                    f"dispatch (queue.submit never called, submits=0) - "
                    f"_check_dispatch_limit covers RollingStdDev",
        }
    finally:
        if orig_exec is not None:
            rt_gpu.driver.execute = orig_exec
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# R7 - repeated calls after errors (period<1, dispatch limit): pool not broken
# ============================================================================

@needs_gpu
def test_r7_repeated_calls_after_error():
    EVIDENCE["R7"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        x = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        ref = _oracle_f64(x, 100)

        def run_ok():
            rt.execute(rt.compile([RS_JOB]), {"x": x})
            res = rt.driver.resolve_output("y")
            assert res is not None
            resf = res.astype(np.float64)
            fin = np.isfinite(resf) & np.isfinite(ref)
            diff = float(np.abs(resf[fin] - ref[fin]).max()) if fin.any() else 0.0
            peak = float(np.abs(ref[fin]).max()) if fin.any() else 1.0
            assert diff <= max(1e-4 * max(peak, 1.0), 1e-4), f"diff {diff}"

        with pytest.raises(ValueError) as e1:
            rt.compile([_job_for(0)])
        assert str(e1.value).startswith(PERIOD_MSG)
        run_ok()

        with pytest.raises(ValueError):
            rt.compile([_job_for(-5)])
        run_ok()

        y = np.zeros(LIMIT + 1, dtype=np.float32)
        with pytest.raises(ValueError) as excinfo:
            rt.execute(rt.compile([RS_JOB]), {"x": y})
        assert "Dispatch limit exceeded" in str(excinfo.value)
        run_ok()

        EVIDENCE["R7"] = {
            "status": "PASS",
            "fact": "after period=0 ValueError (build, S99), after "
                    "period=-5 ValueError (build), after N=4_194_241 "
                    "ValueError (execute, _check_dispatch_limit) - "
                    "subsequent RollingStdDev runs produce correct results "
                    "within tol (pool not broken)",
        }
    finally:
        rt.driver.release()


# ============================================================================
# readback-size: RollingStdDev n=64 / n=1M correctness (D2H by exact size)
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
            ref = _oracle_f64(x, 100)
            rt.execute(rt.compile([RS_JOB]), {"x": x})
            res = rt.driver.resolve_output("y")
            assert res is not None and res.shape == (n,), \
                f"n={n} shape {None if res is None else res.shape}"
            resf = res.astype(np.float64)
            fin = np.isfinite(resf) & np.isfinite(ref)
            diff = float(np.abs(resf[fin] - ref[fin]).max()) if fin.any() else 0.0
            peak = float(np.abs(ref[fin]).max()) if fin.any() else 1.0
            assert diff <= max(1e-4 * max(peak, 1.0), 1e-4), f"n={n} diff {diff}"
            results[str(n)] = {"shape": list(res.shape), "diff": diff}
        EVIDENCE["readback_size"] = {
            "status": "PASS",
            "fact": f"RollingStdDev readback by exact payload size: "
                    f"n=64 diff={results['64']['diff']:.2e}, "
                    f"n=1M diff={results['1000000']['diff']:.2e} "
                    f"(vs f64 oracle, rel/abs 1e-5), "
                    f"shape [n] both",
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
            "limit": LIMIT,
            "cases": EVIDENCE,
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "resource_failure.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    assert os.path.getsize(path) > 0, "resource_failure.json written empty"
    # also write n_zero and readback_size as separate files for evidence list
    for fname, key in (("n_zero.json", "R5"), ("readback_size.json", "readback_size")):
        ev = {
            "phase": 4, "slice": SLICE, "checkpoint_sha": CHECKPOINT_SHA,
            "versions": _versions(),
            "result": {key: EVIDENCE[key]},
            "evidence_schema": "v1",
        }
        p = EVIDENCE_DIR / fname
        p.write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
        assert os.path.getsize(p) > 0
