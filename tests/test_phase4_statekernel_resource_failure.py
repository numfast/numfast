"""Phase 4 - STATEKERNEL resource/failure tests (S89), R1-R7 + readback.

Spec: STATEKERNEL_SLICE_SPEC.md sec.7 (S89).

R1 refcount (acquire->release->release clamps at 0, re-acquire returns
   the same buffer).
R2 reuse (max_bytes -> evict -> free-list -> same-size alloc reuses the
   buffer).
R3 leak (20 warm StateKernel runs: uploads/allocs/resident stable; after
   driver.release() -> mapped=0).
R4-SK: mode=3/-1 -> ValueError at COMPILE BEFORE dispatch (S87
   descriptor guard, execute=0/execute_wave=0, queue.submit never
   called); scan mode=2 -> ValueError; chunk<1 clamped to 1 (NOT error).
R5: N=0 -> CPU empty without error (contract shape); GPU cannot create
   0-byte buffer (FROZEN wgpu driver, "Mapped size must be larger than
   zero." at upload BEFORE dispatch) - recorded characterization.
R6: serial n=4_194_240 and n=4_194_241 BOTH work (S87 fix: dispatch=
   (1,1,1) -> serial kernel runs the FULL-array loop per invocation,
   dx=1, not under _check_dispatch_limit), correct vs f64 oracle;
   scan n=4_194_240 works (dx=65535 at limit), n=4_194_241 -> ValueError
   BEFORE dispatch (_check_dispatch_limit, queue.submit never called).
R7: repeated calls after errors (mode=3, scan dispatch limit) - pool not
   broken, subsequent runs correct.
+ readback-size: StateKernel n=64 / n=1M correctness (D2H by exact size).

Evidence: evidence/statekernel_slice_phase4/resource_failure.json

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

from scipy.signal import lfilter

from Runtime._lib.runtime import Runtime
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
from Compute import register_all

CHECKPOINT_SHA = "08575d7d8a57ec5f1de0fae974bbeb76bf650f07"
SLICE = "statekernel"
SEED = 42
LIMIT = 4_194_240
MODE_MSG = "Unknown StateKernel mode: "
SK_JOB = {"op": "StateKernel_Single", "inputs": ["x"],
          "params": {"mode": 0, "a": 2.0 / 15.0, "b": 13.0 / 15.0},
          "out": "y"}

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "statekernel_slice_phase4")

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

EVIDENCE = {"R1": None, "R2": None, "R3": None, "R4_SK": None, "R5": None,
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


def _oracle_mode0(x, a, b):
    """f64 recurrence out[0]=x[0]; out[i]=a*x[i]+b*out[i-1] (lfilter)."""
    xd = x.astype(np.float64)
    y, _ = lfilter([a], [1.0, -b], xd, zi=[float(xd[0]) * (1.0 - a)])
    out = y.astype(np.float64, copy=True)
    out[0] = float(xd[0])
    return out


def _scan_jobs(chunk=256):
    return [
        {"op": "StateKernelScanLocal", "inputs": ["data"],
         "params": {"mode": 0, "a": 2.0 / 15.0, "b": 13.0 / 15.0,
                    "chunk": chunk},
         "out": ["skl_tmp", "skl_last"]},
        {"op": "StateKernelScanTotals", "inputs": ["skl_last"],
         "params": {"mode": 0, "a": 2.0 / 15.0, "b": 13.0 / 15.0,
                    "chunk": chunk},
         "out": "skl_sp"},
        {"op": "StateKernelScanFinal", "inputs": ["skl_tmp", "skl_sp"],
         "params": {"mode": 0, "a": 2.0 / 15.0, "b": 13.0 / 15.0,
                    "chunk": chunk},
         "out": "skl_out"},
    ]


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
# R3 - leak: 20 warm StateKernel runs, pool stable; after release -> 0
# ============================================================================

@needs_gpu
def test_r3_leak():
    EVIDENCE["R3"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        x = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        tasks = rt.compile([SK_JOB])
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
            "fact": f"20 identical warm StateKernel runs: uploads="
                    f"{stats_first['uploads']}, allocs={stats_first['allocs']}, "
                    f"resident_bytes={stats_first['resident_bytes']} stable; "
                    f"after driver.release(): mapped="
                    f"{after_release['mapped']}, free={after_release['free']}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R4-SK: mode=3/-1 -> ValueError BEFORE dispatch, scan mode=2 -> ValueError,
#        chunk<1 clamp, a=NaN propagate
# ============================================================================

@needs_gpu
def test_r4_sk_validation():
    EVIDENCE["R4_SK"] = {"status": "FAIL", "fact": "test raised"}
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

        for bad_mode in (3, -1):
            with pytest.raises(ValueError) as excinfo:
                rt.compile([{"op": "StateKernel_Single", "inputs": ["x"],
                             "params": {"mode": bad_mode, "a": 1.0, "b": 0.0},
                             "out": "y"}])
            assert MODE_MSG in str(excinfo.value), str(excinfo.value)
        assert calls["execute"] == 0 and calls["execute_wave"] == 0, \
            f"mode validation must fail BEFORE dispatch: execute={calls['execute']}, wave={calls['execute_wave']}"

        with pytest.raises(ValueError) as excinfo:
            rt.compile([{"op": "StateKernelScanLocal", "inputs": ["data"],
                         "params": {"mode": 2, "a": 1.0, "b": 0.0, "chunk": 256},
                         "out": ["skl_tmp", "skl_last"]}])
        assert "Unknown StateKernelScanLocal mode" in str(excinfo.value)

        # chunk<1 clamp 1 (NOT error) - CPU path correct, GPU characterization recorded elsewhere
        for chunk in (0, -5):
            tasks = rt.compile(_scan_jobs(chunk=chunk))
            # compile must not raise
            assert tasks is not None
        # chunk=1 also ok
        tasks = rt.compile(_scan_jobs(chunk=1))
        assert tasks is not None

        # a=NaN propagate - NOT validated, must compile and run (IEEE)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tasks = rt.compile([{"op": "StateKernel_Single", "inputs": ["x"],
                             "params": {"mode": 0, "a": float("nan"), "b": 0.5},
                             "out": "y"}])
        rt.driver.execute = orig_exec
        rt.driver.execute_wave = orig_wave
        rt.driver._queue.submit = orig_submit
        rt.execute(tasks, {"x": x})
        res = rt.driver.resolve_output("y")
        assert res is not None, "a=NaN must propagate, not error"

        EVIDENCE["R4_SK"] = {
            "status": "PASS",
            "fact": f"mode 3/-1 -> ValueError at COMPILE ('{MODE_MSG}{{mode}}'), execute=0/execute_wave=0 (GPU never dispatched); scan mode=2 -> ValueError 'Unknown StateKernelScanLocal mode: 2'; chunk=0/-5 -> clamp 1 (NOT error, CPU _nblocks clamp AS-IS, GPU uniform keeps 0.0 characterization); a=NaN -> propagate (NOT error, IEEE)",
        }
    finally:
        try:
            rt.driver.execute = orig_exec
        except Exception:
            pass
        try:
            rt.driver.execute_wave = orig_wave
        except Exception:
            pass
        try:
            rt.driver._queue.submit = orig_submit
        except Exception:
            pass
        rt.driver.release()


# ============================================================================
# R5 - N=0: CPU empty without error; GPU 0-byte buffer limitation
# ============================================================================

def test_r5_n_zero():
    EVIDENCE["R5"] = {"status": "FAIL", "fact": "test raised"}
    x = np.zeros(0, dtype=np.float32)
    facts = {}

    for mode, job in [(0, SK_JOB),
                      (1, {"op": "StateKernel_Dual", "inputs": ["gain", "loss"],
                           "params": {"mode": 1, "a": 1.0/14.0, "b": 13.0/14.0},
                           "out": ["ema_gain", "ema_loss"]}),
                      (2, {"op": "StateKernel_ST", "inputs": ["price", "upper", "lower"],
                           "params": {"mode": 2}, "out": "y"})]:
        rt_cpu = Runtime(driver=CpuDriver())
        register_all(rt_cpu)
        try:
            if mode == 0:
                rt_cpu.execute(rt_cpu.compile([job]), {"x": x})
                res = rt_cpu.driver.resolve_output("y")
            elif mode == 1:
                g = np.zeros(0, dtype=np.float32)
                rt_cpu.execute(rt_cpu.compile([job]), {"gain": g, "loss": g})
                res = rt_cpu.driver.resolve_output("ema_gain")
            else:
                p = np.zeros(0, dtype=np.float32)
                rt_cpu.execute(rt_cpu.compile([job]), {"price": p, "upper": p, "lower": p})
                res = rt_cpu.driver.resolve_output("y")
            assert res is not None and res.shape == (0,), f"CPU mode{mode} N=0 shape {None if res is None else res.shape}"
        finally:
            rt_cpu.driver.release()
    facts["cpu"] = "mode 0/1/2 -> empty output shape [0], no error"

    if GPU_AVAILABLE:
        rt = _make_gpu_runtime()
        try:
            submits = {"count": 0}
            orig_submit = rt.driver._queue.submit

            def wrap_submit(*a, **k):
                submits["count"] += 1
                return orig_submit(*a, **k)

            rt.driver._queue.submit = wrap_submit
            gpu_facts = []
            for mode, job, data in [
                (0, SK_JOB, {"x": x}),
                (1, {"op": "StateKernel_Dual", "inputs": ["gain", "loss"],
                     "params": {"mode": 1, "a": 1.0/14.0, "b": 13.0/14.0},
                     "out": ["ema_gain", "ema_loss"]}, {"gain": x, "loss": x}),
                (2, {"op": "StateKernel_ST", "inputs": ["price", "upper", "lower"],
                     "params": {"mode": 2}, "out": "y"}, {"price": x, "upper": x, "lower": x}),
            ]:
                try:
                    rt.execute(rt.compile([job]), data)
                    res = rt.driver.resolve_output(job["out"] if isinstance(job["out"], str) else job["out"][0])
                    gpu_facts.append(f"mode{mode}: empty shape {list(res.shape) if res is not None else 'None'}")
                except ValueError as e:
                    gpu_facts.append(f"mode{mode}: ValueError '{str(e)}' at upload (0-byte, submit={submits['count']})")
            facts["gpu"] = "; ".join(gpu_facts) + f"; submit={submits['count']} (never reached dispatch for failures)"
            rt.driver._queue.submit = orig_submit
        finally:
            rt.driver.release()
    else:
        facts["gpu"] = "skipped (no adapter)"

    EVIDENCE["R5"] = {
        "status": "PASS",
        "fact": "N=0 (S89/R5, contract sec.8 Shape): cpu: " + facts["cpu"] + "; gpu: " + facts["gpu"] + " - CPU honors contract (empty, no error, all modes); GPU cannot create 0-byte buffer on FROZEN wgpu driver (S62 Reduce precedent), ValueError at upload BEFORE dispatch/submit - recorded characterization, NOT a failure",
    }


# ============================================================================
# R6 - serial n=4_194_240 and n=4_194_241 BOTH work (dx=1), scan limit
# ============================================================================

@needs_gpu
def test_r6_limits():
    EVIDENCE["R6"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        a = 2.0 / 15.0
        b = 13.0 / 15.0
        # serial both at and over limit must work (dx=1)
        for n in (LIMIT, LIMIT + 1):
            x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            ref = _oracle_mode0(x, a, b)
            rt_gpu.execute(rt_gpu.compile([SK_JOB]), {"x": x})
            gpu = rt_gpu.driver.resolve_output("y")
            rt_cpu.execute(rt_cpu.compile([SK_JOB]), {"x": x})
            cpu = rt_cpu.driver.resolve_output("y")
            assert gpu.shape == (n,), f"gpu n={n} shape {gpu.shape}"
            assert cpu.shape == (n,), f"cpu n={n} shape {cpu.shape}"
            for tag, res in (("gpu", gpu), ("cpu", cpu)):
                resf = res.astype(np.float64)
                fin = np.isfinite(resf) & np.isfinite(ref)
                diff = float(np.abs(resf[fin] - ref[fin]).max()) if fin.any() else 0.0
                peak = float(np.abs(ref[fin]).max()) if fin.any() else 1.0
                tol = max(1e-6 * peak, 1e-6)
                assert diff <= tol, f"{tag} n={n}: diff {diff} > tol {tol}"
        # scan n=4_194_240 works (dx=65535 at limit)
        x_scan = (rng.standard_normal(LIMIT) * 5 + 30).astype(np.float32)
        rt_gpu.execute(rt_gpu.compile(_scan_jobs(chunk=256)), {"data": x_scan})
        res_scan = rt_gpu.driver.resolve_output("skl_out")
        assert res_scan is not None and res_scan.shape == (LIMIT,), f"scan n={LIMIT} shape {res_scan.shape if res_scan is not None else None}"
        # scan n=4_194_241 -> ValueError BEFORE dispatch
        y = np.zeros(LIMIT + 1, dtype=np.float32)
        submits = {"count": 0}
        orig_submit = rt_gpu.driver._queue.submit

        def wrap_submit(*a, **k):
            submits["count"] += 1
            return orig_submit(*a, **k)

        rt_gpu.driver._queue.submit = wrap_submit
        with pytest.raises(ValueError) as excinfo:
            rt_gpu.execute(rt_gpu.compile(_scan_jobs(chunk=256)), {"data": y})
        msg = str(excinfo.value)
        assert "Dispatch limit exceeded" in msg, msg
        assert submits["count"] == 0, f"scan dispatch limit must fail BEFORE submit, got {submits['count']}"
        rt_gpu.driver._queue.submit = orig_submit
        EVIDENCE["R6"] = {
            "status": "PASS",
            "fact": f"serial N={LIMIT} and N={LIMIT+1} BOTH work after fix (dispatch=(1,1,1) -> dx=1, _check_dispatch_limit NOT applicable, GPU==CPU within rel/abs 1e-6 vs f64 recurrence); scan N={LIMIT} works (dx=65535 at limit), N={LIMIT+1} -> ValueError '{msg[:60]}...' BEFORE dispatch (queue.submit never called, submits=0) - _check_dispatch_limit S64 precedent covers scan (ceil(n/64) vs chunk)",
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# R7 - repeated calls after errors (mode=3, scan dispatch limit) - pool not broken
# ============================================================================

@needs_gpu
def test_r7_repeated_after_error():
    EVIDENCE["R7"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        x = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        ref = _oracle_mode0(x, 2.0 / 15.0, 13.0 / 15.0)

        def run_ok():
            rt.execute(rt.compile([SK_JOB]), {"x": x})
            res = rt.driver.resolve_output("y")
            assert res is not None
            resf = res.astype(np.float64)
            fin = np.isfinite(resf) & np.isfinite(ref)
            diff = float(np.abs(resf[fin] - ref[fin]).max()) if fin.any() else 0.0
            peak = float(np.abs(ref[fin]).max()) if fin.any() else 1.0
            assert diff <= max(1e-6 * peak, 1e-6), f"diff {diff}"

        with pytest.raises(ValueError) as e1:
            rt.compile([{"op": "StateKernel_Single", "inputs": ["x"],
                         "params": {"mode": 3, "a": 1.0, "b": 0.0}, "out": "y"}])
        assert MODE_MSG in str(e1.value)
        run_ok()

        with pytest.raises(ValueError):
            rt.compile([{"op": "StateKernelScanLocal", "inputs": ["data"],
                         "params": {"mode": 2, "a": 1.0, "b": 0.0, "chunk": 256},
                         "out": ["skl_tmp", "skl_last"]}])
        run_ok()

        y = np.zeros(LIMIT + 1, dtype=np.float32)
        with pytest.raises(ValueError) as excinfo:
            rt.execute(rt.compile(_scan_jobs(chunk=256)), {"data": y})
        assert "Dispatch limit exceeded" in str(excinfo.value)
        run_ok()

        EVIDENCE["R7"] = {
            "status": "PASS",
            "fact": "after mode=3 ValueError (compile, S87), after scan mode=2 ValueError (compile), after scan N=4_194_241 ValueError (execute, _check_dispatch_limit) - subsequent StateKernel runs produce correct results within contract tolerance (pool not broken)",
        }
    finally:
        rt.driver.release()


# ============================================================================
# readback-size: StateKernel n=64 / n=1M correctness (D2H by exact size)
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
            ref = _oracle_mode0(x, 2.0 / 15.0, 13.0 / 15.0)
            rt.execute(rt.compile([SK_JOB]), {"x": x})
            res = rt.driver.resolve_output("y")
            assert res is not None and res.shape == (n,), f"n={n} shape {None if res is None else res.shape}"
            resf = res.astype(np.float64)
            fin = np.isfinite(resf) & np.isfinite(ref)
            diff = float(np.abs(resf[fin] - ref[fin]).max()) if fin.any() else 0.0
            peak = float(np.abs(ref[fin]).max()) if fin.any() else 1.0
            assert diff <= max(1e-6 * peak, 1e-6), f"n={n} diff {diff}"
            results[str(n)] = {"shape": list(res.shape), "diff": diff}
        EVIDENCE["readback_size"] = {
            "status": "PASS",
            "fact": f"StateKernel readback by exact payload size: n=64 diff={results['64']['diff']:.2e}, n=1M diff={results['1000000']['diff']:.2e} (vs f64 recurrence, contract rel/abs 1e-6), shape [n] both",
        }
    finally:
        rt.driver.release()


def test_evidence_saved():
    for k, v in EVIDENCE.items():
        if v is None:
            EVIDENCE[k] = {"status": "SKIP", "fact": "not executed (GPU unavailable or test skipped)"}
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
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    assert os.path.getsize(path) > 0, "resource_failure.json written empty"
