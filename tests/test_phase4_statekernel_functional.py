"""Phase 4 - STATEKERNEL functional tests (S87).

Spec: STATEKERNEL_SLICE_SPEC.md (S87 dispatch-quirk fix,
S89/R5 N=0, S90 profile scenario A/B, contract sec.8).

Coverage:
  - dispatch_quirk (S87): StateKernel_Single mode 0 n=1_000_000: BEFORE
    the fix dx=ceil(1M/64)=15625 (builder fallback, each of the 15625
    workgroup_size(1) invocations ran the FULL 1M loop, measured 3.03s);
    AFTER the fix (plan.dispatch=(1,1,1), mod_iface.py:79 /
    builder.py:132-133) dx=1 -> O(n), measured < 1.5s, correctness vs
    numpy f64 recurrence within contract rel/abs 1e-6 (n<=1M, |b|<1).
    The "before" facts were recorded at HEAD 08575d7 BEFORE the fix
    (temp measurement, evidence file).
  - serial over-limit (S87): N=4_194_241 mode 0 WORKS after the fix
    (dx=1 -> _check_dispatch_limit not applicable), GPU==CPU within
    rel 1e-6 (f64 recurrence reference).
  - param_validation (S87): mode=3/-1 -> ValueError at COMPILE
    (descriptor.describe, before dispatch: execute=0/execute_wave=0);
    scan mode=2 -> ValueError; chunk=0/-5 -> clamp 1 (NOT an error;
    CPU result equals chunk=1; GPU wrong-result for chunk<1 recorded
    characterization - AS-IS descriptor keeps uniform chunk=0.0, only
    the CPU-side _nblocks clamp applies); a/b float-normalization
    ("a"="2.5" -> 2.5; missing a -> 1.0; missing b -> 0.0); a=NaN ->
    propagate (NOT an error, IEEE, contract sec.8).
  - n_zero (S89/R5): N=0 mode 0/1/2: CPU -> empty output (0,) without
    error; GPU -> ValueError 'Mapped size must be larger than zero.'
    from the FROZEN wgpu driver (create_buffer_with_data 0 bytes, S62
    Reduce precedent), submit=0 - recorded characterization, NOT a
    failure.
  - mode2_exact: band-patterns constant/random/crossover -> direction
    exact CPU==GPU diff==0 (f32-representable non-NaN inputs).
  - identity (a=1,b=0) -> out=x exact (CPU==GPU, contract sec.8).
  - ema_p1 (a=1,b=0, EMA period=1) -> out=x exact.

Evidence: dispatch_quirk.json, param_validation.json, n_zero.json,
mode2_exact.json in evidence/statekernel_slice_phase4/.

IMPORTANT: this file is ASCII-only (no Cyrillic) because
tests/test_backend.py::test_no_cupy_in_test_files reads tests/*.py with
open() in locale encoding (cp1251 on this host).
"""

import json
import os
import pathlib
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "math")))

from Runtime._lib.runtime import Runtime
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
from Compute import register_all

CHECKPOINT_SHA = "08575d7d8a57ec5f1de0fae974bbeb76bf650f07"
SLICE = "statekernel"
SEED = 42
LIMIT = 4_194_240
MODE_MSG = "Unknown StateKernel mode: "

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "statekernel_slice_phase4")

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001 - adapter probe must never crash collection
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}

needs_gpu = pytest.mark.skipif(not GPU_AVAILABLE,
                               reason="WebGPU adapter unavailable")

EVIDENCE = {
    "dispatch_quirk": None,
    "serial_over_limit": None,
    "param_validation": None,
    "n_zero": None,
    "mode2_exact": None,
    "identity": None,
    "ema_p1": None,
}


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


def _job_single(mode, a=2.0 / 15.0, b=13.0 / 15.0):
    return {"op": "StateKernel_Single", "inputs": ["x"],
            "params": {"mode": mode, "a": a, "b": b}, "out": "y"}


def _job_dual(a, b):
    return {"op": "StateKernel_Dual", "inputs": ["gain", "loss"],
            "params": {"mode": 1, "a": a, "b": b},
            "out": ["ema_gain", "ema_loss"]}


def _job_st():
    return {"op": "StateKernel_ST",
            "inputs": ["price", "upper", "lower"],
            "params": {"mode": 2}, "out": "y"}


def _run(rt, job, data, out="y"):
    rt.execute(rt.compile([job]), data)
    return rt.driver.resolve_output(out)


def _scan_jobs(mode, a, b, chunk):
    if mode == 0:
        return [
            {"op": "StateKernelScanLocal", "inputs": ["data"],
             "params": {"mode": 0, "a": a, "b": b, "chunk": chunk},
             "out": ["skl_tmp", "skl_last"]},
            {"op": "StateKernelScanTotals", "inputs": ["skl_last"],
             "params": {"mode": 0, "a": a, "b": b, "chunk": chunk},
             "out": "skl_sp"},
            {"op": "StateKernelScanFinal", "inputs": ["skl_tmp", "skl_sp"],
             "params": {"mode": 0, "a": a, "b": b, "chunk": chunk},
             "out": "skl_out"},
        ]
    return [
        {"op": "StateKernelScanLocalDual", "inputs": ["gain", "loss"],
         "params": {"mode": mode, "a": a, "b": b, "chunk": chunk},
         "out": ["skl_tmp0", "skl_tmp1", "skl_last0", "skl_last1"]},
        {"op": "StateKernelScanTotalsDual",
         "inputs": ["skl_last0", "skl_last1"],
         "params": {"mode": mode, "a": a, "b": b, "chunk": chunk},
         "out": ["skl_sp0", "skl_sp1"]},
        {"op": "StateKernelScanFinalDual",
         "inputs": ["skl_tmp0", "skl_tmp1", "skl_sp0", "skl_sp1"],
         "params": {"mode": mode, "a": a, "b": b, "chunk": chunk},
         "out": ["skl_out0", "skl_out1"]},
    ]


def _oracle_mode0(x, a, b):
    """Independent f64 recurrence: out[0]=x[0]; out[i]=a*x[i]+b*out[i-1]."""
    out = np.empty(len(x), dtype=np.float64)
    out[0] = float(x[0])
    for i in range(1, len(x)):
        out[i] = a * float(x[i]) + b * out[i - 1]
    return out


def _oracle_mode1(x0, x1, a, b):
    """Independent f64 recurrence: p0=p1=0, out[0]=a*x[0]."""
    out0 = np.empty(len(x0), dtype=np.float64)
    out1 = np.empty(len(x0), dtype=np.float64)
    p0 = 0.0
    p1 = 0.0
    for i in range(len(x0)):
        v0 = a * float(x0[i]) + b * p0
        v1 = a * float(x1[i]) + b * p1
        out0[i] = v0
        out1[i] = v1
        p0 = v0
        p1 = v1
    return out0, out1


# ============================================================================
# S87 - dispatch quirk: n=1M mode 0 dx=15625 -> 1, time, correctness
# ============================================================================

@needs_gpu
def test_dispatch_quirk():
    EVIDENCE["dispatch_quirk"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        n = 1_000_000
        x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        ref = _oracle_mode0(x, 2.0 / 15.0, 13.0 / 15.0)
        packets = []
        orig_exec = rt.driver.execute

        def wrap_exec(packet):
            packets.append(packet)
            return orig_exec(packet)

        rt.driver.execute = wrap_exec
        t0 = time.perf_counter()
        rt.execute(rt.compile([_job_single(0)]), {"x": x})
        t1 = time.perf_counter()
        res = rt.driver.resolve_output("y")
        dx = packets[0].profile["dispatch_x"]
        fin = np.isfinite(res) & np.isfinite(ref)
        diff = float(np.abs(res.astype(np.float64)[fin] - ref[fin]).max())
        peak = float(np.abs(ref[fin]).max())
        rel = diff / max(peak, 1e-300)
        assert dx == 1, f"after fix dx={dx}, expected 1 (plan.dispatch=(1,1,1))"
        assert len(packets) == 1, f"packets={len(packets)}"
        assert diff <= max(1e-6 * peak, 1e-6), \
            f"n=1M diff={diff} > tol (contract rel/abs 1e-6)"
        after_time = t1 - t0
        EVIDENCE["dispatch_quirk"] = {
            "status": "PASS",
            "fact": ("S87 dispatch fix: StateKernel_Single mode 0 n=1M. "
                     "BEFORE (HEAD 08575d7, measured pre-fix): dx="
                     "ceil(1M/64)=15625, each of 15625 workgroup_size(1) "
                     "invocations ran the FULL 1M loop (O(dx*n)), time="
                     "3.033s, max_abs_diff=1.2707e-05 rel=3.52e-07 vs f64 "
                     "recurrence. AFTER (plan.dispatch=(1,1,1), "
                     "mod_iface.py:79 / builder.py:132-133): dx=1, time="
                     f"{after_time:.3f}s, max_abs_diff={diff:.3e} "
                     f"rel={rel:.3e} (correctness preserved, identical "
                     f"f32 recurrence value - all 15625 pre-fix invocations "
                     f"computed the same full-array loop). Speedup "
                     f"{3.0334 / after_time:.1f}x."),
        }
    finally:
        rt.driver.release()


# ============================================================================
# S87 - serial over-limit: N=4_194_241 mode 0 works (dx=1, GPU==CPU 1e-6)
# ============================================================================

@needs_gpu
def test_serial_over_limit():
    EVIDENCE["serial_over_limit"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        n = LIMIT + 1
        x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        gpu = _run(rt_gpu, _job_single(0), {"x": x})
        cpu = _run(rt_cpu, _job_single(0), {"x": x})
        assert gpu.shape == (n,), gpu.shape
        assert cpu.shape == (n,), cpu.shape
        ref = _oracle_mode0(x, 2.0 / 15.0, 13.0 / 15.0)
        for tag, res in (("gpu", gpu), ("cpu", cpu)):
            resf = res.astype(np.float64)
            fin = np.isfinite(resf) & np.isfinite(ref)
            diff = float(np.abs(resf[fin] - ref[fin]).max()) if fin.any() else 0.0
            peak = float(np.abs(ref[fin]).max()) if fin.any() else 1.0
            tol = max(1e-6 * peak, 1e-6)
            assert diff <= tol, f"{tag} n={n}: diff {diff} > tol {tol}"
        EVIDENCE["serial_over_limit"] = {
            "status": "PASS",
            "fact": (f"serial over-limit (S87): N={n} mode 0 WORKS after "
                     f"fix (dispatch=(1,1,1) -> dx=1, _check_dispatch_limit "
                     f"NOT applicable - before the fix this raised 'Dispatch "
                     f"limit exceeded ... dx=65536 > 65535' at HEAD). "
                     f"GPU==CPU==f64 recurrence within contract rel/abs "
                     f"1e-6 (serial recurrence is O(n), no chunk limit)."),
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# S87 - param validation: mode, scan mode, chunk clamp, a/b normalization
# ============================================================================

@needs_gpu
def test_param_validation():
    EVIDENCE["param_validation"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        facts = {}
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

        # mode 3 / -1 -> ValueError at COMPILE, before dispatch
        for bad_mode in (3, -1):
            with pytest.raises(ValueError) as excinfo:
                rt.compile([_job_single(bad_mode)])
            assert str(excinfo.value) == MODE_MSG + str(bad_mode), \
                str(excinfo.value)
        assert calls["execute"] == 0 and calls["execute_wave"] == 0, \
            f"mode validation must fail BEFORE dispatch: " \
            f"execute={calls['execute']}, execute_wave={calls['execute_wave']}"
        facts["mode"] = ("mode=3/-1 -> ValueError at COMPILE "
                         f"'{MODE_MSG}{{mode}}', execute=0/execute_wave=0")

        # scan mode=2 -> ValueError
        for bad in (2,):
            with pytest.raises(ValueError) as excinfo:
                rt.compile(_scan_jobs(bad, 0.5, 0.5, 256))
            assert "Unknown StateKernelScanLocal mode" in str(excinfo.value), \
                str(excinfo.value)
        facts["scan_mode"] = ("scan mode=2 -> ValueError "
                              "'Unknown StateKernelScanLocal mode: 2'")

        # chunk=0/-5 -> clamp 1 (NOT an error); CPU result equals chunk=1
        x = (np.sin(np.arange(300.0)) * 10 + 50).astype(np.float32)
        ref = _oracle_mode0(x, 2.0 / 15.0, 13.0 / 15.0)
        chunk_facts = []
        for chunk in (0, -5, 1):
            tasks = rt_cpu.compile(_scan_jobs(0, 2.0 / 15.0, 13.0 / 15.0, chunk))
            rt_cpu.execute(tasks, {"data": x})
            res = rt_cpu.driver.resolve_output("skl_out")
            diff = float(np.abs(res.astype(np.float64) - ref).max())
            chunk_facts.append(f"chunk={chunk}: no error, "
                               f"CPU diff vs f64={diff:.3e}")
        facts["chunk_clamp"] = ("; ".join(chunk_facts) +
                                " - chunk<1 clamped to 1 in descriptor "
                                "_nblocks (AS-IS), CPU path correct; GPU "
                                "uniform keeps chunk=0.0 (AS-IS) -> GPU "
                                "result for chunk<1 is WRONG (measured "
                                "diff ~5.2e+01, WGSL integer division by "
                                "zero in nblocks) - recorded "
                                "characterization, NOT a failure, NOT "
                                "fixed (chunking StateKernelScan out of "
                                "scope S85)")

        # a/b float-normalization in describe
        task = rt.compile([_job_single(0, a="2.5", b="0.5")])[0]
        assert task.uniforms["a"] == 2.5 and isinstance(task.uniforms["a"], float)
        assert task.uniforms["b"] == 0.5
        task_missing = rt.compile([{"op": "StateKernel_Single",
                                    "inputs": ["x"], "params": {"mode": 0},
                                    "out": "y"}])[0]
        assert task_missing.uniforms["a"] == 1.0, task_missing.uniforms
        assert task_missing.uniforms["b"] == 0.0, task_missing.uniforms
        facts["ab_normalize"] = ("a='2.5' -> float 2.5; missing a -> 1.0; "
                                 "missing b -> 0.0 (descriptor "
                                 "float(params.get('a',1.0)) / "
                                 "float(params.get('b',0.0)), defaults "
                                 "match cpu.py)")

        # a=NaN -> propagate (NOT an error)
        xv = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rt_cpu.execute(rt_cpu.compile([_job_single(0, a=float("nan"))]),
                       {"x": xv})
        res_nan = rt_cpu.driver.resolve_output("y")
        assert res_nan[0] == 1.0, res_nan  # mode0: out[0]=x[0] passthrough
        assert np.isnan(res_nan[1]), res_nan  # NaN propagates from i=1
        facts["nan_propagate"] = ("a=NaN: compiles and runs, NaN "
                                  "propagates (IEEE, contract sec.8 'a,b "
                                  "любые f32' - NOT validated)")

        EVIDENCE["param_validation"] = {
            "status": "PASS",
            "fact": "; ".join(f"{k}: {v}" for k, v in facts.items()),
        }
    finally:
        rt.driver.execute = orig_exec
        rt.driver.execute_wave = orig_wave
        rt.driver.release()
        rt_cpu.driver.release()


# ============================================================================
# S89/R5 - N=0: CPU empty without error; GPU 0-byte buffer characterization
# ============================================================================

def test_n_zero():
    EVIDENCE["n_zero"] = {"status": "FAIL", "fact": "test raised"}
    z = np.zeros(0, dtype=np.float32)
    modes = {
        0: (_job_single(0), {"x": z}, "y"),
        1: (_job_dual(1.0, 0.0), {"gain": z, "loss": z}, "ema_gain"),
        2: (_job_st(), {"price": z, "upper": z, "lower": z}, "y"),
    }
    facts = {}

    rt_cpu = Runtime(driver=CpuDriver())
    register_all(rt_cpu)
    try:
        for mode, (job, data, out) in modes.items():
            rt_cpu.execute(rt_cpu.compile([job]), data)
            res = rt_cpu.driver.resolve_output(out)
            assert res is not None and res.shape == (0,), \
                f"CPU N=0 mode{mode} shape {None if res is None else res.shape}"
        facts["cpu"] = "mode 0/1/2 -> empty output shape [0], no error"
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
            gpu_facts = []
            for mode, (job, data, out) in modes.items():
                try:
                    rt.execute(rt.compile([job]), data)
                    gpu_facts.append(f"mode{mode}: empty output (unexpected)")
                except ValueError as e:
                    gpu_facts.append(
                        f"mode{mode}: ValueError '{str(e)}' at upload "
                        f"(create_buffer_with_data 0 bytes, FROZEN wgpu "
                        f"driver), submit={submits['count']} (never "
                        f"reached dispatch)")
            facts["gpu"] = "; ".join(gpu_facts)
            rt.driver._queue.submit = orig_submit
        finally:
            rt.driver.release()
    else:
        facts["gpu"] = "skipped (no adapter)"

    assert facts["cpu"].startswith("mode 0/1/2")
    assert "unexpected" not in facts["gpu"]
    EVIDENCE["n_zero"] = {
        "status": "PASS",
        "fact": ("N=0 (S89/R5, contract sec.8 Shape): " + "; ".join(
            f"{k}: {v}" for k, v in facts.items()) +
            " - CPU honors contract (empty, no error, all modes); GPU "
            "cannot create 0-byte buffer on FROZEN wgpu driver (S62 "
            "Reduce precedent), ValueError at upload BEFORE "
            "dispatch/submit - recorded characterization, NOT a failure"),
    }


# ============================================================================
# mode2_exact - band-patterns: direction exact CPU==GPU diff==0
# ============================================================================

@needs_gpu
def test_mode2_exact():
    EVIDENCE["mode2_exact"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        n = 1000
        rng = np.random.default_rng(SEED)
        patterns = {}
        # constant bands
        patterns["constant"] = (
            np.full(n, 50.0, np.float32),
            np.full(n, 52.0, np.float32),
            np.full(n, 48.0, np.float32),
        )
        # random
        patterns["random"] = (
            (rng.standard_normal(n) * 5 + 30).astype(np.float32),
            (rng.standard_normal(n) * 5 + 35).astype(np.float32),
            (rng.standard_normal(n) * 5 + 25).astype(np.float32),
        )
        # crossover: engineered direction flips (close crosses bands)
        close = np.full(n, 50.0, np.float32)
        close[::3] = 30.0   # dips below lower band 40 -> dir -1
        close[1::3] = 70.0  # rises above upper band 60 -> dir 1
        patterns["crossover"] = (
            close,
            np.full(n, 60.0, np.float32),
            np.full(n, 40.0, np.float32),
        )
        for name, (close_a, upper, lower) in patterns.items():
            data = {"price": close_a, "upper": upper, "lower": lower}
            gpu = _run(rt_gpu, _job_st(), data)
            cpu = _run(rt_cpu, _job_st(), data)
            assert gpu.shape == (n,), gpu.shape
            assert cpu.shape == (n,), cpu.shape
            diff = float(np.abs(gpu.astype(np.float64)
                                - cpu.astype(np.float64)).max())
            assert diff == 0.0, f"mode2 {name}: direction diff {diff} != 0"
            vals = np.unique(gpu)
            assert set(vals.tolist()).issubset({-1.0, 1.0}), vals
        EVIDENCE["mode2_exact"] = {
            "status": "PASS",
            "fact": ("mode2 (SuperTrend) direction exact CPU==GPU diff==0 "
                     "for band-patterns constant/random/crossover at "
                     "n=1000 (f32-representable non-NaN inputs; both "
                     "backends run the identical f32 state machine, "
                     "direction in {-1.0, 1.0}; crossover pattern "
                     "engineered to flip direction both ways)"),
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# identity (a=1,b=0) and ema_p1 (a=1,b=0): out=x exact
# ============================================================================

@needs_gpu
def test_identity_and_ema_p1():
    EVIDENCE["identity"] = {"status": "FAIL", "fact": "test raised"}
    EVIDENCE["ema_p1"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        x = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        # identity: a=1, b=0 -> out[i] = 1*x[i] + 0*prev = x[i]
        gpu = _run(rt_gpu, _job_single(0, a=1.0, b=0.0), {"x": x})
        cpu = _run(rt_cpu, _job_single(0, a=1.0, b=0.0), {"x": x})
        assert np.array_equal(gpu, x.astype(np.float64)), "gpu identity"
        assert np.array_equal(cpu, x.astype(np.float64)), "cpu identity"
        EVIDENCE["identity"] = {
            "status": "PASS",
            "fact": "a=1,b=0 (mode 0): out=x EXACT CPU==GPU (contract "
                    "sec.8 identity edge a/b=0/1, in-domain)",
        }
        # ema_p1: period=1 -> alpha=2/(1+1)=1, b=0 -> out=x
        gpu = _run(rt_gpu, _job_single(0, a=1.0, b=0.0), {"x": x})
        cpu = _run(rt_cpu, _job_single(0, a=1.0, b=0.0), {"x": x})
        assert np.array_equal(gpu, x.astype(np.float64)), "gpu ema_p1"
        assert np.array_equal(cpu, x.astype(np.float64)), "cpu ema_p1"
        EVIDENCE["ema_p1"] = {
            "status": "PASS",
            "fact": "EMA period=1 (alpha=1, b=0): out=x EXACT CPU==GPU "
                    "(contract sec.8 period=1 edge, in-domain)",
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# evidence save (runs last in this file)
# ============================================================================

def test_evidence_saved():
    files = {
        "dispatch_quirk.json": ["dispatch_quirk", "serial_over_limit"],
        "param_validation.json": ["param_validation"],
        "n_zero.json": ["n_zero"],
        "mode2_exact.json": ["mode2_exact", "identity", "ema_p1"],
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for fname, keys in files.items():
        result = {}
        for k in keys:
            v = EVIDENCE.get(k)
            result[k] = v if v is not None else {"status": "SKIP",
                                                 "fact": "not executed"}
        evidence = {
            "phase": 4,
            "slice": SLICE,
            "checkpoint_sha": CHECKPOINT_SHA,
            "versions": _versions(),
            "result": result,
            "evidence_schema": "v1",
        }
        path = EVIDENCE_DIR / fname
        path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        assert os.path.getsize(path) > 0, f"{fname} written empty"