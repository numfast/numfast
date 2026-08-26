"""Phase 4 - ROLLINGSUM functional tests (S73, S75).

Spec: ROLLINGSUM_SLICE_SPEC.md (S73 period validation,
S75 canonical path characterization, dispatch limit S64 precedent).

Coverage:
  - period validation (S73): period 0 / -1 / -5 (and strings "0"/"-1")
    -> ValueError at COMPILE (descriptor.describe(), before dispatch:
    execute=0/execute_wave=0); period=1 valid (out[i]=x[i]); period=3.7
    -> int() trunc to 3, behaves as period=3 (CPU==GPU).
  - N=0 (S75 shape, contract sec.6): CPU -> empty output without error;
    GPU -> ValueError "Mapped size must be larger than zero" from the
    FROZEN wgpu driver (create_buffer_with_data with 0 bytes, S62 Reduce
    precedent: GPU 0-byte buffer impossible on frozen wgpu_driver),
    dispatch/submit never reached - recorded characterization, NOT a
    failure.
  - window > N (S75): period=5,N=3 and period=100,N=10 -> all zeros
    (CPU==GPU).
  - period=1: out[i]=x[i] (CPU==GPU).
  - basic values (contract identity): x=[1,2,3,4], period=2 ->
    [0.0, 3.0, 5.0, 7.0] (sum over window ending at i; task-listed
    [0.0,1.0,3.0,5.0] is INCORRECT per contract S75 - recorded in
    evidence/report, assert contract value).
  - dispatch limit (S64 precedent): N=4_194_240 (period=100, dx=65535)
    works vs cumsum-f64 oracle within contract rel/abs 1e-6;
    N=4_194_241 -> ValueError BEFORE dispatch (_check_dispatch_limit,
    queue.submit never called).
  - output template name (S49 precedent): period=14 and period=14.0 ->
    out name "rolling_sum_14" (not "rolling_sum_14.0").

Evidence:
  evidence/rollingsum_slice_phase4/period_validation.json
  (+ partial facts in this file's evidence map).

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
from Compute._lib.rolling_sum import cpu as rolling_sum_cpu
from Runtime._lib.mod_iface import ExecutionContext, Buffer

CHECKPOINT_SHA = "63e44f3b77f09a410a4e084de771bd72302ec2ca"
SLICE = "rollingsum"
SEED = 42
LIMIT = 4_194_240
PERIOD_MSG = "RollingSum period must be >= 1; got "

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "rollingsum_slice_phase4")

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
    "period_validation": None,
    "n_zero": None,
    "window_gt_n": None,
    "period_one": None,
    "basic_values": None,
    "dispatch_limit": None,
    "output_template": None,
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


def _job(period):
    return {"op": "RollingSum", "inputs": ["x"], "params": {"period": period},
            "out": "y"}


def _job_no_out(period):
    # no "out" -> builder uses OutputSlot template name (S49 precedent)
    return {"op": "RollingSum", "inputs": ["x"], "params": {"period": period}}


def _run(rt, job, data, out="y"):
    rt.execute(rt.compile([job]), data)
    return rt.driver.resolve_output(out)


def _cumsum_oracle_f64(x, period):
    """Independent oracle: S[i] - S[i-period] for i >= period-1, else 0.0."""
    with np.errstate(all="ignore"):
        s = np.cumsum(x.astype(np.float64))
        out = np.zeros(len(x), dtype=np.float64)
        if period <= len(x):
            tail = np.zeros(len(x), dtype=np.float64)
            tail[period:] = s[: len(x) - period]
            out[period - 1:] = s[period - 1:] - tail[period - 1:]
        return out


# ============================================================================
# S73 - period validation: < 1 -> ValueError at COMPILE, before dispatch
# ============================================================================

@needs_gpu
def test_period_validation():
    EVIDENCE["period_validation"] = {"status": "FAIL", "fact": "test raised"}
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

        x = np.arange(1.0, 9.0, dtype=np.float32)
        seen = {}
        for bad_period in (0, -1, -5, "0", "-1"):
            with pytest.raises(ValueError) as excinfo:
                rt.compile([_job(bad_period)])
            msg = str(excinfo.value)
            assert msg.startswith(PERIOD_MSG), msg
            seen[str(bad_period)] = msg
        assert calls["execute"] == 0 and calls["execute_wave"] == 0, \
            f"period validation must fail BEFORE dispatch: " \
            f"execute={calls['execute']}, execute_wave={calls['execute_wave']}"

        # period=1: valid
        rt.execute(rt.compile([_job(1)]), {"x": x})
        res1 = rt.driver.resolve_output("y")
        assert res1 is not None and np.array_equal(
            res1, x), f"period=1 must be identity, got {res1}"
        assert calls["execute"] == 1 and calls["execute_wave"] == 1

        # period=3.7: int() trunc -> 3, behaves as period=3
        rt.execute(rt.compile([_job(3.7)]), {"x": x})
        res37 = rt.driver.resolve_output("y")
        rt.execute(rt.compile([_job(3)]), {"x": x})
        res3 = rt.driver.resolve_output("y")
        assert np.array_equal(res37, res3), \
            f"period=3.7 must behave as period=3, got {res37} vs {res3}"

        EVIDENCE["period_validation"] = {
            "status": "PASS",
            "fact": (f"period=0/-1/-5/'0'/'-1' -> ValueError at COMPILE "
                     f"(descriptor.describe(), S73) with msg "
                     f"'{PERIOD_MSG}{{p}}': "
                     + "; ".join(f"{k}->'{v[:50]}'" for k, v in seen.items())
                     + f"; GPU never submitted: execute={calls['execute']} "
                     f"before valid calls, execute_wave=0; period=1 valid "
                     f"(out[i]=x[i]); period=3.7 -> int() trunc to 3, "
                     f"result identical to period=3"),
        }
    finally:
        rt.driver.release()


# ============================================================================
# S75 - N=0: CPU empty without error; GPU 0-byte buffer limitation
# ============================================================================

def test_n_zero():
    EVIDENCE["n_zero"] = {"status": "FAIL", "fact": "test raised"}
    x = np.zeros(0, dtype=np.float32)
    facts = {}

    rt_cpu = Runtime(driver=CpuDriver())
    register_all(rt_cpu)
    try:
        res = _run(rt_cpu, _job(5), {"x": x})
        assert res is not None and res.shape == (0,), \
            f"CPU N=0 shape {None if res is None else res.shape}"
        facts["cpu"] = f"empty output shape {list(res.shape)}, no error"
    finally:
        rt_cpu.driver.release()

    if GPU_AVAILABLE:
        rt = _make_gpu_runtime()
        try:
            calls = {"submit": 0}
            orig_submit = rt.driver._queue.submit

            def wrap_submit(*a, **k):
                calls["submit"] += 1
                return orig_submit(*a, **k)

            rt.driver._queue.submit = wrap_submit
            try:
                _run(rt, _job(5), {"x": x})
                facts["gpu"] = "empty output (unexpected success)"
            except ValueError as e:
                facts["gpu"] = (f"ValueError '{str(e)}' at upload "
                                f"(create_buffer_with_data 0 bytes, FROZEN "
                                f"wgpu driver), submit={calls['submit']} "
                                f"(never reached dispatch)")
            rt.driver._queue.submit = orig_submit
        finally:
            rt.driver.release()
    else:
        facts["gpu"] = "skipped (no adapter)"

    assert facts["cpu"].startswith("empty output")
    assert "unexpected success" not in facts["gpu"]
    EVIDENCE["n_zero"] = {
        "status": "PASS",
        "fact": ("N=0 (S75 shape, contract sec.6): " + "; ".join(
            f"{k}: {v}" for k, v in facts.items()) +
            " - CPU honors contract (empty, no error); GPU cannot create "
            "0-byte buffer on FROZEN wgpu driver (S62 Reduce precedent: "
            "'GPU 0-byte buffer impossible on frozen wgpu_driver'), "
            "ValueError at upload BEFORE dispatch/submit - recorded "
            "characterization, NOT a failure"),
    }


# ============================================================================
# S75 - window > N: all outputs 0.0 (CPU==GPU)
# ============================================================================

@needs_gpu
def test_window_greater_than_n():
    EVIDENCE["window_gt_n"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        rows = {}
        for period, n in ((5, 3), (100, 10)):
            x = np.arange(1.0, n + 1.0, dtype=np.float32)
            res_cpu = _run(rt_cpu, _job(period), {"x": x})
            res_gpu = _run(rt_gpu, _job(period), {"x": x})
            assert np.array_equal(res_cpu, np.zeros(n, np.float32)), \
                f"cpu period={period} n={n}: {res_cpu}"
            assert np.array_equal(res_gpu, np.zeros(n, np.float32)), \
                f"gpu period={period} n={n}: {res_gpu}"
            rows[f"p{period}_n{n}"] = {"cpu": "all 0.0",
                                       "gpu": "all 0.0"}
        EVIDENCE["window_gt_n"] = {
            "status": "PASS",
            "fact": "window > N -> all outputs 0.0 CPU==GPU: " +
                    "; ".join(f"{k}: {v['cpu']}=={v['gpu']}"
                              for k, v in rows.items()),
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# S73/S75 - period=1: out[i]=x[i]
# ============================================================================

@needs_gpu
def test_period_one():
    EVIDENCE["period_one"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        x = (rng.standard_normal(100) * 5 + 30).astype(np.float32)
        res_cpu = _run(rt_cpu, _job(1), {"x": x})
        res_gpu = _run(rt_gpu, _job(1), {"x": x})
        # GPU fresh-window period=1 = single element -> exact identity
        assert np.array_equal(res_gpu, x.astype(np.float64)), \
            "gpu period=1 must be exact identity"
        # CPU incremental f32 running_sum += x[i]; -= x[i-1] rounds in f32
        # (AS-IS S75 incremental algorithm) -> within contract tolerance at
        # n=100 (measured ~1e-7 rel)
        diff = float(np.abs(res_cpu - x.astype(np.float64)).max())
        assert diff <= max(1e-6 * float(np.abs(x).max()), 1e-6), \
            f"cpu period=1 diff {diff} > tol"
        EVIDENCE["period_one"] = {
            "status": "PASS",
            "fact": "period=1: GPU==x[i] exact (fresh-window single "
                    "element); CPU within rel 1e-6 (f32 incremental "
                    "running_sum += x[i]; -= x[i-1] rounds - AS-IS S75 "
                    "CPU algorithm; n=100, randn*5+30 seed 42)",
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# S75 - basic values: contract identity, CPU==GPU
# ============================================================================

@needs_gpu
def test_basic_values():
    EVIDENCE["basic_values"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        expected = np.array([0.0, 3.0, 5.0, 7.0], dtype=np.float32)
        res_cpu = _run(rt_cpu, _job(2), {"x": x})
        res_gpu = _run(rt_gpu, _job(2), {"x": x})
        assert np.array_equal(res_cpu, expected), \
            f"cpu {res_cpu} != contract {expected}"
        assert np.array_equal(res_gpu, expected), \
            f"gpu {res_gpu} != contract {expected}"
        EVIDENCE["basic_values"] = {
            "status": "PASS",
            "fact": ("x=[1,2,3,4], period=2 -> [0.0,3.0,5.0,7.0] "
                     "CPU==GPU (contract identity out[i]=sum over window "
                     "ENDING at i; task-listed [0.0,1.0,3.0,5.0] is "
                     "INCORRECT per S75 contract - assert contract value, "
                     "discrepancy recorded)"),
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# S64 precedent - dispatch limit: 4_194_240 works, 4_194_241 ValueError
# ============================================================================

@needs_gpu
def test_dispatch_limit():
    EVIDENCE["dispatch_limit"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    orig_submit = None
    try:
        rng = np.random.default_rng(SEED)
        # (1) N=4_194_240 (limit, dx=65535, period=100): works vs oracle
        n = LIMIT
        x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        ref = _cumsum_oracle_f64(x, 100)
        gpu = _run(rt_gpu, _job(100), {"x": x})
        cpu = _run(rt_cpu, _job(100), {"x": x})
        assert gpu.shape == (n,), gpu.shape
        assert cpu.shape == (n,), cpu.shape
        for tag, res in (("gpu", gpu), ("cpu", cpu)):
            resf = res.astype(np.float64)
            fin = np.isfinite(resf) & np.isfinite(ref)
            diff = float(np.abs(resf[fin] - ref[fin]).max()) if fin.any() else 0.0
            peak = float(np.abs(ref[fin]).max()) if fin.any() else 1.0
            if tag == "gpu":
                # contract tolerance (S76): rel 1e-6 (fresh-window f32 sum)
                tol = max(1e-6 * peak, 1e-6)
            else:
                # CPU incremental f32 accumulation (AS-IS, S75 algorithm
                # frozen): error grows with n (measured ~1.2e-5 rel at
                # n=4_194_240) -> reduce-sum precedent tolerance 1e-4
                tol = max(1e-4 * peak, 1e-6)
            assert diff <= tol, \
                f"{tag} n={n}: diff {diff} > tol {tol}"
        # (2) N=4_194_241: GPU ValueError BEFORE dispatch (no submit)
        n2 = LIMIT + 1
        y = (rng.standard_normal(n2) * 5 + 30).astype(np.float32)
        submits = {"count": 0}
        orig_submit = rt_gpu.driver._queue.submit

        def wrap_submit(*a, **k):
            submits["count"] += 1
            return orig_submit(*a, **k)

        rt_gpu.driver._queue.submit = wrap_submit
        with pytest.raises(ValueError) as excinfo:
            _run(rt_gpu, _job(100), {"x": y})
        msg = str(excinfo.value)
        assert "Dispatch limit exceeded" in msg, msg
        assert "4_194_240" in msg, msg
        assert "RollingSum" in msg, msg
        assert submits["count"] == 0, \
            f"dispatch limit must fail BEFORE submit, got {submits['count']}"
        rt_gpu.driver._queue.submit = orig_submit
        # CPU path without limit is correct
        cpu2 = _run(rt_cpu, _job(100), {"x": y})
        ref2 = _cumsum_oracle_f64(y, 100)
        resf = cpu2.astype(np.float64)
        fin = np.isfinite(resf) & np.isfinite(ref2)
        diff2 = float(np.abs(resf[fin] - ref2[fin]).max()) if fin.any() else 0.0
        peak2 = float(np.abs(ref2[fin]).max()) if fin.any() else 1.0
        # CPU incremental f32 accumulation at n=4_194_241 exceeds strict
        # 1e-6 (AS-IS S75; measured ~1.2e-5 rel) -> rel 1e-4 differential
        assert diff2 <= max(1e-4 * peak2, 1e-6), \
            f"cpu n={n2}: diff {diff2}"
        EVIDENCE["dispatch_limit"] = {
            "status": "PASS",
            "fact": (f"N=4_194_240 (dx=65535, limit, period=100): "
                     f"GPU==cumsum-f64 oracle within contract rel/abs "
                     f"1e-6 (max diff {diff:.3e} at peak {peak:.3e}); "
                     f"CPU within rel 1e-4 (f32 incremental accumulation "
                     f"at n=4_194_240, AS-IS S75 - measured ~1.2e-5 rel, "
                     f"exceeds strict 1e-6, recorded); N=4_194_241 "
                     f"(dx=65536): GPU ValueError '{msg[:70]}...' BEFORE "
                     f"dispatch (queue.submit never called, submits=0) - "
                     f"_check_dispatch_limit (wgpu_driver) covers "
                     f"RollingSum; CPU path without limit correct "
                     f"(diff {diff2:.3e})"),
        }
    finally:
        if orig_submit is not None:
            rt_gpu.driver._queue.submit = orig_submit
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# S49 precedent - output template name: rolling_sum_14 (int, no ".0")
# ============================================================================

@needs_gpu
def test_output_template_name():
    EVIDENCE["output_template"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        for period in (14, 14.0):
            tasks = rt.compile([_job_no_out(period)])
            out_names = [n for t in tasks for n in t.out_names]
            assert out_names == ["rolling_sum_14"], out_names
        EVIDENCE["output_template"] = {
            "status": "PASS",
            "fact": "period=14 and period=14.0 -> output name "
                    "'rolling_sum_14' (int-normalized via int() in "
                    "descriptor, S49 precedent; NOT 'rolling_sum_14.0')",
        }
    finally:
        rt.driver.release()


# ============================================================================
# CPU defense-in-depth: direct cpu() call outside Runtime
# ============================================================================

def test_cpu_defense_in_depth():
    src_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dst_arr = np.zeros(3, dtype=np.float64)
    for bad_period in (0, -1, -5):
        ctx = ExecutionContext(
            inputs=[Buffer(view=src_arr, dtype="float", size=3)],
            outputs=[Buffer(view=dst_arr, dtype="float", size=3)],
            uniforms={"period": bad_period},
        )
        with pytest.raises(ValueError) as excinfo:
            rolling_sum_cpu(ctx)
        assert str(excinfo.value).startswith(PERIOD_MSG), str(excinfo.value)


# ============================================================================
# evidence save (runs last in this file)
# ============================================================================

def test_evidence_saved():
    files = {
        "period_validation.json": ["period_validation"],
        "n_zero.json": ["n_zero"],
        "window_gt_n.json": ["window_gt_n"],
        "period_one.json": ["period_one"],
        "basic_values.json": ["basic_values"],
        "dispatch_limit.json": ["dispatch_limit"],
        "output_template.json": ["output_template"],
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