"""Phase 4 - REDUCE functional tests (S62-S64).

Spec: REDUCE_SLICE_SPEC.md (S62 N=0 semantics,
S63 CPU min/max IEEE, S64 dispatch limit 4_194_240).

Coverage:
  - N=0 (S62, unified by user decision 2026-08-20): ValueError on BOTH
    backends for ALL ops (sum/min/max), raised by descriptor._output_size
    (builder, before dispatch): execute=0/execute_wave=0, GPU not
    submitted, message identical to cpu.py defense-in-depth. sum 0.0
    neutral NOT implemented - GPU 0-byte buffer impossible on frozen
    wgpu_driver (PHASE4_REDUCE_GATE sec 5.1), single predictable error.
  - min/max IEEE (S63): CPU==oracle (np.min/np.max f64): NaN mask
    (isnan parity, propagate), zero-sign (min(-0,+0)=-0.0,
    max(-0,+0)=+0.0); f32-representable exact CPU==GPU==oracle;
    GPU NaN min/max - known WGSL characterization (PHASE4_MAPBINARY_GATE
    sec 5.1: unordered max/min return the NON-NaN operand on NVIDIA
    RTX 2060 / Vulkan), recorded, NOT a failure.
  - dispatch limit (S64): N=4_194_240 works (sum rel 1e-4 vs np.sum,
    min/max exact); N=4_194_241 -> ValueError BEFORE dispatch on GPU
    (queue.submit never called, _check_dispatch_limit); CPU path without
    limit is correct for 4_194_241.
  - N=1: out = x[0] for all ops; out shape [1] for all ops.

Evidence:
  evidence/reduce_slice_phase4/
    n_zero.json (S62), maxmin_ieee.json (S63), dispatch_limit.json (S64).

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

CHECKPOINT_SHA = "f3a21d9"
SLICE = "reduce"
SEED = 42
LIMIT = 4_194_240

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "reduce_slice_phase4")

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
    "n_zero": None,
    "maxmin_ieee": None,
    "dispatch_limit": None,
}

_SUM_JOB = {"op": "Reduce", "inputs": ["x"], "params": {}, "out": "y"}


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


def _make_cpu_runtime():
    rt = Runtime(driver=CpuDriver())
    register_all(rt)
    return rt


def _run(rt, job, data, out="y"):
    rt.execute(rt.compile([job]), data)
    return rt.driver.resolve_output(out)


def _job_for(op):
    return {"op": "Reduce", "inputs": ["x"], "params": {"op": op}, "out": "y"}


# ============================================================================
# S62 - N=0: ValueError on BOTH backends for ALL ops, BEFORE dispatch
# ============================================================================

def test_n_zero_all_ops_error():
    EVIDENCE["n_zero"] = {"status": "FAIL", "fact": "test raised"}
    x = np.zeros(0, dtype=np.float32)
    msg_expect = ("Reduce on empty input (n=0) is undefined; "
                  "use Fill or check input length")
    results = {}
    # op forms: string and numeric (int/float) - descriptor handles both
    for tag, cls in (("cpu", CpuDriver), ("gpu", WebGpuDriver)):
        if tag == "gpu" and not GPU_AVAILABLE:
            results["gpu"] = "skipped (no adapter)"
            continue
        rt = Runtime(driver=cls())
        register_all(rt)
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
            for op in ("sum", 0, 0.0, "min", 1, 1.0, "max", 2, 2.0):
                with pytest.raises(ValueError) as excinfo:
                    _run(rt, _job_for(op), {"x": x})
                msg = str(excinfo.value)
                assert msg == msg_expect, msg
            assert calls["execute"] == 0 and calls["execute_wave"] == 0, \
                f"{tag}: N=0 must fail BEFORE dispatch: " \
                f"execute={calls['execute']}, execute_wave={calls['execute_wave']}"
            results[tag] = {
                "op_forms_tested": ["sum", "0", "0.0", "min", "1", "1.0",
                                    "max", "2", "2.0"],
                "valueerror": "same message on all forms",
                "execute_calls": calls["execute"],
                "execute_wave_calls": calls["execute_wave"],
                "msg": msg_expect,
            }
        finally:
            rt.driver.release()

    fact = ("N=0 -> ValueError BEFORE dispatch on both backends for ALL "
            "ops (sum/min/max; descriptor._output_size, builder, single "
            "shared path): "
            + "; ".join(f"{k}={v}" for k, v in results.items())
            + "; GPU not submitted (guard before dispatch); sum 0.0 "
            "neutral NOT implemented (GPU 0-byte buffer impossible on "
            "frozen wgpu_driver, PHASE4_REDUCE_GATE sec 5.1; decision "
            "user 2026-08-20)")
    EVIDENCE["n_zero"] = {"status": "PASS", "fact": fact}


# ============================================================================
# S62 - min/max N=0: ValueError on BOTH backends, BEFORE dispatch
# ============================================================================

def test_minmax_n_zero_error():
    EVIDENCE["n_zero"] = {"status": "FAIL", "fact": "test raised"}
    x = np.zeros(0, dtype=np.float32)
    msg_expect = ("Reduce on empty input (n=0) is undefined; "
                  "use Fill or check input length")
    results = {}
    # op forms: string and numeric (int/float) - descriptor handles both
    for tag, cls in (("cpu", CpuDriver), ("gpu", WebGpuDriver)):
        if tag == "gpu" and not GPU_AVAILABLE:
            results["gpu"] = "skipped (no adapter)"
            continue
        rt = Runtime(driver=cls())
        register_all(rt)
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
            seen = {}
            for op in ("min", 1, 1.0, "max", 2, 2.0):
                with pytest.raises(ValueError) as excinfo:
                    _run(rt, _job_for(op), {"x": x})
                msg = str(excinfo.value)
                assert msg == msg_expect, msg
                seen[str(op)] = msg[:40]
            assert calls["execute"] == 0 and calls["execute_wave"] == 0, \
                f"{tag}: min/max N=0 must fail BEFORE dispatch: " \
                f"execute={calls['execute']}, execute_wave={calls['execute_wave']}"
            results[tag] = {
                "op_forms_tested": ["min", "1", "1.0", "max", "2", "2.0"],
                "valueerror": "same message on all forms",
                "execute_calls": calls["execute"],
                "execute_wave_calls": calls["execute_wave"],
                "msg": msg_expect,
            }
        finally:
            rt.driver.release()
    # CPU defense-in-depth: direct cpu() call outside Runtime must also raise
    from Compute._lib.reduce import cpu as reduce_cpu
    from Runtime._lib.mod_iface import ExecutionContext, Buffer
    for op, name in ((0, "sum"), (1, "min"), (2, "max")):
        src_arr = np.zeros(0, dtype=np.float32)
        dst_arr = np.zeros(1, dtype=np.float64)
        ctx = ExecutionContext(
            inputs=[Buffer(view=src_arr, dtype="float", size=0)],
            outputs=[Buffer(view=dst_arr, dtype="float", size=1)],
            uniforms={"op": op},
        )
        with pytest.raises(ValueError) as excinfo:
            reduce_cpu(ctx)
        assert msg_expect in str(excinfo.value), str(excinfo.value)

    fact = ("min/max N=0 -> ValueError BEFORE dispatch on both backends "
            "(descriptor._output_size, builder, single shared path): "
            + "; ".join(f"{k}={v}" for k, v in results.items())
            + "; CPU defense-in-depth direct cpu(): same ValueError for "
            "sum/min/max")
    if EVIDENCE["n_zero"] and EVIDENCE["n_zero"]["status"] == "PASS":
        prev = EVIDENCE["n_zero"]["fact"]
        EVIDENCE["n_zero"] = {"status": "PASS", "fact": prev + " | " + fact}
    else:
        EVIDENCE["n_zero"] = {"status": "PASS", "fact": fact}


# ============================================================================
# S63 - min/max IEEE: CPU==oracle (NaN mask, zero sign); f32 exact; GPU
#       NaN = known WGSL characterization (recorded, NOT a failure)
# ============================================================================

@needs_gpu
def test_minmax_ieee():
    EVIDENCE["maxmin_ieee"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        rows = {}
        # (1) f32-representable: CPU==GPU==oracle, exact
        for n in (1, 100, 65536):
            x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            for op, name in ((1, "min"), (2, "max")):
                np_fn = np.min if name == "min" else np.max
                oracle = float(np_fn(x.astype(np.float64)))
                cpu = float(_run(rt_cpu, _job_for(op), {"x": x})[0])
                gpu = float(_run(rt_gpu, _job_for(op), {"x": x})[0])
                assert cpu == oracle, f"cpu {name} n={n}: {cpu} != {oracle}"
                assert gpu == oracle, f"gpu {name} n={n}: {gpu} != {oracle}"
                rows[f"f32_n{n}_{name}"] = {"cpu": cpu, "gpu": gpu,
                                            "oracle": oracle, "exact": True}
        # (2) NaN: CPU==oracle (IEEE propagate, isnan mask); GPU - fact
        nan_facts = {}
        for op, name in ((1, "min"), (2, "max")):
            x = np.array([1.0, np.nan, 3.0, np.nan, 2.0], dtype=np.float32)
            oracle = (np.min if name == "min" else np.max)(x.astype(np.float64))
            cpu = float(_run(rt_cpu, _job_for(op), {"x": x})[0])
            gpu = float(_run(rt_gpu, _job_for(op), {"x": x})[0])
            assert np.isnan(cpu) and np.isnan(oracle), \
                f"cpu/oracle {name} must propagate NaN"
            nan_facts[name] = {
                "cpu": "nan (IEEE propagate)",
                "oracle": "nan",
                "gpu": _num(gpu),
            }
            rows[f"nan_{name}"] = {
                "cpu": "nan", "oracle": "nan", "gpu": _num(gpu),
                "note": "GPU WGSL min/max unordered -> non-NaN operand "
                        "(PHASE4_MAPBINARY_GATE sec 5.1 characterization)",
            }
        # (3) zero-sign: IEEE 754-2019 - CPU must return the IEEE value
        #     (min(-0,+0)=-0, max(-0,+0)=+0); oracle np.min/np.max f64 is
        #     ORDER-DEPENDENT for mixed zeros on numpy 2.5.1 (returns the
        #     second operand's sign: np.min([-0.,0.])=+0.) - numpy platform
        #     fact, recorded as observation, NOT a CPU failure (CPU follows
        #     IEEE 754-2019 by construction, _min_ieee/_max_ieee correct
        #     numpy); GPU - fact (WGSL unordered for equal values)
        zfact = {}
        for op, name, ieee_exp in ((1, "min", -0.0), (2, "max", +0.0)):
            x = np.array([-0.0, 0.0], dtype=np.float32)
            oracle = float((np.min if name == "min" else np.max)(
                x.astype(np.float64)))
            cpu = float(_run(rt_cpu, _job_for(op), {"x": x})[0])
            gpu = float(_run(rt_gpu, _job_for(op), {"x": x})[0])
            cpu_sb = bool(np.signbit(cpu))
            gpu_sb = bool(np.signbit(gpu))
            oracle_sb = bool(np.signbit(oracle))
            ieee_sb = bool(np.signbit(ieee_exp))
            assert cpu == 0.0 and gpu == 0.0
            assert cpu_sb == ieee_sb, \
                f"{name}: cpu sign {cpu_sb} != IEEE {ieee_sb}"
            zfact[name] = {
                "cpu": f"{cpu!r}(signbit={cpu_sb})",
                "ieee_expected": f"{ieee_exp!r}(signbit={ieee_sb})",
                "oracle_np": f"{oracle!r}(signbit={oracle_sb})",
                "gpu": f"{gpu!r}(signbit={gpu_sb})",
                "note": "numpy 2.5.1 np.min/np.max reduction is "
                        "order-dependent for mixed zeros (returns second "
                        "operand sign) - platform fact, CPU follows IEEE "
                        "754-2019 (corrected in _min_ieee/_max_ieee); "
                        "GPU zero-sign - WGSL unordered, fact",
            }
            rows[f"zerosign_{name}"] = zfact[name]
        EVIDENCE["maxmin_ieee"] = {
            "status": "PASS",
            "fact": ("f32-representable exact CPU==GPU==oracle: "
                     + "; ".join(
                         f"n{n}_{name}: cpu={r['cpu']!r} gpu={r['gpu']!r}"
                         for k, r in rows.items() if k.startswith("f32"))
                     + "; NaN: CPU==oracle (IEEE propagate, isnan mask), "
                       "GPU WGSL max/min returns the NON-NaN operand on "
                       "this platform (NVIDIA RTX 2060, Vulkan) - known "
                       "characterization PHASE4_MAPBINARY_GATE sec 5.1, "
                       "recorded NOT failed: "
                     + "; ".join(f"{k}: cpu={v['cpu']}, gpu={v['gpu']!r}"
                                 for k, v in nan_facts.items())
                     + "; zero-sign IEEE (CPU follows IEEE 754-2019; "
                       "oracle np.min/np.max on numpy 2.5.1 is "
                       "order-dependent for mixed zeros - platform fact "
                       "recorded; GPU - WGSL unordered, fact): "
                     + "; ".join(f"{k}: cpu={v['cpu']} "
                                 f"ieee={v['ieee_expected']} "
                                 f"oracle_np={v['oracle_np']} "
                                 f"gpu={v['gpu']}" for k, v in zfact.items())),
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# S64 - dispatch limit: 4_194_240 works; 4_194_241 -> ValueError BEFORE
#       dispatch on GPU (_check_dispatch_limit); CPU path correct
# ============================================================================

@needs_gpu
def test_dispatch_limit():
    EVIDENCE["dispatch_limit"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    orig_submit = None
    try:
        rng = np.random.default_rng(SEED)
        # (1) N=4_194_240 (limit, dx=65535): works, sum rel 1e-4, min/max exact
        n = LIMIT
        x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        ref_sum = float(np.sum(x.astype(np.float64)))
        rows = {}
        for op, name in ((0, "sum"), (1, "min"), (2, "max")):
            job = _job_for(op)
            gpu = float(_run(rt_gpu, job, {"x": x})[0])
            cpu = float(_run(rt_cpu, job, {"x": x})[0])
            oracle = ({"sum": np.sum, "min": np.min, "max": np.max}[name])(
                x.astype(np.float64))
            rows[name] = {"gpu": gpu, "cpu": cpu, "oracle": float(oracle)}
            if name == "sum":
                rel_g = abs(gpu - oracle) / max(abs(oracle), 1e-300)
                rel_c = abs(cpu - oracle) / max(abs(oracle), 1e-300)
                assert rel_g <= 1e-4, \
                    f"n={n} sum: gpu {gpu} vs oracle {oracle}, rel {rel_g}"
                assert rel_c <= 1e-4, \
                    f"n={n} sum: cpu {cpu} vs oracle {oracle}, rel {rel_c}"
            else:
                assert gpu == oracle, f"{name} n={n}: gpu {gpu} != oracle {oracle}"
                assert cpu == oracle, f"{name} n={n}: cpu {cpu} != oracle {oracle}"
        # (2) N=4_194_241: GPU -> ValueError BEFORE dispatch (no submit);
        #     CPU path without limit is correct
        n2 = LIMIT + 1
        y = (rng.standard_normal(n2) * 5 + 30).astype(np.float32)
        submits = {"count": 0}
        orig_submit = rt_gpu.driver._queue.submit

        def wrap_submit(*a, **k):
            submits["count"] += 1
            return orig_submit(*a, **k)

        rt_gpu.driver._queue.submit = wrap_submit
        with pytest.raises(ValueError) as excinfo:
            _run(rt_gpu, _SUM_JOB, {"x": y})
        msg = str(excinfo.value)
        assert "Dispatch limit exceeded" in msg, msg
        assert "4_194_240" in msg, msg
        assert "Reduce" in msg, msg
        assert submits["count"] == 0, \
            f"dispatch limit must fail BEFORE submit, got {submits['count']}"
        rt_gpu.driver._queue.submit = orig_submit
        # CPU path: no limit, correct sum
        cpu_sum = float(_run(rt_cpu, _SUM_JOB, {"x": y})[0])
        ref2 = float(np.sum(y.astype(np.float64)))
        rel2 = abs(cpu_sum - ref2) / max(abs(ref2), 1e-300)
        assert rel2 <= 1e-4, \
            f"cpu n={n2} sum {cpu_sum} vs {ref2}, rel {rel2}"
        EVIDENCE["dispatch_limit"] = {
            "status": "PASS",
            "fact": (f"N=4_194_240 (dx=65535, limit): sum gpu="
                     f"{rows['sum']['gpu']:.6e} cpu={rows['sum']['cpu']:.6e} "
                     f"oracle={rows['sum']['oracle']:.6e} (rel "
                     f"{abs(rows['sum']['gpu'] - rows['sum']['oracle']) / abs(rows['sum']['oracle']):.2e} "
                     f"<= 1e-4); min gpu==cpu==oracle="
                     f"{rows['min']['gpu']!r}; max gpu==cpu==oracle="
                     f"{rows['max']['gpu']!r}; N=4_194_241 (dx=65536): "
                     f"GPU ValueError '{msg[:60]}...' BEFORE dispatch "
                     f"(queue.submit never called, submits=0) - "
                     f"_check_dispatch_limit (wgpu_driver.execute) covers "
                     f"Reduce, dx=ceil(n/64)>65535 <=> n>4_194_240; CPU "
                     f"path without limit correct: sum={cpu_sum:.6e} vs "
                     f"oracle={ref2:.6e}, rel {rel2:.2e}"),
        }
    finally:
        if orig_submit is not None:
            rt_gpu.driver._queue.submit = orig_submit
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# N=1: out = x[0] for all ops
# ============================================================================

@needs_gpu
def test_n_one():
    x = np.array([7.25], dtype=np.float32)
    results = {}
    for tag, cls in (("cpu", CpuDriver), ("gpu", WebGpuDriver)):
        if tag == "gpu" and not GPU_AVAILABLE:
            results["gpu"] = "skipped (no adapter)"
            continue
        rt = Runtime(driver=cls())
        register_all(rt)
        try:
            outs = {}
            for op, name in ((0, "sum"), (1, "min"), (2, "max")):
                res = _run(rt, _job_for(op), {"x": x})
                assert res is not None and res.shape == (1,), \
                    f"{tag} {name} n=1 shape {None if res is None else res.shape}"
                assert float(res[0]) == 7.25, f"{tag} {name} n=1 = {res[0]}"
                outs[name] = float(res[0])
            results[tag] = outs
        finally:
            rt.driver.release()
    assert results["cpu"] == {"sum": 7.25, "min": 7.25, "max": 7.25}
    if "gpu" in results and results["gpu"] != "skipped (no adapter)":
        assert results["gpu"] == results["cpu"], results["gpu"]


# ============================================================================
# out shape [1] for all ops
# ============================================================================

@needs_gpu
def test_output_shape():
    rng = np.random.default_rng(SEED)
    x = (rng.standard_normal(100) * 5 + 30).astype(np.float32)
    results = {}
    for tag, cls in (("cpu", CpuDriver), ("gpu", WebGpuDriver)):
        if tag == "gpu" and not GPU_AVAILABLE:
            results["gpu"] = "skipped (no adapter)"
            continue
        rt = Runtime(driver=cls())
        register_all(rt)
        try:
            shapes = {}
            for op, name in ((0, "sum"), (1, "min"), (2, "max")):
                res = _run(rt, _job_for(op), {"x": x})
                assert res.shape == (1,), f"{tag} {name} shape {res.shape}"
                shapes[name] = list(res.shape)
            results[tag] = shapes
        finally:
            rt.driver.release()
    assert results["cpu"] == {"sum": [1], "min": [1], "max": [1]}, results
    if "gpu" in results and results["gpu"] != "skipped (no adapter)":
        assert results["gpu"] == results["cpu"], results["gpu"]


# ============================================================================
# evidence save (runs last in this file)
# ============================================================================

def test_evidence_saved():
    files = {
        "n_zero.json": ["n_zero"],
        "maxmin_ieee.json": ["maxmin_ieee"],
        "dispatch_limit.json": ["dispatch_limit"],
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
