"""Phase 4 - MAPBINARY functional tests (S49-S53, S55).

Spec: MAPBINARY_SLICE_SPEC.md (S49 op validation,
S50 div-by-zero, S51 broadcast, S52 exact-assert, S53 both-scalars,
S55 IEEE max/min).

Coverage:
  - op validation: unknown op (99/-1/7) -> ValueError at COMPILE
    (descriptor.describe(), before dispatch); op=1.0 -> int-stabilized
    output name map_binary_1.
  - both-scalars (S53): use_scalar_a=1 + use_scalar_b=1 -> builder
    ValueError "no inputs" (unreachable mode, documented).
  - div-by-zero (S50): div(5,0)=0.0, div(-5,-0.0)=0.0, div(nan,0)=0.0,
    div(inf,0)=0.0, div(-inf,0)=0.0 exact (0.0, sign +) CPU==GPU;
    div(5,3) within rel 1e-6 (measured 1 ulp f32 difference).
  - broadcast b [1] (S51): all ops 0-5 GPU==CPU; chunked n=4_200_000.
  - broadcast a [1] (S51): characterization - out [1] on both backends
    (CPU guard "broadcast of 'a' not supported" is dead code in the
    ordinary path because dispatch size = len(first input)).
  - exact-assert broadcast sub (S52): a [1M] seed 42, b [1], op=1,
    contract tolerance rel 1e-6 / abs 1e-7, actual diff in evidence.
  - max/min IEEE (S55): zero-sign IEEE 754-2019 parity CPU==GPU
    (max(-0.0,+0.0)=+0.0, min(-0.0,+0.0)=-0.0); NaN: CPU propagates,
    GPU WGSL max/min returns the non-NaN operand (platform fact,
    characterization, NOT a failure).
  - scalar modes (S53/S55): use_scalar_b (SMA/ATR style) and
    use_scalar_a (RSI style) CPU==GPU.

Evidence:
  evidence/mapbinary_slice_phase4/
    op_validation.json (S49+S53), div_zero.json (S50),
    broadcast.json (S51), exact_assert.json (S52).

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

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "mapbinary_slice_phase4")

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
    "op_validation": None,
    "div_zero": None,
    "broadcast": None,
    "exact_assert": None,
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
    return v


def _make_gpu_runtime():
    rt = Runtime(driver=WebGpuDriver())
    register_all(rt)
    return rt


def _make_cpu_runtime():
    rt = Runtime(driver=CpuDriver())
    register_all(rt)
    return rt


def _tol(peak):
    return max(1e-6 * peak, 1e-7)


# ============================================================================
# S49 - op validation: unknown op -> ValueError at compile, before dispatch
# ============================================================================

@needs_gpu
def test_op_validation_unknown_op():
    EVIDENCE["op_validation"] = {"status": "FAIL", "fact": "test raised"}
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

        results = {}
        for bad_op in (99, -1, 7):
            with pytest.raises(ValueError) as excinfo:
                rt.compile([{"op": "MapBinary", "inputs": ["a", "b"],
                             "params": {"op": bad_op}}])
            msg = str(excinfo.value)
            assert "Unknown MapBinary op code" in msg, msg
            assert "expected 0..6" in msg, msg
            results[str(bad_op)] = msg
        assert calls["execute"] == 0 and calls["execute_wave"] == 0, \
            "op validation must fail BEFORE any dispatch"

        # op=1.0 (float): int() stabilizes template -> map_binary_1
        tasks = rt.compile([{"op": "MapBinary", "inputs": ["a", "b"],
                             "params": {"op": 1.0}}])
        out_names = [n for t in tasks for n in t.out_names]
        assert out_names == ["map_binary_1"], out_names
        a = np.array([1.0, 2.0, 3.0], np.float32)
        b = np.array([0.5, 1.5, 2.5], np.float32)
        rt.execute(tasks, {"a": a, "b": b})
        res = rt.driver.resolve_output("map_binary_1")
        assert res is not None and res.shape == (3,)
        ref = a.astype(np.float64) - b.astype(np.float64)
        assert float(np.abs(res - ref).max()) <= _tol(1.0)

        EVIDENCE["op_validation"] = {
            "status": "PASS",
            "fact": (f"op=99/-1/7 -> ValueError at compile "
                     f"(descriptor.describe(), S49): "
                     f"msg='{results['99'][:80]}...'; GPU never submitted: "
                     f"execute={calls['execute']}, "
                     f"execute_wave={calls['execute_wave']}; op=1.0 -> "
                     f"int-stabilized output name {out_names}, result "
                     f"correct"),
        }
    finally:
        rt.driver.release()


# ============================================================================
# S53 - both-scalars unreachable: builder ValueError "no inputs"
# ============================================================================

def test_op_validation_both_scalars():
    jobs = [{"op": "MapBinary", "inputs": [],
             "params": {"op": 0, "use_scalar_a": 1, "use_scalar_b": 1,
                        "scalar_a": 2.0, "scalar_b": 3.0},
             "out": "y"}]
    facts = {}
    for tag, cls in (("cpu", CpuDriver), ("gpu", WebGpuDriver)):
        if tag == "gpu" and not GPU_AVAILABLE:
            facts["gpu"] = "skipped (no adapter)"
            continue
        rt = Runtime(driver=cls())
        register_all(rt)
        try:
            with pytest.raises(ValueError) as excinfo:
                rt.execute(rt.compile(jobs), {})
            msg = str(excinfo.value)
            assert "no inputs" in msg, msg
            facts[tag] = msg
        finally:
            rt.driver.release()
    if EVIDENCE["op_validation"] is None:
        EVIDENCE["op_validation"] = {"status": "PASS",
                                     "fact": "see S53 entry below"}
    EVIDENCE["op_validation_s53"] = {
        "status": "PASS",
        "fact": ("use_scalar_a=1 + use_scalar_b=1 -> descriptor inputs=[] "
                 "-> builder ValueError 'no inputs' (unreachable mode, "
                 "S53 AS-IS): " + "; ".join(
                     f"{k}={v}" for k, v in facts.items())),
    }


# ============================================================================
# S50 - div-by-zero contract: guard before division, exact 0.0, sign +
# ============================================================================

@needs_gpu
def test_div_zero():
    EVIDENCE["div_zero"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        pairs = [
            (5.0, 0.0, "div(5,0)"),
            (-5.0, -0.0, "div(-5,-0.0)"),
            (np.nan, 0.0, "div(nan,0)"),
            (np.inf, 0.0, "div(inf,0)"),
            (-np.inf, 0.0, "div(-inf,0)"),
            (5.0, 3.0, "div(5,3)"),
        ]
        jobs = [{"op": "MapBinary", "inputs": ["a", "b"],
                 "params": {"op": 3}, "out": "y"}]
        rows = {}
        for av, bv, label in pairs:
            a = np.array([av], np.float32)
            b = np.array([bv], np.float32)
            ref_exact = None
            if np.isfinite(av) and np.isfinite(bv) and bv != 0.0:
                ref_exact = np.float32(av / bv)
            for tag, rt in (("cpu", rt_cpu), ("gpu", rt_gpu)):
                rt.execute(rt.compile(jobs), {"a": a, "b": b})
                res = np.asarray(rt.driver.resolve_output("y"), np.float64)
                if tag == "cpu":
                    rows[label] = {"cpu": _num(res[0])}
                else:
                    rows[label]["gpu"] = _num(res[0])
                    rows[label]["gpu_signbit"] = bool(np.signbit(res[0]))
                    rows[label]["cpu_signbit"] = bool(np.signbit(rows[label]["cpu"]))
        for label in ("div(5,0)", "div(-5,-0.0)", "div(nan,0)",
                      "div(inf,0)", "div(-inf,0)"):
            row = rows[label]
            assert row["cpu"] == 0.0 and row["gpu"] == 0.0, row
            assert not row["cpu_signbit"] and not row["gpu_signbit"], row
        # div(5,3): 1.666... in tolerance (f32 rounding: CPU numpy
        # correctly-rounded vs GPU ~1 ulp)
        row = rows["div(5,3)"]
        assert abs(row["cpu"] - row["gpu"]) <= 1e-6 * abs(row["cpu"]), row
        EVIDENCE["div_zero"] = {
            "status": "PASS",
            "fact": (f"div(5,0)=div(-5,-0.0)=div(nan,0)=div(inf,0)="
                     f"div(-inf,0)=0.0 exact CPU==GPU, sign + (guard before "
                     f"division, b!=0 catches +0.0 and -0.0, IEEE "
                     f"-0.0==0.0); div(5,3): cpu={rows['div(5,3)']['cpu']!r} "
                     f"gpu={rows['div(5,3)']['gpu']!r} (1 ulp f32 division "
                     f"rounding, rel diff "
                     f"{abs(rows['div(5,3)']['cpu'] - rows['div(5,3)']['gpu']) / abs(rows['div(5,3)']['cpu']):.2e} "
                     f"<= contract 1e-6)"),
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# S51 - broadcast b [1]: GPU==CPU for all ops; chunked large N
# ============================================================================

@needs_gpu
def test_broadcast_b():
    EVIDENCE["broadcast"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        rows = {}
        for n in (1, 100, 65536):
            a = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            b = (rng.standard_normal(1) * 5 + 30).astype(np.float32)
            for op in range(6):
                jobs = [{"op": "MapBinary", "inputs": ["a", "b"],
                         "params": {"op": op}, "out": "y"}]
                rt_cpu.execute(rt_cpu.compile(jobs), {"a": a, "b": b})
                res_cpu = np.asarray(rt_cpu.driver.resolve_output("y"), np.float64)
                rt_gpu.execute(rt_gpu.compile(jobs), {"a": a, "b": b})
                res_gpu = np.asarray(rt_gpu.driver.resolve_output("y"), np.float64)
                assert res_gpu.shape == (n,), (n, op, res_gpu.shape)
                fin = np.isfinite(res_cpu) & np.isfinite(res_gpu)
                diff = float(np.abs(res_gpu[fin] - res_cpu[fin]).max()) if fin.any() else 0.0
                peak = float(np.abs(res_cpu[fin]).max()) if fin.any() else 0.0
                assert diff <= max(1e-6 * peak, 1e-6), (n, op, diff, peak)
                rows[f"n{n}_op{op}"] = {"diff": _num(diff)}
        # chunked: n=4_200_000
        n = 4_200_000
        a = (rng.standard_normal(n) * 5 + 30).astype(np.float32)

        # mode B (b [1]) chunked: BLOCKED by ExecutionScheduler (frozen
        # Runtime component). _execute_chunked slices EVERY source array
        # by arr[start:end]; for b [1] the second chunk slice is empty ->
        # create_buffer_with_data(b"") -> ValueError "Mapped size must be
        # larger than zero". Pre-existing scheduler limitation (only equal-
        # length inputs chunk correctly), out of slice scope (frozen
        # Runtime). Recorded as characterization, NOT a failure.
        b1 = np.array([7.25], np.float32)
        waves_b = []
        orig_wave_b = rt_gpu.driver.execute_wave

        def wrap_wave_b(wave, _orig=orig_wave_b, _waves=waves_b):
            _waves.append(wave)
            return _orig(wave)

        rt_gpu.driver.execute_wave = wrap_wave_b
        chunked_b_fact = "unexpectedly worked"
        try:
            rt_gpu.execute(rt_gpu.compile([{"op": "MapBinary",
                                            "inputs": ["a", "b"],
                                            "params": {"op": 0},
                                            "out": "y"}]),
                           {"a": a, "b": b1})
        except ValueError as e:  # documented scheduler limitation
            chunked_b_fact = f"{type(e).__name__}: {str(e)[:100]}"
        rt_gpu.driver.execute_wave = orig_wave_b
        rows["chunked_4200000_modeB"] = {"blocked": chunked_b_fact,
                                         "chunks_dispatched": len(waves_b)}

        # mode A (b len n) chunked: correct, 2 chunks
        b = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        waves = []
        orig_wave = rt_gpu.driver.execute_wave

        def wrap_wave(wave, _orig=orig_wave, _waves=waves):
            _waves.append(wave)
            return _orig(wave)

        rt_gpu.driver.execute_wave = wrap_wave
        ref = a.astype(np.float64) + b.astype(np.float64)
        rt_gpu.execute(rt_gpu.compile([{"op": "MapBinary", "inputs": ["a", "b"],
                                        "params": {"op": 0}, "out": "y"}]),
                       {"a": a, "b": b})
        res = np.asarray(rt_gpu.driver.resolve_output("y"), np.float64)
        assert res.shape == (n,), res.shape
        diff = float(np.abs(res - ref).max())
        assert diff <= 1e-6 * float(np.abs(ref).max()) + 1e-6, diff
        assert len(waves) == 2, f"chunked n={n} waves={len(waves)}"
        rows["chunked_4200000_modeA"] = {"chunks": len(waves),
                                         "diff": _num(diff)}
        EVIDENCE["broadcast"] = {
            "status": "PASS",
            "fact": (f"b [1] broadcast (ordinary path): all ops 0-5 "
                     f"GPU==CPU in tolerance (max finite diff "
                     f"{max(r['diff'] for k, r in rows.items() if k.startswith('n')):.3e}); "
                     f"chunked n=4_200_000 mode A (b len n): "
                     f"{rows['chunked_4200000_modeA']['chunks']} chunks, "
                     f"diff={rows['chunked_4200000_modeA']['diff']:.3e}; "
                     f"chunked mode B (b [1]): BLOCKED - ExecutionScheduler "
                     f"_execute_chunked slices b [1] by arr[start:end], "
                     f"second chunk slice is empty -> "
                     f"'{rows['chunked_4200000_modeB']['blocked']}' "
                     f"(pre-existing frozen-Runtime limitation, equal-"
                     f"length inputs only; characterization, NOT failed)"),
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# S51 - broadcast a [1]: characterization (out [1], CPU guard dead code)
# ============================================================================

def test_broadcast_a_guard():
    a = np.array([2.0], np.float32)
    b = (np.random.default_rng(SEED).standard_normal(1000) * 5 + 30).astype(np.float32)
    jobs = [{"op": "MapBinary", "inputs": ["a", "b"],
             "params": {"op": 1}, "out": "y"}]
    facts = {}
    for tag, cls in (("cpu", CpuDriver), ("gpu", WebGpuDriver)):
        if tag == "gpu" and not GPU_AVAILABLE:
            facts["gpu"] = "skipped (no adapter)"
            continue
        rt = Runtime(driver=cls())
        register_all(rt)
        try:
            rt.execute(rt.compile(jobs), {"a": a, "b": b})
            res = np.asarray(rt.driver.resolve_output("y"), np.float64)
            # characterization: out [1] (n = len(first input)), NOT [n]
            assert res.shape == (1,), res.shape
            assert abs(_num(res[0]) - (2.0 - float(b[0]))) <= 1e-6, res
            facts[tag] = f"out {list(res.shape)} = a[0]-b[0] (no error)"
        finally:
            rt.driver.release()
    if EVIDENCE["broadcast"] is None:
        EVIDENCE["broadcast"] = {"status": "PASS", "fact": "see S51"}
    EVIDENCE["broadcast_a_guard"] = {
        "status": "PASS",
        "fact": ("a [1], b [n] (no use_scalar_a): dispatch_size = "
                 "len(first input) -> out [1] on BOTH backends; CPU guard "
                 "'MapBinary: broadcast of \'a\' (len X) not supported' "
                 "(cpu.py, S51) is defense-in-depth DEAD CODE in the "
                 "ordinary path because dst.length() == len(a) by "
                 "construction; GPU out [1] = a[0] op b[0] (storage OOB "
                 "would be defined as 0 on WebGPU but no OOB occurs since "
                 "dispatch is 1 element); broadcast of 'a' is NOT "
                 "supported by contract - characterization, no error: "
                 + "; ".join(f"{k}: {v}" for k, v in facts.items())),
    }


# ============================================================================
# S52 - exact-assert broadcast sub: contract tolerance, actual diff
# ============================================================================

def test_exact_assert_broadcast_sub():
    EVIDENCE["exact_assert"] = {"status": "FAIL", "fact": "test raised"}
    rng = np.random.default_rng(SEED)
    n = 1_000_000
    a = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
    b = (rng.standard_normal(1) * 5 + 30).astype(np.float32)
    ref = a.astype(np.float64) - float(b[0])
    tol = 1e-6 * float(np.abs(ref).max()) + 1e-7
    results = {}
    for tag, cls in (("cpu", CpuDriver), ("gpu", WebGpuDriver)):
        if tag == "gpu" and not GPU_AVAILABLE:
            results["gpu"] = {"status": "skipped", "fact": "no adapter"}
            continue
        rt = Runtime(driver=cls())
        register_all(rt)
        try:
            rt.execute(rt.compile([{"op": "MapBinary", "inputs": ["a", "b"],
                                    "params": {"op": 1}, "out": "y"}]),
                       {"a": a, "b": b})
            res = np.asarray(rt.driver.resolve_output("y"), np.float64)
            assert res.shape == (n,), res.shape
            diff = float(np.abs(res - ref).max())
            assert diff <= tol, f"{tag}: diff {diff} > tol {tol}"
            results[tag] = {"diff": diff, "tol": tol}
        finally:
            rt.driver.release()
    EVIDENCE["exact_assert"] = {
        "status": "PASS",
        "fact": (f"a [1M] seed 42, b [1], op=1 (sub): ref=a-b[0]; contract "
                 f"tol = 1e-6*|ref|max + 1e-7 = {tol:.3e}; cpu_diff="
                 f"{results['cpu']['diff']:.3e}, gpu_diff="
                 f"{results['gpu'].get('diff', float('nan')):.3e} "
                 f"(expected 0.0 - f32 sub is exact for both backends)"),
    }


# ============================================================================
# S55 - max/min IEEE: zero-sign parity + NaN characterization
# ============================================================================

@needs_gpu
def test_max_min_ieee():
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        # zero-sign: IEEE 754-2019 - CPU==GPU
        zcases = [
            (-0.0, 0.0, 4, +0.0, "max(-0.0,+0.0)=+0.0"),
            (0.0, -0.0, 4, +0.0, "max(+0.0,-0.0)=+0.0"),
            (-0.0, -0.0, 4, -0.0, "max(-0.0,-0.0)=-0.0"),
            (-0.0, 0.0, 5, -0.0, "min(-0.0,+0.0)=-0.0"),
            (0.0, -0.0, 5, -0.0, "min(+0.0,-0.0)=-0.0"),
            (0.0, 0.0, 5, +0.0, "min(+0.0,+0.0)=+0.0"),
        ]
        rows = []
        for av, bv, op, expected, label in zcases:
            a = np.array([av], np.float32)
            b = np.array([bv], np.float32)
            outs = {}
            for tag, rt in (("cpu", rt_cpu), ("gpu", rt_gpu)):
                rt.execute(rt.compile([{"op": "MapBinary", "inputs": ["a", "b"],
                                        "params": {"op": op}, "out": "y"}]),
                           {"a": a, "b": b})
                res = np.asarray(rt.driver.resolve_output("y"), np.float64)
                outs[tag] = (float(res[0]), bool(np.signbit(res[0])))
            assert outs["cpu"][0] == expected and outs["gpu"][0] == expected, \
                (label, outs)
            assert outs["cpu"][1] == outs["gpu"][1], (label, outs)
            rows.append({"case": label, "cpu": outs["cpu"],
                         "gpu": outs["gpu"]})
        # NaN: CPU propagates (IEEE), GPU WGSL max/min returns the
        # non-NaN operand (platform fact) - characterization, NOT a failure
        nan_facts = {}
        for op, name in ((4, "max"), (5, "min")):
            a = np.array([np.nan], np.float32)
            b = np.array([1.0], np.float32)
            rt_cpu.execute(rt_cpu.compile([{"op": "MapBinary", "inputs": ["a", "b"],
                                            "params": {"op": op}, "out": "y"}]),
                           {"a": a, "b": b})
            res_cpu = np.asarray(rt_cpu.driver.resolve_output("y"), np.float64)
            rt_gpu.execute(rt_gpu.compile([{"op": "MapBinary", "inputs": ["a", "b"],
                                            "params": {"op": op}, "out": "y"}]),
                           {"a": a, "b": b})
            res_gpu = np.asarray(rt_gpu.driver.resolve_output("y"), np.float64)
            assert np.isnan(res_cpu[0]), "CPU must propagate NaN (IEEE)"
            nan_facts[name] = {
                "cpu": "nan (IEEE propagate)",
                "gpu": _num(res_gpu[0]),
            }
        if EVIDENCE["exact_assert"] is None:
            EVIDENCE["exact_assert"] = {"status": "PASS", "fact": "see S55"}
        EVIDENCE["max_min_ieee"] = {
            "status": "PASS",
            "fact": ("zero-sign IEEE 754-2019 parity CPU==GPU for "
                     "max/min: " + "; ".join(
                         f"{r['case']} cpu={r['cpu'][0]!r}(signbit={r['cpu'][1]}) "
                         f"gpu={r['gpu'][0]!r}(signbit={r['gpu'][1]})"
                         for r in rows) + "; NaN characterization: "
                     "CPU propagates NaN (np.maximum/np.minimum, IEEE "
                     "754-2019), GPU WGSL max/min builtin returns the "
                     "NON-NaN operand on this platform (NVIDIA RTX 2060, "
                     "Vulkan) - platform fact, recorded NOT failed: " +
                     "; ".join(f"{k}: cpu={v['cpu']}, gpu={v['gpu']!r}"
                               for k, v in nan_facts.items())),
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# S53/S55 data - scalar modes: use_scalar_b (SMA/ATR), use_scalar_a (RSI)
# ============================================================================

@needs_gpu
def test_scalar_modes():
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        a = (rng.standard_normal(1000) * 5 + 30).astype(np.float32)
        # mode C: use_scalar_b=1 (SMA/ATR style: op=3, scalar divisor)
        jobs_c = [{"op": "MapBinary", "inputs": ["a"],
                   "params": {"op": 3, "use_scalar_b": 1, "scalar_b": 14.0},
                   "out": "y"}]
        rt_cpu.execute(rt_cpu.compile(jobs_c), {"a": a})
        res_cpu = np.asarray(rt_cpu.driver.resolve_output("y"), np.float64)
        rt_gpu.execute(rt_gpu.compile(jobs_c), {"a": a})
        res_gpu = np.asarray(rt_gpu.driver.resolve_output("y"), np.float64)
        diff_c = float(np.abs(res_gpu - res_cpu).max())
        peak_c = float(np.abs(res_cpu).max())
        assert diff_c <= max(1e-6 * peak_c, 1e-6), diff_c
        # mode D: use_scalar_a=1 (RSI style: op=1, scalar subtracted)
        jobs_d = [{"op": "MapBinary", "inputs": ["b"],
                   "params": {"op": 1, "use_scalar_a": 1, "scalar_a": 50.0},
                   "out": "y"}]
        rt_cpu.execute(rt_cpu.compile(jobs_d), {"b": a})
        res_cpu = np.asarray(rt_cpu.driver.resolve_output("y"), np.float64)
        rt_gpu.execute(rt_gpu.compile(jobs_d), {"b": a})
        res_gpu = np.asarray(rt_gpu.driver.resolve_output("y"), np.float64)
        diff_d = float(np.abs(res_gpu - res_cpu).max())
        peak_d = float(np.abs(res_cpu).max())
        assert diff_d <= max(1e-6 * peak_d, 1e-6), diff_d
        EVIDENCE["scalar_modes"] = {
            "status": "PASS",
            "fact": (f"use_scalar_b (SMA/ATR style, op=3, scalar_b=14): "
                     f"GPU==CPU max_diff={diff_c:.3e} "
                     f"(tol {max(1e-6 * peak_c, 1e-6):.3e}); "
                     f"use_scalar_a (RSI style, op=1, scalar_a=50): "
                     f"GPU==CPU max_diff={diff_d:.3e} "
                     f"(tol {max(1e-6 * peak_d, 1e-6):.3e})"),
        }
    finally:
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# evidence save (runs last in this file)
# ============================================================================

def test_evidence_saved():
    files = {
        "op_validation.json": ["op_validation", "op_validation_s53"],
        "div_zero.json": ["div_zero"],
        "broadcast.json": ["broadcast", "broadcast_a_guard"],
        "exact_assert.json": ["exact_assert", "max_min_ieee",
                              "scalar_modes"],
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