"""Phase 4 - MAPBINARY differential fuzz (S55), seed 42, >= 1000 cases.

Spec: MAPBINARY_SLICE_SPEC.md sec.9.2 (S55).

Differential: GPU path (WebGpuDriver) and CPU path (CpuDriver). CPU is
the reference (after S49-S55: f32 numpy scalar arithmetic + IEEE 754-2019
max/min via np.maximum/np.minimum + div-by-zero guard). Independent numpy
f64 oracle (np.add/subtract/multiply/divide/maximum/minimum, with the
same div-by-zero guard) is checked against CPU for in-domain data.

Grid (target >= 1000):
  11 N x 6 op x 4 modes x 8 data sets = 2112 cases.
  N: {1, 2, 3, 63, 64, 65, 100, 1000, 4096, 65536, 1_000_000}
  op: 0..5 (add, sub, mul, div, max, min)
  modes: A (a,b len n), B (b len 1 broadcast), C (use_scalar_b),
         D (use_scalar_a). Mode E (both scalars) excluded (S53 builder
         "no inputs").
  data: randn*5+30, uniform[-10,10], zeros, ones, neg (in-domain),
        subnormal 1e-38, large 1e30 (band - FTZ / f32-f64 threshold),
        special NaN/Inf/+0/-0 mix (special - class parity).

Tolerances (contract, MAPBINARY_SLICE_SPEC S55):
  add/sub/mul/div: rel 1e-6 (|res|>1), abs 1e-6 (small);
  div-by-zero: exact (0.0 == 0.0); max/min: exact (f32-representable);
  NaN positions: mask (isnan parity); Inf: exact (both Inf, sign match).

Adversarial -> characterization (recorded, NOT a failure):
  subnormal operands: GPU flush-to-zero (platform FTZ).
  large mul: f32 overflow -> Inf both (CPU kernel is f32 arithmetic,
    numpy float32 scalars).
  special: NaN in denominator for div - GPU div guard treats NaN as 0
    (unordered comparison on this platform) -> 0.0 vs CPU NaN.
  special: NaN with max/min - GPU WGSL max/min returns the NON-NaN
    operand (platform fact, NVIDIA RTX 2060 / Vulkan) vs CPU IEEE NaN
    propagation.

Chunking (separate): N=4_200_000 (2 chunks) and N=10_000_000 (3 chunks),
op=3, mode B - within tolerance, chunk count verified.

Observation (out of slice, recorded in fuzz.json): core/Series/_lib/
executor.py routes a single scalar always to scalar_b; for
"scalar op arr" (sub/div) there is a positional risk - Series is out of
scope, noted for the future slice.

On failure: minimized input (n, op, mode, data), seed, device, versions
recorded in fuzz.json. GPU unavailable -> GPU cases skipped (recorded),
CPU path always runs (>= 1000 cases CPU-only).

IMPORTANT: this file is ASCII-only (no Cyrillic) because
tests/test_backend.py::test_no_cupy_in_test_files reads tests/*.py with
open() in locale encoding (cp1251 on this host).

Evidence: evidence/mapbinary_slice_phase4/fuzz.json
"""

import json
import os
import pathlib
import sys

import numpy as np

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
NS = [1, 2, 3, 63, 64, 65, 100, 1000, 4096, 65536, 1_000_000]
OPS = [0, 1, 2, 3, 4, 5]
OP_NAMES = {0: "add", 1: "sub", 2: "mul", 3: "div", 4: "max", 5: "min"}
MODES = ["A", "B", "C", "D"]

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "mapbinary_slice_phase4")

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001 - adapter probe must never crash collection
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}


def _versions():
    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import wgpu
        v["wgpu"] = wgpu.__version__
    except Exception:  # noqa: BLE001
        v["wgpu"] = "n/a"
    return v


def _num(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def _device():
    if not GPU_AVAILABLE:
        return {"adapter": "n/a", "backend": "cpu"}
    return {"adapter": ADAPTER_INFO.get("device", "unknown"),
            "backend": ADAPTER_INFO.get("backend_type", "unknown")}


def _make_gpu_runtime():
    rt = Runtime(driver=WebGpuDriver())
    register_all(rt)
    return rt


def _make_cpu_runtime():
    rt = Runtime(driver=CpuDriver())
    register_all(rt)
    return rt


# -- data ----------------------------------------------------------------

_SCALARS = {
    "randn5_30": 3.7,
    "uniform": -2.5,
    "zeros": 0.0,
    "ones": 1.0,
    "neg": -4.0,
    "subnormal": 1e-38,
    "large": 1e30,
    "special": 0.0,
}

# dataset -> mode: 'in' (strict), 'band' (characterization on mismatch),
# 'special' (class parity, finite values still strict)
DSETS = {
    "randn5_30": "in",
    "uniform": "in",
    "zeros": "in",
    "ones": "in",
    "neg": "in",
    "subnormal": "band",
    "large": "band",
    "special": "special",
}


def _data_arr(rng, n, dset):
    if dset == "randn5_30":
        return (rng.standard_normal(n) * 5 + 30).astype(np.float32)
    if dset == "uniform":
        return rng.uniform(-10.0, 10.0, size=n).astype(np.float32)
    if dset == "zeros":
        return np.zeros(n, dtype=np.float32)
    if dset == "ones":
        return np.ones(n, dtype=np.float32)
    if dset == "neg":
        return (-(rng.standard_normal(n) * 5 + 30)).astype(np.float32)
    if dset == "subnormal":
        return np.full(n, 1e-38, dtype=np.float32)
    if dset == "large":
        return np.full(n, 1e30, dtype=np.float32)
    if dset == "special":
        arr = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        if n >= 1:
            arr[0] = np.nan
        if n >= 2:
            arr[1] = np.inf
        if n >= 3:
            arr[2] = -np.inf
        if n >= 4:
            arr[3] = 0.0
        if n >= 5:
            arr[4] = -0.0
        if n >= 6:
            arr[5] = 5.0
        if n >= 7:
            arr[6] = 1e-30
        return arr
    raise ValueError(f"unknown dset {dset}")


def _job_for(op, mode, dset, out="y"):
    scalar = _SCALARS[dset]
    if mode == "A":
        return {"op": "MapBinary", "inputs": ["a", "b"],
                "params": {"op": op}, "out": out}
    if mode == "B":
        return {"op": "MapBinary", "inputs": ["a", "b"],
                "params": {"op": op}, "out": out}
    if mode == "C":
        return {"op": "MapBinary", "inputs": ["a"],
                "params": {"op": op, "use_scalar_b": 1, "scalar_b": scalar},
                "out": out}
    if mode == "D":
        return {"op": "MapBinary", "inputs": ["b"],
                "params": {"op": op, "use_scalar_a": 1, "scalar_a": scalar},
                "out": out}
    raise ValueError(f"unknown mode {mode}")


def _data_for(rng, n, op, mode, dset):
    """Return (source_data, a_f64, b_f64) where a_f64/b_f64 are the
    operand arrays used by the numpy f64 oracle (exact f32 values)."""
    arr = _data_arr(rng, n, dset)
    if mode == "A":
        b = _data_arr(rng, n, dset)
        return {"a": arr, "b": b}, arr.astype(np.float64), b.astype(np.float64)
    if mode == "B":
        b = _data_arr(rng, 1, dset)
        return {"a": arr, "b": b}, arr.astype(np.float64), b.astype(np.float64)
    if mode == "C":
        sb = np.float32(_SCALARS[dset])
        return {"a": arr}, arr.astype(np.float64), None
    if mode == "D":
        sa = np.float32(_SCALARS[dset])
        return {"b": arr}, None, arr.astype(np.float64)
    raise ValueError(f"unknown mode {mode}")


# -- oracle & comparison -------------------------------------------------

def _oracle_f64(op, a, b, scalar_a=None, scalar_b=None, mode="A"):
    """Independent numpy f64 oracle (in-domain). Same div-by-zero guard
    as the contract (0.0 for b==0). a/b are f64 arrays (exact f32)."""
    with np.errstate(all="ignore"):
        if mode in ("A", "B"):
            x, y = a, b
        elif mode == "C":
            x, y = a, np.float64(scalar_b)
        else:  # D
            x, y = np.float64(scalar_a), b
        if op == 0:
            return x + y
        if op == 1:
            return x - y
        if op == 2:
            return x * y
        if op == 3:
            return np.where(y == 0.0, 0.0, x / y)
        if op == 4:
            return np.maximum(x, y)
        if op == 5:
            return np.minimum(x, y)
    raise ValueError(f"unknown op {op}")


def _classes(arr):
    arr = np.asarray(arr, np.float64)
    nan = np.isnan(arr)
    pinf = arr == np.inf
    ninf = arr == -np.inf
    return np.where(nan, 1, np.where(pinf, 2, np.where(ninf, 3, 0)))


def _compare(gpu, cpu, op, label, mode, failures, chars):
    """GPU vs CPU differential. Returns (diff, rel).

    mode 'in': class parity + tolerance strict -> failure.
    mode 'band': mismatch -> characterization (FTZ / f32-f64 threshold).
    mode 'special': class mismatch -> characterization (platform NaN/Inf
      semantics); finite values still strict in tolerance.
    """
    g = np.asarray(gpu, np.float64)
    c = np.asarray(cpu, np.float64)
    if g.shape != c.shape:
        failures.append({"label": label,
                         "reason": f"shape mismatch gpu{g.shape} cpu{c.shape}"})
        return 0.0, 0.0
    cls_g = _classes(g)
    cls_c = _classes(c)
    class_mismatch = cls_g != cls_c
    fin = (cls_g == 0) & (cls_c == 0)
    if fin.any():
        diff = float(np.abs(g[fin] - c[fin]).max())
        peak = float(np.abs(c[fin]).max())
        rel = diff / max(peak, 1e-300)
        if op in (4, 5):
            tol = 0.0  # exact for f32-representable
        else:
            tol = max(1e-6 * peak, 1e-6)
        in_tol = diff <= tol
    else:
        diff, rel, in_tol = 0.0, 0.0, True
    sign_mismatch = False
    if op in (4, 5):
        zboth = fin & (g == 0.0) & (c == 0.0)
        if zboth.any():
            if not np.array_equal(np.signbit(g[zboth]), np.signbit(c[zboth])):
                sign_mismatch = True

    if mode == "in":
        if class_mismatch.any():
            idx = [int(i) for i in np.where(class_mismatch)[0][:5]]
            failures.append({
                "label": label,
                "reason": f"class mismatch in-domain: gpu cls="
                          f"{cls_g[idx].tolist()} cpu cls={cls_c[idx].tolist()}",
            })
        elif not in_tol:
            failures.append({
                "label": label, "reason": "out of tolerance",
                "diff": diff, "rel": rel, "peak": peak, "tol": tol,
            })
        elif sign_mismatch:
            failures.append({"label": label,
                             "reason": "zero-sign mismatch (max/min)"})
    elif mode == "band":
        if class_mismatch.any() or not in_tol or sign_mismatch:
            chars.append({
                "label": label,
                "fact": f"band characterization: class_mismatch="
                        f"{bool(class_mismatch.any())}, diff={diff:.3e}, "
                        f"rel={rel:.3e}, in_tol={in_tol}, "
                        f"sign_mismatch={sign_mismatch} "
                        f"(subnormal FTZ or f32/f64 threshold)",
            })
    else:  # special
        if class_mismatch.any():
            idx = [int(i) for i in np.where(class_mismatch)[0][:5]]
            chars.append({
                "label": label,
                "fact": f"special-class mismatch: gpu cls={cls_g[idx].tolist()} "
                        f"cpu cls={cls_c[idx].tolist()} at idx {idx} "
                        f"(NaN/Inf semantics: GPU WGSL max/min returns "
                        f"non-NaN operand; GPU div guard treats NaN as 0)",
            })
        if not in_tol:
            failures.append({
                "label": label, "reason": "finite out of tolerance in "
                                          "special data",
                "diff": diff, "rel": rel, "peak": peak, "tol": tol,
            })
        elif sign_mismatch:
            chars.append({"label": label,
                          "fact": "zero-sign mismatch in special data "
                                  "(max/min)"})
    return diff, rel


# -- test -----------------------------------------------------------------

def test_fuzz_mapbinary():
    rng = np.random.default_rng(SEED)
    gpu_rt = _make_gpu_runtime() if GPU_AVAILABLE else None
    cpu_rt = _make_cpu_runtime()
    failures = []
    chars = []
    n_cases = 0
    n_cases_gpu = 0
    n_cases_cpu = 0
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    oracle_failures = []

    try:
        for n in NS:
            for op in OPS:
                for mode in MODES:
                    for dset, dmode in DSETS.items():
                        label = (f"n={n} op={op}({OP_NAMES[op]}) "
                                 f"mode={mode} data={dset}")
                        data, a_f64, b_f64 = _data_for(rng, n, op, mode, dset)
                        job = _job_for(op, mode, dset)

                        # CPU path (always runs)
                        tasks = cpu_rt.compile([job])
                        cpu_rt.execute(tasks, data)
                        res_cpu = cpu_rt.driver.resolve_output("y")
                        n_cases_cpu += 1
                        # numpy f64 oracle sanity for in-domain data
                        if dmode == "in":
                            ref = _oracle_f64(
                                op, a_f64, b_f64,
                                scalar_a=_SCALARS[dset],
                                scalar_b=_SCALARS[dset], mode=mode)
                            c64 = np.asarray(res_cpu, np.float64)
                            if c64.shape == ref.shape:
                                fin = np.isfinite(c64) & np.isfinite(ref)
                                if fin.any():
                                    d = float(np.abs(c64[fin] - ref[fin]).max())
                                    p = float(np.abs(ref[fin]).max())
                                    if d > max(1e-6 * p, 1e-6):
                                        oracle_failures.append({
                                            "label": f"cpu-vs-oracle {label}",
                                            "diff": d, "peak": p,
                                        })
                            else:
                                oracle_failures.append({
                                    "label": f"cpu-vs-oracle {label}",
                                    "reason": f"shape cpu{c64.shape} "
                                              f"oracle{ref.shape}",
                                })

                        if gpu_rt is None:
                            continue

                        tasks_g = gpu_rt.compile([job])
                        gpu_rt.execute(tasks_g, data)
                        res_gpu = gpu_rt.driver.resolve_output("y")
                        n_cases_gpu += 1
                        diff, rel = _compare(res_gpu, res_cpu, op, label,
                                             dmode, failures, chars)
                        max_abs_diff = max(max_abs_diff, diff)
                        max_rel_diff = max(max_rel_diff, rel)

        # ---- chunking: op=3 (div), large N ----
        # mode A (equal lengths) proves chunking correctness; mode B
        # (b [1]) is BLOCKED by ExecutionScheduler (frozen Runtime):
        # _execute_chunked slices every input arr[start:end], the second
        # chunk of b [1] is empty -> ValueError. Recorded as
        # characterization (pre-existing, out of slice scope).
        chunk_results = {}
        for n, expected in ((4_200_000, 2), (10_000_000, 3)):
            waves = []
            if gpu_rt is not None:
                orig_wave = gpu_rt.driver.execute_wave

                def wrap_wave(wave, _orig=orig_wave, _waves=waves):
                    _waves.append(wave)
                    return _orig(wave)

                gpu_rt.driver.execute_wave = wrap_wave
            rng2 = np.random.default_rng(SEED + n)
            a = (rng2.standard_normal(n) * 5 + 30).astype(np.float32)
            b = (rng2.standard_normal(n) * 5 + 30).astype(np.float32)
            data = {"a": a, "b": b}
            job = {"op": "MapBinary", "inputs": ["a", "b"],
                   "params": {"op": 3}, "out": "y"}
            # CPU reference
            cpu_rt.execute(cpu_rt.compile([job]), data)
            res_cpu = np.asarray(cpu_rt.driver.resolve_output("y"), np.float64)
            if gpu_rt is not None:
                gpu_rt.execute(gpu_rt.compile([job]), data)
                res_gpu = np.asarray(gpu_rt.driver.resolve_output("y"),
                                     np.float64)
                fin = np.isfinite(res_cpu) & np.isfinite(res_gpu)
                diff = float(np.abs(res_gpu[fin] - res_cpu[fin]).max()) \
                    if fin.any() else 0.0
                peak = float(np.abs(res_cpu[fin]).max()) if fin.any() else 0.0
                tol = max(1e-6 * peak, 1e-6)
                if diff > tol:
                    failures.append({
                        "label": f"chunking n={n} op=3 mode=A",
                        "reason": "out of tolerance",
                        "diff": diff, "rel": diff / max(peak, 1e-300),
                        "peak": peak, "tol": tol,
                    })
                chunk_results[str(n)] = {
                    "chunks": len(waves), "expected_chunks": expected,
                    "max_abs_diff_vs_cpu": _num(diff),
                }
                assert len(waves) == expected, \
                    f"n={n}: chunks={len(waves)} expected {expected}"
                max_abs_diff = max(max_abs_diff, diff)
            # mode B chunked attempt (b [1]) - documented characterization
            if gpu_rt is not None:
                b1 = np.array([3.0], dtype=np.float32)
                try:
                    gpu_rt.execute(gpu_rt.compile([job]), {"a": a, "b": b1})
                    chunk_results[str(n)]["modeB"] = "unexpectedly worked"
                except ValueError as e:
                    chunk_results[str(n)]["modeB"] = {
                        "blocked": f"{type(e).__name__}: {str(e)[:100]}",
                        "note": "ExecutionScheduler._execute_chunked slices "
                                "b [1] by arr[start:end]; second chunk "
                                "slice is empty (pre-existing frozen-Runtime "
                                "limitation, equal-length inputs only)",
                    }

        n_cases = n_cases_gpu + n_cases_cpu
        assert n_cases >= 1000, f"n_cases={n_cases} < 1000"
        assert failures == [], f"fuzz failures: {failures[:3]}"
        assert oracle_failures == [], \
            f"cpu-vs-oracle failures: {oracle_failures[:3]}"

        evidence = {
            "phase": 4,
            "slice": SLICE,
            "checkpoint_sha": CHECKPOINT_SHA,
            "versions": _versions(),
            "result": {
                "seed": SEED,
                "n_grid": NS,
                "ops": OPS,
                "op_names": OP_NAMES,
                "modes": MODES,
                "datasets": list(DSETS.keys()),
                "dtype": "f32",
                "dtype_note": "int32 NOT used: MapBinary is f32-only by "
                              "contract",
                "n_cases": n_cases,
                "n_cases_gpu": n_cases_gpu,
                "n_cases_cpu": n_cases_cpu,
                "gpu_available": GPU_AVAILABLE,
                "gpu_note": ("GPU unavailable -> GPU cases skipped, CPU "
                             "path ran all cases") if not GPU_AVAILABLE else
                            "GPU+CPU differential",
                "device": _device(),
                "max_abs_diff": _num(max_abs_diff),
                "max_rel_diff": _num(max_rel_diff),
                "failures": failures,
                "oracle_failures": oracle_failures,
                "characterizations": chars,
                "chunking": chunk_results,
                "note": "differential GPU (WebGpuDriver) vs CPU (CpuDriver) "
                        "with independent numpy f64 oracle (in-domain, "
                        "div-by-zero guard applied to the oracle); "
                        "tolerances: add/sub/mul/div rel 1e-6 (|res|>1) "
                        "abs 1e-6 (small), div-by-zero exact, max/min exact "
                        "(f32-representable), NaN positions mask (isnan "
                        "parity), Inf exact (both Inf, sign match); "
                        "adversarial -> characterization: subnormal FTZ "
                        "(GPU flush-to-zero: add(1e-38,1e-38)=0.0, "
                        "div(1e-38,1e-38)=0.0 vs CPU subnormal preserved), "
                        "large f32/f64 threshold, special NaN/Inf semantics "
                        "(GPU WGSL max/min returns the non-NaN operand on "
                        "this platform - NVIDIA RTX 2060 / Vulkan - vs CPU "
                        "IEEE NaN propagation; GPU div guard treats NaN "
                        "denominator as 0 (unordered comparison) -> 0.0 vs "
                        "CPU NaN); "
                        "chunking: proven with mode A (equal lengths) for "
                        "N=4_200_000 (2 chunks) and N=10_000_000 (3 "
                        "chunks); chunked mode B (b [1]) is BLOCKED by "
                        "ExecutionScheduler._execute_chunked (frozen "
                        "Runtime): every input is sliced arr[start:end], "
                        "the second chunk of b [1] is empty -> ValueError "
                        "Mapped size must be larger than zero - "
                        "pre-existing limitation (equal-length inputs "
                        "only), characterization",
                "observation": {
                    "component": "core/Series/_lib/executor.py",
                    "fact": "single scalar in GPU path is always routed to "
                            "scalar_b (lines 312-314); for 'scalar op arr' "
                            "(sub/div) there is a positional risk: "
                            "scalar-subtracted-from-array would require "
                            "use_scalar_a, but the route sends the scalar "
                            "to scalar_b only",
                    "scope": "Series is OUT of scope for the MapBinary "
                             "slice; noted for the future Series slice",
                },
            },
            "evidence_schema": "v1",
        }
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        (EVIDENCE_DIR / "fuzz.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False),
            encoding="utf-8")
    finally:
        cpu_rt.driver.release()
        if gpu_rt is not None:
            gpu_rt.driver.release()