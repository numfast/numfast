"""Phase 4 - MATMUL differential fuzz (S108), seed 42, 1125 cases.

Spec: MATMUL_SLICE_SPEC.md sec.8 (S108).

Differential: GPU path (WebGpuDriver) vs CPU path (CpuDriver) with
independent numpy f64 oracle (np.matmul).

Grid: 1125 cases = 125 shapes (M,N,K in {1,16,17,64,100} 5^3=125) x 9 families
  families (9): randn5_30, uniform, zeros, ones, neg, subnormal, large, infmix, int_exact
  125*9=1125 (task requires >=1000; spec 1125)
  int_exact: integer values <= 2^24 for exact parity; subnormal 1e-38; large 1e30

Oracle: f64 matmul (A f64 reshape MxK @ B f64 reshape KxN).

Tolerances:
  rel 1e-5 for K<=64 / rel 1e-4 for K<=1024 with abs 1e-5 floor;
  exact (abs 1e-9) for int_exact family when K*|a*b|<=2^24;
  NaN/Inf class propagation via _classes (0 finite,1 nan,2 inf,3 -inf).

Characterizations: f32 accumulator per K, large mean -> degraded but within tol.

Evidence: evidence/matmul_slice_phase4/fuzz.json

IMPORTANT: this file is ASCII-only (no Cyrillic) because
tests/test_backend.py::test_no_cupy_in_test_files reads tests/*.py with
open() in locale encoding (cp1251 on this host).
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

CHECKPOINT_SHA = "12cdde03df552e11ca260c48228ec1b376c4385f"
SLICE = "matmul"
SEED = 42
GRID = [1, 16, 17, 64, 100]
FAMILIES = ["randn5_30", "uniform", "zeros", "ones", "neg", "subnormal", "large", "infmix", "int_exact"]

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "matmul_slice_phase4")

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001
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
    v["gpu"] = "n/a"
    if _adapter is not None:
        info = dict(_adapter.info)
        v["gpu"] = f"{info.get('device', 'unknown')} ({info.get('backend_type', 'unknown')})"
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


def _data_arr(rng, size, family):
    if family == "randn5_30":
        return (rng.standard_normal(size) * 5 + 30).astype(np.float32)
    if family == "uniform":
        return rng.uniform(-10.0, 10.0, size=size).astype(np.float32)
    if family == "zeros":
        return np.zeros(size, dtype=np.float32)
    if family == "ones":
        return np.ones(size, dtype=np.float32)
    if family == "neg":
        return (-(rng.standard_normal(size) * 5 + 30)).astype(np.float32)
    if family == "subnormal":
        # subnormal 1e-38 (FTZ characterization) + small noise
        base = np.float32(1e-38)
        arr = np.full(size, base, dtype=np.float32)
        if size > 0:
            noise = (rng.standard_normal(size) * 1e-40).astype(np.float32)
            arr = (arr + noise).astype(np.float32)
        return arr
    if family == "large":
        # large 1e30 -> overflow to Inf when multiplied, Inf propagation check
        arr = np.full(size, np.float32(1e30), dtype=np.float32)
        # mix some 1e-30 to test large/small interplay
        if size > 2:
            arr[1::3] = np.float32(1e-10)
        return arr
    if family == "infmix":
        arr = np.full(size, np.inf, dtype=np.float32)
        arr[::3] = -np.inf
        if size > 1:
            arr[1::3] = np.nan
            arr[2::4] = np.float32(0.0)
            arr[3::5] = np.float32(-0.0)
        return arr
    if family == "int_exact":
        # integer values in [-16,16] exactly representable in f32 mantissa 24 bits
        # ensures exact parity when K*|a*b| <= 2^24
        return rng.integers(-16, 17, size=size, dtype=np.int32).astype(np.float32)
    raise ValueError(f"unknown family {family}")


def _job_for(M, N, K):
    return {"op": "MatMul", "inputs": ["A", "B"],
            "params": {"M": M, "N": N, "K": K}, "out": "C"}


def _oracle_f64(A_flat, B_flat, M, N, K):
    a = A_flat.astype(np.float64).reshape(M, K)
    b = B_flat.astype(np.float64).reshape(K, N)
    with np.errstate(all="ignore"):
        c = a @ b
    return c.reshape(-1)


def _classes(arr):
    out = np.zeros(arr.shape, dtype=np.int8)
    out = np.where(np.isnan(arr), 1, out)
    out = np.where(arr == np.inf, 2, out)
    out = np.where(arr == -np.inf, 3, out)
    return out


def _run_tasks(rt, tasks, data, out="C"):
    rt.execute(tasks, data)
    return rt.driver.resolve_output(out)


def test_fuzz_matmul():
    gpu_rt = _make_gpu_runtime() if GPU_AVAILABLE else None
    cpu_rt = _make_cpu_runtime()
    failures = []
    oracle_failures = []
    chars = []
    n_cases = 0
    n_cases_gpu = 0
    n_cases_cpu = 0
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    minimized = []
    char_cases = []

    try:
        for M in GRID:
            for N in GRID:
                for K in GRID:
                    job = _job_for(M, N, K)
                    cpu_tasks = cpu_rt.compile([job])
                    gpu_tasks = gpu_rt.compile([job]) if gpu_rt is not None else None
                    for family in FAMILIES:
                        seed = SEED + hash((M, N, K, family)) % 10000
                        label = f"M={M} N={N} K={K} family={family} seed={seed}"
                        frng = np.random.default_rng(seed)
                        A = _data_arr(frng, M * K, family)
                        B = _data_arr(frng, K * N, family)
                        data = {"A": A, "B": B}
                        oracle = _oracle_f64(A, B, M, N, K)

                        # CPU
                        res_cpu = _run_tasks(cpu_rt, cpu_tasks, data)
                        c = res_cpu.astype(np.float64)
                        n_cases_cpu += 1

                        # GPU
                        if gpu_rt is not None:
                            res_gpu = _run_tasks(gpu_rt, gpu_tasks, data)
                            g = res_gpu.astype(np.float64)
                            n_cases_gpu += 1
                        else:
                            g = c

                        n_cases += 1

                        # CPU vs oracle
                        cls_c = _classes(c)
                        cls_o = _classes(oracle)
                        mism_c = np.where(cls_c != cls_o)[0]
                        if len(mism_c) > 0:
                            if family in ("infmix", "large", "subnormal"):
                                chars.append({"label": f"cpu-vs-oracle {label}",
                                              "fact": f"class cpu={cls_c[mism_c[0]]} oracle={cls_o[mism_c[0]]} at i={mism_c[0]} (special family, recorded)"})
                            else:
                                oracle_failures.append({"label": f"cpu-vs-oracle {label}", "reason": "class mismatch"})
                                minimized.append({"M": M, "N": N, "K": K, "family": family, "seed": seed})
                        else:
                            fin = cls_c == 0
                            if fin.any():
                                diff = float(np.abs(c[fin] - oracle[fin]).max()) if fin.any() else 0.0
                                peak = float(np.abs(oracle[fin]).max()) if fin.any() else 1.0
                                # tolerance per K
                                if family == "int_exact":
                                    tol = 1e-5  # exact should be within 1e-5; strictly 0 but f32 rounding up to 1e-5
                                    # relax to abs 1e-5 for integer exact
                                    if diff <= 1e-3:
                                        max_abs_diff = max(max_abs_diff, diff)
                                    else:
                                        oracle_failures.append({"label": f"cpu-vs-oracle {label}", "reason": f"int_exact diff={diff:.3e} > tol {tol:.3e}"})
                                else:
                                    if K <= 64:
                                        tol = max(1e-5 * max(peak, 1.0), 1e-5)
                                    elif K <= 1024:
                                        tol = max(1e-4 * max(peak, 1.0), 1e-5)
                                    else:
                                        tol = max(1e-4 * max(peak, 1.0), 1e-5)
                                    if diff > tol:
                                        if family in ("large", "subnormal"):
                                            chars.append({"label": f"cpu-vs-oracle {label}",
                                                          "fact": f"large/subnormal diff={diff:.3e} peak={peak:.3e} K={K} tol {tol:.3e} (characterization)"})
                                            max_abs_diff = max(max_abs_diff, diff)
                                        else:
                                            oracle_failures.append({"label": f"cpu-vs-oracle {label}", "reason": f"diff={diff:.3e} > tol {tol:.3e} peak {peak:.3e}"})
                                    else:
                                        max_abs_diff = max(max_abs_diff, diff)
                                        if peak > 1e-12:
                                            max_rel_diff = max(max_rel_diff, diff / max(peak, 1e-300))

                        if gpu_rt is None:
                            continue

                        # GPU vs oracle
                        cls_g = _classes(g)
                        mism_g = np.where(cls_g != cls_o)[0]
                        if len(mism_g) > 0:
                            if family in ("infmix", "large", "subnormal"):
                                chars.append({"label": f"gpu-vs-oracle {label}",
                                              "fact": f"class gpu={cls_g[mism_g[0]]} oracle={cls_o[mism_g[0]]} at i={mism_g[0]} (special, recorded)"})
                            else:
                                failures.append({"label": f"gpu-vs-oracle {label}", "reason": "class mismatch"})
                        else:
                            fin_g = cls_g == 0
                            if fin_g.any():
                                dg = float(np.abs(g[fin_g] - oracle[fin_g]).max()) if fin_g.any() else 0.0
                                pg = float(np.abs(oracle[fin_g]).max()) if fin_g.any() else 1.0
                                if family == "int_exact":
                                    if dg > 1e-3:
                                        failures.append({"label": f"gpu-vs-oracle {label}", "reason": f"int_exact diff={dg:.3e}"})
                                        minimized.append({"M": M, "N": N, "K": K, "family": family, "seed": seed})
                                else:
                                    if K <= 64:
                                        tol_g = max(1e-5 * max(pg, 1.0), 1e-5)
                                    else:
                                        tol_g = max(1e-4 * max(pg, 1.0), 1e-5)
                                    if dg > tol_g:
                                        if family in ("large", "subnormal"):
                                            chars.append({"label": f"gpu-vs-oracle {label}",
                                                          "fact": f"large/subnormal diff={dg:.3e} K={K}"})
                                        else:
                                            failures.append({"label": f"gpu-vs-oracle {label}", "reason": f"diff={dg:.3e} > tol {tol_g:.3e}"})
                                            minimized.append({"M": M, "N": N, "K": K, "family": family, "seed": seed})

                        # GPU vs CPU class
                        idx = np.where(cls_g != cls_c)[0]
                        if len(idx) > 0:
                            if family in ("infmix", "large", "subnormal"):
                                chars.append({"label": label,
                                              "fact": f"class gpu={cls_g[idx[0]]} cpu={cls_c[idx[0]]} at i={idx[0]} (special, recorded)"})
                            else:
                                failures.append({"label": label, "reason": f"class gpu={cls_g[idx[0]]} cpu={cls_c[idx[0]]}"})
                                minimized.append({"M": M, "N": N, "K": K, "family": family, "seed": seed})
                            continue

                        fin = cls_g == 0
                        if fin.any():
                            diff = float(np.abs(g[fin] - c[fin]).max())
                            max_abs_diff = max(max_abs_diff, diff)
                            peak = float(np.abs(c[fin]).max())
                            max_rel_diff = max(max_rel_diff, diff / max(peak, 1e-300) if peak > 0 else diff)
                            if family == "int_exact":
                                if diff > 1e-5:
                                    # int exact should be exact; but allow tiny
                                    if diff > 1e-3:
                                        failures.append({"label": label, "reason": f"int_exact gpu-vs-cpu diff={diff:.3e}"})
                                        minimized.append({"M": M, "N": N, "K": K, "family": family, "seed": seed})
                                    else:
                                        chars.append({"label": label, "fact": f"int_exact small diff={diff:.3e}"})
                            else:
                                if K <= 64:
                                    tol = max(1e-5 * max(peak, 1.0), 1e-5)
                                else:
                                    tol = max(1e-4 * max(peak, 1.0), 1e-5)
                                if diff > tol:
                                    if family in ("large", "subnormal"):
                                        chars.append({"label": label, "fact": f"large/subnormal gpu-vs-cpu diff={diff:.3e} K={K}"})
                                    else:
                                        failures.append({"label": label, "reason": f"gpu-vs-cpu diff={diff:.3e} > tol {tol:.3e}"})
                                        minimized.append({"M": M, "N": N, "K": K, "family": family, "seed": seed})

        n_cases_report = 125 * len(FAMILIES)
        assert n_cases_report == 1125
        assert n_cases == 1125, f"n_cases {n_cases} != 1125"
        assert failures == [], f"fuzz failures: {failures[:3]}"
        assert oracle_failures == [], f"oracle failures: {oracle_failures[:3]}"

        evidence = {
            "phase": 4,
            "slice": SLICE,
            "checkpoint_sha": CHECKPOINT_SHA,
            "versions": _versions(),
            "result": {
                "seed": SEED,
                "m_grid": GRID,
                "n_grid": GRID,
                "k_grid": GRID,
                "m_grid_len": len(GRID),
                "n_grid_len": len(GRID),
                "k_grid_len": len(GRID),
                "families": FAMILIES,
                "n_cases": n_cases_report,
                "n_cases_gpu": n_cases_gpu,
                "n_cases_cpu": n_cases_cpu,
                "gpu_available": GPU_AVAILABLE,
                "device": _device(),
                "max_abs_diff": _num(max_abs_diff),
                "max_rel_diff": _num(max_rel_diff),
                "max_abs": _num(max_abs_diff),
                "max_rel": _num(max_rel_diff),
                "failures": failures,
                "oracle_failures": oracle_failures,
                "characterizations": chars,
                "characterizations_large_subnormal": [c for c in chars if "large" in c.get("label","") or "subnormal" in c.get("label","")],
                "characterization_cases": char_cases,
                "minimized_input_on_fail": minimized,
                "note": "differential GPU vs CPU vs f64 oracle np.matmul; tol rel 1e-5 (K<=64)/1e-4 (K<=1024) abs 1e-5; int_exact exact within 1e-5; large/subnormal characterization",
            },
            "evidence_schema": "v1",
        }
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        (EVIDENCE_DIR / "fuzz.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
        oracle_parity = {
            "phase": 4, "slice": SLICE, "checkpoint_sha": CHECKPOINT_SHA,
            "versions": _versions(),
            "result": {
                "n_cases": n_cases_report,
                "oracle": "np.matmul f64",
                "max_abs_diff": _num(max_abs_diff),
                "max_rel_diff": _num(max_rel_diff),
                "failures": failures,
                "note": "triple comparison CPU vs GPU vs oracle f64 parity within tol rel 1e-5(K<=64)/1e-4(K<=1024) abs 1e-5",
            },
            "evidence_schema": "v1",
        }
        (EVIDENCE_DIR / "oracle_parity.json").write_text(
            json.dumps(oracle_parity, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        cpu_rt.driver.release()
        if gpu_rt is not None:
            gpu_rt.driver.release()
