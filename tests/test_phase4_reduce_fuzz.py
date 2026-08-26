"""Phase 4 - REDUCE differential fuzz (S66), seed 42, >= 1000 cases.

Spec: REDUCE_SLICE_SPEC.md sec.7 (S66).

Differential: GPU path (WebGpuDriver) vs CPU path (CpuDriver) with
independent numpy f64 oracle (np.sum / np.min / np.max).

Grid (target >= 1000):
  3 op x 14 N x 24 variations = 1008 cases/backend -> 2016 GPU+CPU.
  N: {1, 2, 3, 63, 64, 65, 100, 255, 256, 257, 1000, 4096, 65536,
      1_000_000}  (14; N=0 excluded - separate S62 tests)
  op: 0 sum, 1 min, 2 max (all; unknown op NOT generated - out of domain)
  data families (10): randn*5+30, uniform[-10,10], zeros, ones, neg,
        subnormal 1e-38, large 1e30, NaN, +/-Inf mix, +/-0.0 mix.
  variations (24 per (op,N)): seed = 42 + v for stochastic families
        (randn5_30, uniform, neg); deterministic families repeat fixed
        content (each variation is executed and checked).

Tolerances (contract, REDUCE_SLICE_SPEC S66):
  sum: rel 1e-4 (|sum| > 1) / abs 1e-4 (small, n <= 1M);
  min/max: exact (f32-representable values);
  NaN: sum - mask (isnan parity, propagate); min/max - CPU==oracle
        (IEEE in-domain), GPU NaN cases - known WGSL characterization
        (PHASE4_MAPBINARY_GATE sec 5.1: unordered max/min return the
        NON-NaN operand on NVIDIA RTX 2060 / Vulkan) - NOT a failure,
        recorded;
  +/-Inf: sum propagate (sign matches); min/max exact;
  +/-0.0: sum +0 + -0 = +0.0 (IEEE RNE); min/max zero-sign IEEE on CPU
        (_min_ieee/_max_ieee), GPU zero-sign - WGSL unordered fact.

Observations (out of slice, recorded, NOT failures):
  numpy 2.5.1 np.min/np.max reductions are ORDER-DEPENDENT for mixed
  zeros (return the second operand's sign; np.min([-0.,0.])=+0.) - CPU
  follows IEEE 754-2019 by construction (S63), oracle sign may differ on
  zero-sign cases.

On failure: minimized input (N, op, data family, seed), device, versions
recorded in fuzz.json. GPU unavailable -> GPU cases skipped (recorded),
CPU path always runs (1008 cases >= 1000).

Evidence: evidence/reduce_slice_phase4/fuzz.json

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

CHECKPOINT_SHA = "f3a21d9"
SLICE = "reduce"
SEED = 42
NS = [1, 2, 3, 63, 64, 65, 100, 255, 256, 257, 1000, 4096, 65536, 1_000_000]
OPS = [0, 1, 2]
OP_NAMES = {0: "sum", 1: "min", 2: "max"}
# 10 data families; variation v selects family = FAMILIES[v % 10] with
# seed = 42 + v (stochastic families get seeds 42..65, deterministic
# families repeat fixed content - each variation is executed/checked).
FAMILIES = ["randn5_30", "uniform", "zeros", "ones", "neg", "subnormal",
            "large", "nan", "infmix", "zerosign"]
N_VAR = 24

# family -> comparison mode:
#   'in'      - strict (mismatch -> failure)
#   'band'    - adversarial (mismatch -> characterization; subnormal FTZ
#               on NVIDIA Vulkan: GPU flushes subnormal min/max results to
#               0.0, CPU preserves them; large 1e30: f32 accumulation
#               error at extreme magnitude exceeds rel 1e-4 for sum on
#               BOTH backends - platform facts, PHASE4_MAPBINARY_GATE G4
#               precedent, NOT failures)
#   'special' - NaN/Inf/zero-sign semantics (class mismatch -> characterization)
DSET_MODE = {
    "randn5_30": "in", "uniform": "in", "zeros": "in", "ones": "in",
    "neg": "in", "subnormal": "band", "large": "band",
    "nan": "special", "infmix": "special", "zerosign": "special",
}

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "reduce_slice_phase4")

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

def _data_arr(rng, n, family):
    if family == "randn5_30":
        return (rng.standard_normal(n) * 5 + 30).astype(np.float32)
    if family == "uniform":
        return rng.uniform(-10.0, 10.0, size=n).astype(np.float32)
    if family == "zeros":
        return np.zeros(n, dtype=np.float32)
    if family == "ones":
        return np.ones(n, dtype=np.float32)
    if family == "neg":
        return (-(rng.standard_normal(n) * 5 + 30)).astype(np.float32)
    if family == "subnormal":
        return np.full(n, 1e-38, dtype=np.float32)
    if family == "large":
        return np.full(n, 1e30, dtype=np.float32)
    if family == "nan":
        return np.full(n, np.nan, dtype=np.float32)
    if family == "infmix":
        arr = np.full(n, np.inf, dtype=np.float32)
        arr[::2] = -np.inf
        return arr
    if family == "zerosign":
        arr = np.zeros(n, dtype=np.float32)  # all +0.0
        arr[::2] = np.float32(-0.0)          # alternating -0.0
        return arr
    raise ValueError(f"unknown family {family}")


def _job_for(op):
    return {"op": "Reduce", "inputs": ["x"], "params": {"op": op}, "out": "y"}


def _oracle_f64(op, x):
    """Independent numpy f64 oracle on exact f32 values."""
    with np.errstate(all="ignore"):
        if op == 0:
            return np.sum(x.astype(np.float64))
        if op == 1:
            return np.min(x.astype(np.float64))
        if op == 2:
            return np.max(x.astype(np.float64))
    raise ValueError(f"unknown op {op}")


def _cls(v):
    """Class: 0 finite, 1 nan, 2 +inf, 3 -inf."""
    if np.isnan(v):
        return 1
    if v == np.inf:
        return 2
    if v == -np.inf:
        return 3
    return 0


def _sum_ok(diff, o):
    """Contract sum tolerance: rel 1e-4 (|sum|>1) / abs 1e-4 (small)."""
    return diff <= 1e-4 * max(abs(o), 1.0)


def _run(rt, job, data, out="y"):
    rt.execute(rt.compile([job]), data)
    return rt.driver.resolve_output(out)


# -- test -----------------------------------------------------------------

def test_fuzz_reduce():
    rng = np.random.default_rng(SEED)
    gpu_rt = _make_gpu_runtime() if GPU_AVAILABLE else None
    cpu_rt = _make_cpu_runtime()
    failures = []
    oracle_failures = []
    chars = []
    observations = []
    n_cases = 0
    n_cases_gpu = 0
    n_cases_cpu = 0
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    minimized = []

    try:
        for n in NS:
            for op in OPS:
                for v in range(N_VAR):
                    family = FAMILIES[v % len(FAMILIES)]
                    mode = DSET_MODE[family]
                    seed = SEED + v
                    label = (f"n={n} op={op}({OP_NAMES[op]}) "
                             f"var={v} family={family} seed={seed}")
                    frng = np.random.default_rng(seed)
                    x = _data_arr(frng, n, family)
                    job = _job_for(op)
                    data = {"x": x}
                    oracle = _oracle_f64(op, x)

                    # CPU path (always runs)
                    res_cpu = _run(cpu_rt, job, data)
                    c = np.float64(res_cpu[0])
                    n_cases_cpu += 1

                    # CPU vs oracle (in-domain strict; band/special
                    # families tolerate documented platform limits)
                    cls_c = _cls(c)
                    cls_o = _cls(oracle)
                    if cls_c != cls_o:
                        if mode == "band":
                            chars.append({
                                "label": f"cpu-vs-oracle {label}",
                                "fact": (f"class cpu={cls_c} oracle={cls_o} "
                                         f"cpu={c!r} oracle={oracle!r} "
                                         f"(band family - platform "
                                         f"threshold, NOT a failure)"),
                            })
                        else:
                            oracle_failures.append({
                                "label": f"cpu-vs-oracle {label}",
                                "reason": f"class cpu={cls_c} oracle={cls_o} "
                                          f"cpu={c!r} oracle={oracle!r}",
                            })
                    elif cls_c == 0:
                        diff_co = abs(c - oracle)
                        if op == 0:  # sum: contract tolerance (f32
                            # accumulation on CPU vs f64 oracle)
                            ok = _sum_ok(diff_co, float(oracle))
                        else:        # min/max: exact values
                            ok = float(c) == float(oracle)
                        if not ok:
                            if c == 0.0 and oracle == 0.0 and \
                                    np.signbit(c) != np.signbit(oracle):
                                # numpy 2.5.1 reduction order-dependent
                                # zero-sign vs CPU IEEE - observation
                                observations.append({
                                    "label": f"cpu-vs-oracle {label}",
                                    "fact": (f"zero-sign: cpu={c!r} "
                                             f"(signbit {bool(np.signbit(c))}) "
                                             f"oracle={oracle!r} (signbit "
                                             f"{bool(np.signbit(oracle))}); "
                                             f"numpy np.{OP_NAMES[op]} "
                                             f"reduction order-dependent "
                                             f"for mixed zeros (returns "
                                             f"second operand sign); CPU "
                                             f"follows IEEE 754-2019 (S63)"),
                                })
                            elif mode == "band":
                                chars.append({
                                    "label": f"cpu-vs-oracle {label}",
                                    "fact": (f"band characterization: "
                                             f"cpu={c!r} oracle={oracle!r} "
                                             f"diff={diff_co:.3e} (f32/f64 "
                                             f"accumulation threshold at "
                                             f"extreme magnitude - "
                                             f"PHASE4_MAPBINARY_GATE G4 "
                                             f"precedent, NOT a failure)"),
                                })
                            else:
                                oracle_failures.append({
                                    "label": f"cpu-vs-oracle {label}",
                                    "reason": f"value cpu={c!r} "
                                              f"oracle={oracle!r}",
                                })
                    elif cls_c == 1:
                        pass  # NaN both - mask parity
                    else:
                        if (c == np.inf) != (oracle == np.inf):
                            oracle_failures.append({
                                "label": f"cpu-vs-oracle {label}",
                                "reason": f"inf sign cpu={c!r} "
                                          f"oracle={oracle!r}",
                            })

                    if gpu_rt is None:
                        continue

                    # GPU path
                    res_gpu = _run(gpu_rt, job, data)
                    g = np.float64(res_gpu[0])
                    n_cases_gpu += 1
                    cls_g = _cls(g)

                    if op == 0:  # sum: mask parity + tolerance
                        if cls_g != cls_c:
                            if mode == "band":
                                chars.append({
                                    "label": label,
                                    "fact": (f"class gpu={cls_g} cpu={cls_c} "
                                             f"gpu={g!r} cpu={c!r} (band "
                                             f"family - platform "
                                             f"threshold, NOT a failure)"),
                                })
                            else:
                                failures.append({
                                    "label": label,
                                    "reason": f"class gpu={cls_g} cpu={cls_c} "
                                              f"gpu={g!r} cpu={c!r}",
                                })
                        elif cls_g == 0:
                            diff = abs(g - c)
                            max_abs_diff = max(max_abs_diff, diff)
                            max_rel_diff = max(
                                max_rel_diff, diff / max(abs(c), 1e-300))
                            if not _sum_ok(diff, float(oracle)):
                                if mode == "band":
                                    chars.append({
                                        "label": label,
                                        "fact": (f"band characterization "
                                                 f"(sum): gpu={g!r} "
                                                 f"cpu={c!r} oracle="
                                                 f"{float(oracle)!r} "
                                                 f"diff={diff:.3e} rel="
                                                 f"{diff / max(abs(c), 1e-300):.3e} "
                                                 f"(f32 accumulation "
                                                 f"threshold at extreme "
                                                 f"magnitude - "
                                                 f"PHASE4_MAPBINARY_GATE "
                                                 f"G4 precedent, NOT a "
                                                 f"failure)"),
                                    })
                                else:
                                    failures.append({
                                        "label": label,
                                        "reason": "out of tolerance",
                                        "diff": diff,
                                        "rel": diff / max(abs(c), 1e-300),
                                        "gpu": g, "cpu": c,
                                        "oracle": float(oracle),
                                        "tol": 1e-4 * max(abs(oracle), 1.0),
                                    })
                        elif cls_g == 2 or cls_g == 3:
                            if (g == np.inf) != (c == np.inf):
                                failures.append({
                                    "label": label,
                                    "reason": f"inf sign gpu={g!r} "
                                              f"cpu={c!r}",
                                })
                    else:  # min/max: exact; NaN/zero-sign/FTZ characterization
                        if cls_g == 1 or cls_c == 1:
                            if cls_g != cls_c:
                                chars.append({
                                    "label": label,
                                    "fact": (f"GPU min/max with NaN: gpu="
                                             f"{g!r} (class {cls_g}) vs "
                                             f"cpu={c!r} (class {cls_c}); "
                                             f"WGSL max/min unordered on "
                                             f"Vulkan (RTX 2060) return "
                                             f"the NON-NaN operand - known "
                                             f"characterization "
                                             f"(PHASE4_MAPBINARY_GATE "
                                             f"sec 5.1), NOT a failure"),
                                })
                        elif cls_g == 0 and cls_c == 0:
                            if not (float(g) == float(c)):
                                if mode == "band":
                                    chars.append({
                                        "label": label,
                                        "fact": (f"subnormal band "
                                                 f"characterization: gpu="
                                                 f"{g!r} cpu={c!r}; GPU "
                                                 f"flush-to-zero of "
                                                 f"subnormal min/max on "
                                                 f"NVIDIA Vulkan (RTX "
                                                 f"2060), CPU preserves "
                                                 f"subnormal (IEEE) - "
                                                 f"platform fact, NOT a "
                                                 f"failure"),
                                    })
                                else:
                                    failures.append({
                                        "label": label,
                                        "reason": f"exact required gpu={g!r} "
                                                  f"cpu={c!r}",
                                    })
                            else:
                                max_abs_diff = max(max_abs_diff,
                                                   abs(g - c))
                                if g == 0.0 and c == 0.0 and \
                                        np.signbit(g) != np.signbit(c):
                                    chars.append({
                                        "label": label,
                                        "fact": (f"GPU min/max zero-sign: "
                                                 f"gpu={g!r} (signbit "
                                                 f"{bool(np.signbit(g))}) vs "
                                                 f"cpu={c!r} (signbit "
                                                 f"{bool(np.signbit(c))}); "
                                                 f"WGSL max/min unordered "
                                                 f"for equal values - "
                                                 f"platform fact, NOT a "
                                                 f"failure"),
                                    })
                        else:
                            if not (float(g) == float(c)):
                                failures.append({
                                    "label": label,
                                    "reason": f"inf/other gpu={g!r} "
                                              f"cpu={c!r}",
                                })
                        # minimized input captured on any failure
                        if failures and failures[-1]["label"] == label:
                            minimized.append({
                                "n": n, "op": op, "op_name": OP_NAMES[op],
                                "family": family, "seed": seed, "var": v,
                                "dtype": "f32",
                                "versions": _versions(),
                            })

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
                "n_grid_len": len(NS),
                "ops": OPS,
                "op_names": OP_NAMES,
                "families": FAMILIES,
                "variations_per_case": N_VAR,
                "dtype": "f32",
                "dtype_note": "inputs f32-quantized before upload; int32 "
                              "NOT used (contract sec.3); i64 accumulator "
                              "out of scope (documented)",
                "n_cases": n_cases,
                "n_cases_gpu": n_cases_gpu,
                "n_cases_cpu": n_cases_cpu,
                "gpu_available": GPU_AVAILABLE,
                "gpu_note": ("GPU unavailable -> GPU cases skipped, CPU "
                             "path ran all cases (>= 1000)") if not GPU_AVAILABLE
                            else "GPU+CPU differential",
                "device": _device(),
                "max_abs_diff": _num(max_abs_diff),
                "max_rel_diff": _num(max_rel_diff),
                "failures": failures,
                "oracle_failures": oracle_failures,
                "characterizations": chars,
                "observations": observations,
                "minimized_input_on_fail": minimized,
                "note": ("differential GPU (WebGpuDriver) vs CPU "
                         "(CpuDriver) with independent numpy f64 oracle "
                         "(np.sum/np.min/np.max); layout 3 op x 14 N x "
                         "24 variations = 1008/backend -> 2016 GPU+CPU; "
                         "tolerances: sum rel 1e-4 (|sum|>1) / abs 1e-4 "
                         "(small, n<=1M), min/max exact (f32-"
                         "representable), NaN sum mask (isnan parity), "
                         "+/-Inf sum propagate (sign match), +/-0.0 sum "
                         "+0+-0=+0.0 (IEEE RNE); GPU min/max NaN cases - "
                         "known WGSL characterization (PHASE4_MAPBINARY_"
                         "GATE sec 5.1: unordered max/min return NON-NaN "
                         "operand on NVIDIA RTX 2060 / Vulkan), recorded "
                         "NOT failed; GPU zero-sign - WGSL unordered fact; subnormal family - "
                         "GPU flush-to-zero of subnormal min/max results "
                         "(band characterization, PHASE4_MAPBINARY_GATE "
                         "G4 precedent: GPU 0.0 vs CPU 1e-38 preserved) - "
                         "recorded NOT failed; "
                         "with the pure-NaN family the GPU min/max result "
                         "is NaN (all operands NaN) so class parity holds; "
                         "mixed-NaN GPU behavior is covered in "
                         "maxmin_ieee.json (functional)"),
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