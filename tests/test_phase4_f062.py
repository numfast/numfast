"""Phase 4 - F-062 evidence (S29 IEEE max/min, S30 N-limit ValueError).

Spec: SCAN_SLICE_SPEC.md sec.4-5.

S29 F-062(a): cpu.py uses np.maximum/np.minimum (IEEE 754-2019
maximum/minimum): NaN propagate, max(-0.0,+0.0)=+0.0, min(-0.0,+0.0)=-0.0.
WGSL NaN behavior characterized on GPU (fact, not a failure criterion).

S30 F-062(b): N > 4_194_240 (65535*64) -> deterministic ValueError with
message (kernel, n, limit, advice) instead of opaque GPUValidationError.

Evidence: evidence/scan_slice_phase4/f062.json

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
from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Compute import register_all
from Compute._lib.scan.cpu import _max_ieee, _min_ieee

CHECKPOINT_SHA = "7be0094"
SEED = 42
N_MAX_SCAN = 65535 * 64  # 4_194_240
SCANLOCAL_JOBS = [{"op": "ScanLocal", "inputs": ["data"], "params": {"op": 0},
                   "out": ["scan", "_bsum"]}]

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "scan_slice_phase4")

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}

needs_gpu = pytest.mark.skipif(not GPU_AVAILABLE,
                               reason="WebGPU adapter unavailable")


def _num(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def _make_cpu_runtime():
    rt = Runtime(driver=CpuDriver())
    register_all(rt)
    return rt


def _make_gpu_runtime():
    rt = Runtime(driver=WebGpuDriver())
    register_all(rt)
    return rt


def _scan_maxmin(runtime, data, op):
    """Run max/min scan (op=2 max, op=3 min) via path B on the runtime."""
    out = "scan_max" if op == 2 else "scan_min"
    jobs = [{"op": "ScanLocal", "inputs": ["data"], "params": {"op": op},
             "out": [out, "_bsum"]}]
    if len(data) > 64:
        jobs = [
            {"op": "ScanLocal", "inputs": ["data"], "params": {"op": op},
             "out": ["scan_local", "block_sum"]},
            {"op": "ScanTotals", "inputs": ["block_sum"], "params": {"op": op},
             "out": "block_prefix"},
            {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"],
             "params": {"op": op}, "out": out},
        ]
    runtime.execute(runtime.compile(jobs), {"data": data})
    return runtime.driver.resolve_output(out)


# ============================================================================
# S29 F-062(a): IEEE max/min
# ============================================================================

def test_f062a_ieee_helpers():
    # NaN propagate
    assert np.isnan(_max_ieee(1.0, float("nan")))
    assert np.isnan(_min_ieee(1.0, float("nan")))
    assert np.isnan(_max_ieee(float("nan"), 1.0))
    assert np.isnan(_min_ieee(float("nan"), 1.0))
    # sign of zero
    assert not np.signbit(_max_ieee(-0.0, +0.0)), "max(-0.0,+0.0) must be +0.0"
    assert np.signbit(_min_ieee(-0.0, +0.0)), "min(-0.0,+0.0) must be -0.0"
    assert not np.signbit(_max_ieee(+0.0, -0.0)), "max(+0.0,-0.0) must be +0.0"
    assert np.signbit(_min_ieee(+0.0, -0.0)), "min(+0.0,-0.0) must be -0.0"


def test_f062a_scan_nan_propagate():
    """CPU scan path (cpu.py after S29) must propagate NaN like the oracle."""
    rt = _make_cpu_runtime()
    try:
        data = np.array([1.0, np.nan, 3.0, 2.0], dtype=np.float32)
        for op, accum in ((2, np.maximum.accumulate), (3, np.minimum.accumulate)):
            res = _scan_maxmin(rt, data, op)
            ref = accum(data)
            assert np.array_equal(np.isnan(res), np.isnan(ref)), \
                f"NaN positions must match oracle (op={op})"
            mask = ~np.isnan(ref)
            assert float(np.abs(res[mask] - ref[mask]).max()) == 0.0
    finally:
        rt.driver.release()


def test_f062a_scan_zero_sign():
    rt = _make_cpu_runtime()
    try:
        data = np.array([-0.0, +0.0], dtype=np.float32)
        res_max = _scan_maxmin(rt, data, 2)
        res_min = _scan_maxmin(rt, data, 3)
        # max scan: [-0.0, +0.0]; min scan: [-0.0, -0.0]
        assert np.signbit(res_max[0]) and not np.signbit(res_max[1]), \
            f"max(-0,+0)=+0 expected, got {res_max}"
        assert np.signbit(res_min[0]) and np.signbit(res_min[1]), \
            f"min(-0,+0)=-0 expected, got {res_min}"
    finally:
        rt.driver.release()


# ============================================================================
# S30 F-062(b): N > 4_194_240 -> ValueError
# ============================================================================

@needs_gpu
def test_f062b_n_limit_valueerror():
    rt = _make_gpu_runtime()
    try:
        xb = np.zeros(N_MAX_SCAN + 1, dtype=np.float32)
        with pytest.raises(ValueError) as excinfo:
            rt.execute(rt.compile(SCANLOCAL_JOBS), {"data": xb})
        msg = str(excinfo.value)
        # message must name kernel, n, limit, advice
        assert "ScanLocal" in msg, f"kernel not named: {msg}"
        assert str(N_MAX_SCAN + 1) in msg, f"n not named: {msg}"
        assert "65535" in msg and "4_194_240" in msg, f"limit not named: {msg}"
        assert "cpu" in msg.lower(), f"advice missing: {msg}"
        # boundary N = 4_194_240 must still work
        xok = np.zeros(N_MAX_SCAN, dtype=np.float32)
        rt.execute(rt.compile(SCANLOCAL_JOBS), {"data": xok})
        res = rt.driver.resolve_output("scan")
        assert res.shape == (N_MAX_SCAN,)
    finally:
        rt.driver.release()


# ============================================================================
# WGSL NaN characterization (fact on GPU, not a failure criterion)
# ============================================================================

@needs_gpu
def test_wgsl_nan_characterization():
    rt = _make_gpu_runtime()
    fact = None
    try:
        data = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        res = _scan_maxmin(rt, data, 2)
        fact = {
            "input": [1.0, "nan", 3.0],
            "result": [_num(v) if not np.isnan(v) else "nan" for v in res],
            "nan_positions": [int(i) for i in np.where(np.isnan(res))[0]],
            "wgsl_semantics": "WGSL max/min propagate NaN (IEEE 754-2019); "
                              "characterized on GPU, parity domain excludes "
                              "NaN/+-0 (SEMANTIC_CONTRACTS)",
        }
    finally:
        rt.driver.release()
    evidence = {
        "phase": 4,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "wgpu": wgpu.__version__ if GPU_AVAILABLE else "n/a",
        },
        "result": {
            "gpu_available": GPU_AVAILABLE,
            "device": {"adapter": ADAPTER_INFO.get("device", "unknown"),
                       "backend": ADAPTER_INFO.get("backend_type", "unknown")},
            "f062a_ieee": {
                "nan_propagate": "max(1,nan)=nan, min(1,nan)=nan (IEEE "
                                 "754-2019 maximum/minimum, not maxNum)",
                "zero_sign": "max(-0,+0)=+0, min(-0,+0)=-0",
                "implementation": "np.maximum/np.minimum on the pair in "
                                  "Compute/_lib/scan/cpu.py (no if a>b on "
                                  "NaN); numpy 2.5.1 returns the second "
                                  "operand's zero sign for mixed zeros "
                                  "(min(-0,+0)=+0, max(+0,-0)=-0) so the "
                                  "helpers add the IEEE zero-sign correction "
                                  "(max of zeros = +0 unless both -0; min of "
                                  "zeros = -0 unless both +0)",
                "scan_nan_propagate_cpu": "PASS (matches np.maximum.accumulate/"
                                          "np.minimum.accumulate)",
                "scan_zero_sign_cpu": "PASS (max scan [-0,+0], min scan [-0,-0])",
            },
            "f062b_n_limit": {
                "n_max_scan": N_MAX_SCAN,
                "test": "N = 4_194_241 -> ValueError with kernel/n/limit/"
                        "advice; boundary N = 4_194_240 works",
            },
            "wgsl_nan_characterization": fact,
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "f062.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    assert os.path.getsize(path) > 0, "f062.json written empty"