"""Phase 4 - Q1 readback fix evidence (S28): fused readback_ns before/after.

Spec: SCAN_SLICE_SPEC.md sec.3 (S28).

Before: execution_phase2/C.json fused_pool_bucket (fused n64=9,966,800ns,
n1M=10,958,100ns, single cold run) + live pre-fix warm median measured on
checkpoint 7be0094 (n64=2,567,200ns, n1M=10,017,500ns).
After: measured here (post-fix, warm median of 3 runs).

Invariant: correctness unchanged - bytes_to_blockview caps by bv.length(),
count before/after identical.

Evidence: evidence/scan_slice_phase4/q1_fix.json

IMPORTANT: this file is ASCII-only (no Cyrillic) because
tests/test_backend.py::test_no_cupy_in_test_files reads tests/*.py with
open() in locale encoding (cp1251 on this host).
"""

import json
import os
import pathlib
import statistics
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "math")))

from Runtime._lib.runtime import Runtime
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime._lib.planner import planner
from Runtime._lib.optimizer import optimize_graph
from Runtime._lib.builder import builder
from Compute import register_all

CHECKPOINT_SHA = "7be0094"
SEED = 42
SCAN_CHAIN_JOBS = [
    {"op": "ScanLocal", "inputs": ["data"], "params": {"op": 0},
     "out": ["scan_local", "block_sum"]},
    {"op": "ScanTotals", "inputs": ["block_sum"], "params": {"op": 0},
     "out": "block_prefix"},
    {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"],
     "params": {"op": 0}, "out": "scan"},
]

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "scan_slice_phase4")

# Pre-fix reference values (provenance documented in evidence).
BEFORE_C_JSON = {"n64": 9966800, "n1M": 10958100}   # execution_phase2/C.json fused
BEFORE_LIVE = {"n64": 2567200, "n1M": 10017500}     # pre-fix warm median, checkpoint 7be0094

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}

needs_gpu = pytest.mark.skipif(not GPU_AVAILABLE,
                               reason="WebGPU adapter unavailable")


def _make_gpu_runtime():
    rt = Runtime(driver=WebGpuDriver())
    register_all(rt)
    return rt


def _ratio(before, after_ns):
    return round(before / after_ns, 4) if after_ns else None


def _fused_build(rt, source_data):
    rt.driver.kernel_table = rt.kernel_table
    tasks = rt.compile(SCAN_CHAIN_JOBS)
    graph = planner(tasks, rt.kernel_table)
    graph = optimize_graph(graph, level=rt.optimizer_level)
    return builder(graph, source_data, rt.driver, rt.kernel_table)


@needs_gpu
def test_q1_fix_evidence():
    rt = _make_gpu_runtime()
    rng = np.random.default_rng(SEED)
    after = {}
    try:
        for n in (64, 1_000_000):
            x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            ref = np.cumsum(x)
            packets = _fused_build(rt, {"data": x})
            rt.driver.execute_fused(packets)  # warmup
            ns_list = []
            diff = None
            for _ in range(3):
                packets = _fused_build(rt, {"data": x})
                rt.driver.execute_fused(packets)
                last = packets[-1]
                ns_list.append(last.profile.get("readback_ns", 0))
                res = rt.driver.resolve_output("scan")
                peak = float(np.abs(ref).max())
                d = float(np.abs(res - ref).max())
                tol = 0.0 if n <= 64 else max(1e-4 * peak, 1e-4)
                assert d <= tol, f"n={n} diff={d} tol={tol}"
                diff = d
            after[str(n)] = {
                "readback_ns_median": int(statistics.median(ns_list)),
                "readback_ns_list": ns_list,
                "diff": diff,
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
            "device": {"adapter": ADAPTER_INFO.get("device", "unknown"),
                       "backend": ADAPTER_INFO.get("backend_type", "unknown")},
            "before_c_json_ns": BEFORE_C_JSON,
            "before_live_warm_ns": BEFORE_LIVE,
            "after_warm_median_ns": {k: v["readback_ns_median"]
                                     for k, v in after.items()},
            "after_detail": after,
            "n64_ratio_vs_c_json": _ratio(BEFORE_C_JSON["n64"], after["64"]["readback_ns_median"]),
            "n1M_ratio_vs_c_json": _ratio(BEFORE_C_JSON["n1M"], after["1000000"]["readback_ns_median"]),
            "n64_ratio_vs_live": _ratio(BEFORE_LIVE["n64"], after["64"]["readback_ns_median"]),
            "n1M_ratio_vs_live": _ratio(BEFORE_LIVE["n1M"], after["1000000"]["readback_ns_median"]),
            "correctness": {"n64": after["64"]["diff"],
                            "n1M": after["1000000"]["diff"],
                            "invariant": "bytes_to_blockview caps by "
                                         "bv.length(); count identical "
                                         "before/after"},
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "q1_fix.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    assert os.path.getsize(path) > 0, "q1_fix.json written empty"