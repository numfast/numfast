"""Phase 3 - Builder Contract canonical Scan slice tests (S19-S26, G1-G8).

Spec: BUILDER_CONTRACT.md sec.2 (S19-S26), checkpoint 98f4197.

IMPORTANT: this file is ASCII-only (no Cyrillic, no non-cp1251 chars) because
tests/test_backend.py::test_no_cupy_in_test_files reads every tests/*.py with
open() in locale encoding (cp1251 on this host) - the Phase 2 test file
follows the same convention.

Covers:
  G5 (test_g5_scan_conformance)      - nf.scan == jobs-path == np.cumsum,
                                       N in {1,63,64,65,1000,1M}, seed 42,
                                       op in {0,1,2,3}; adversarial
                                       (zeros/neg/NaN/Inf/int); tolerances
                                       SEMANTIC_CONTRACTS Scan sec.1.
  G3 (test_g3_lifecycle_single_runtime) - 10 calls Operations.scan -> 1 Runtime,
                                       1 register_all (S21).
  G4 (test_g4_no_d2_import)          - D2 import disabled (import-check).
  G6 (test_g6_old_path_disabled)     - D2 file exists, nf.scan works without D2.
  G8 (test_g8_semantics_unchanged)   - differential legacy(D2) <-> new path.
  + (test_g5_gpu_jobs_path_phase2)   - jobs path with WebGpuDriver separately
                                       (Phase 2 conformance base, EXECUTION_CONTRACT).

Evidence: evidence/builder_phase3/
  {scan_conformance,lifecycle}.json - written by this file (G7).
  {registration,manifest,delete_map,rollback_drill}.json, run.log - outside.
"""

import importlib.util
import json
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "math")))

import numfast as nf
from operations import Operations
from Runtime import Runtime as Runtime_factory
from Runtime.Runtime import _get_runtime

import operations._lib.scan_via_runtime as svr
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Compute import register_all

CHECKPOINT_SHA = "98f4197"
SEED = 42
NS = [1, 63, 64, 65, 1000, 1_000_000]
OPS = [0, 1, 2, 3]
OUT_NAMES = {0: "scan", 1: "scan_mul", 2: "scan_max", 3: "scan_min"}

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "builder_phase3")

# -- GPU availability (for the separate jobs-path WebGpuDriver test) --------
try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001 - adapter probe must never crash collection
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}

needs_gpu = pytest.mark.skipif(not GPU_AVAILABLE,
                               reason="WebGPU adapter unavailable")


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


def _write_evidence(name, payload):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# -- jobs-path helpers (names/op from descriptor + Phase 2 tests) -----------

def _local_jobs(op):
    return [{"op": "ScanLocal", "inputs": ["data"], "params": {"op": op},
             "out": [OUT_NAMES[op], "_bsum"]}]


def _chain_jobs(op):
    return [
        {"op": "ScanLocal", "inputs": ["data"], "params": {"op": op},
         "out": ["scan_local", "block_sum"]},
        {"op": "ScanTotals", "inputs": ["block_sum"], "params": {"op": op},
         "out": "block_prefix"},
        {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"],
         "params": {"op": op}, "out": OUT_NAMES[op]},
    ]


def _oracle(op, data):
    if op == 0:
        return np.cumsum(data)
    if op == 1:
        return np.cumprod(data)
    if op == 2:
        return np.maximum.accumulate(data)
    return np.minimum.accumulate(data)


def _jobs_scan(data, op):
    """Canonical jobs path via singleton (rt.compile -> rt.execute)."""
    rt = _get_runtime()
    svr._ensure_registered(rt)
    jobs = _local_jobs(op) if len(data) <= 64 else _chain_jobs(op)
    tasks = rt.compile(jobs)
    rt.execute(tasks, {"data": data})
    return rt.driver.resolve_output(OUT_NAMES[op])


def _assert_parity(res, ref, op, n, label):
    """Tolerances SEMANTIC_CONTRACTS Scan sec.1 (Parity)."""
    res = np.asarray(res, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    diff = float(np.abs(res - ref).max())
    if op in (2, 3):
        # max/min: exact for f32-representable inputs
        assert diff == 0.0, f"{label}: max/min diff={diff}"
        return diff
    if n <= 64:
        # single block, f32-representable inputs -> exact
        assert diff == 0.0, f"{label}: N<=64 diff={diff}"
        return diff
    peak = float(np.abs(ref).max())
    tol = max(1e-4 * peak, 1e-4)
    assert diff <= tol, f"{label}: diff={diff} tol={tol} peak={peak}"
    return diff


# ============================================================================
# G5 - conformance: nf.scan == jobs-path == oracle, N x op, adversarial
# ============================================================================

def test_g5_scan_conformance():
    rng = np.random.default_rng(SEED)
    max_diff = {}
    for n in NS:
        data = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        ref0 = _oracle(0, data)

        # op=0: nf.scan (path B) == jobs-path == oracle
        res_nf = nf.scan(data)
        res_jobs = _jobs_scan(data, 0)
        _assert_parity(res_nf, ref0, 0, n, f"nf.scan n={n}")
        _assert_parity(res_jobs, ref0, 0, n, f"jobs n={n}")
        assert float(np.abs(np.asarray(res_nf) - np.asarray(res_jobs)).max()) == 0.0, \
            f"nf.scan != jobs-path n={n}"
        max_diff[str(n)] = _assert_parity(res_jobs, ref0, 0, n, f"op0 n={n}")

        # op 1-3: jobs-path (valid inputs: mul - small magnitudes)
        for op in (1, 2, 3):
            if op == 1:
                data_op = (rng.standard_normal(n) * 0.1 + 1.0).astype(np.float32)
            else:
                data_op = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            res = _jobs_scan(data_op, op)
            _assert_parity(res, _oracle(op, data_op), op, n, f"op{op} n={n}")

    evidence = {
        "phase": 3,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": _versions(),
        "result": {
            "n_list": NS,
            "op_list": OPS,
            "parity": "exact N<=64 / rel 1e-4 (abs 1e-4 for small); max/min exact",
            "max_diff": {k: _num(v) for k, v in max_diff.items()},
            "gpu_available": GPU_AVAILABLE,
            "adapter": ADAPTER_INFO.get("device", "unknown"),
            "note": "nf.scan == jobs-path == np.cumsum (CPU semantic oracle, v1). "
                    "GPU conformance (path B, WebGpuDriver) - Phase 2 evidence "
                    "execution_phase2/B.json + test_g5_gpu_jobs_path_phase2.",
        },
        "evidence_schema": "v1",
    }
    _write_evidence("scan_conformance.json", evidence)


def test_g5_adversarial():
    # zeros
    z = np.zeros(64, dtype=np.float32)
    assert float(np.abs(nf.scan(z) - np.cumsum(z)).max()) == 0.0
    z1000 = np.zeros(1000, dtype=np.float32)
    assert float(np.abs(nf.scan(z1000) - np.cumsum(z1000)).max()) == 0.0

    # negatives
    rng = np.random.default_rng(SEED)
    neg = (-(rng.standard_normal(1000) * 5 + 30)).astype(np.float32)
    res = nf.scan(neg)
    ref = np.cumsum(neg)
    peak = float(np.abs(ref).max())
    assert float(np.abs(res - ref).max()) <= max(1e-4 * peak, 1e-4)

    # NaN (sum): IEEE propagate, no exception, matches oracle
    data_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
    res = nf.scan(data_nan)  # no exception
    ref = np.cumsum(data_nan)
    assert np.array_equal(np.isnan(res), np.isnan(ref))
    mask = ~np.isnan(res)
    assert float(np.abs(res[mask] - ref[mask]).max()) == 0.0

    # Inf (sum): overflow -> Inf, matches oracle
    data_inf = np.array([1e38, 1e38, 1.0], dtype=np.float32)
    res = nf.scan(data_inf)  # no exception
    ref = np.cumsum(data_inf)
    assert np.array_equal(np.isinf(res), np.isinf(ref))
    mask = ~np.isinf(res)
    assert float(np.abs(res[mask] - ref[mask]).max()) == 0.0

    # max/min NaN: IEEE NaN-propagate (Phase 4 S29 F-062(a)) - cpu.py now
    # uses np.maximum/np.minimum (IEEE 754-2019 maximum/minimum, NaN
    # propagate), matching the WGSL target semantics and the numpy oracle
    # (SEMANTIC_CONTRACTS Scan sec.1: target = NaN propagate). The AS-IS
    # python max/min order-dependent quirk is fixed in this slice.
    data_mn = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    for op in (2, 3):
        res = _jobs_scan(data_mn, op)  # no exception
        ref = (np.maximum.accumulate(data_mn) if op == 2
               else np.minimum.accumulate(data_mn))
        assert res.shape == data_mn.shape
        assert np.array_equal(np.isnan(res), np.isnan(ref)), \
            f"max/min NaN must propagate (IEEE) op={op}"
        mask = ~np.isnan(ref)
        assert float(np.abs(res[mask] - ref[mask]).max()) == 0.0, \
            f"max/min NaN mismatch op={op}"

    # int inputs: exact (|cumsum| <= 2^24)
    xi = np.arange(64, dtype=np.int32)
    res = nf.scan(xi)
    ref = np.cumsum(xi.astype(np.float32))
    assert float(np.abs(res - ref).max()) == 0.0


@needs_gpu
def test_g5_gpu_jobs_path_phase2():
    """Separate jobs path with WebGpuDriver (Phase 2 Step B base)."""
    rt = Runtime_factory(driver=WebGpuDriver())
    register_all(rt)
    rng = np.random.default_rng(SEED)
    try:
        for n in (64, 1000):
            data = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            jobs = _local_jobs(0) if n <= 64 else _chain_jobs(0)
            tasks = rt.compile(jobs)
            rt.execute(tasks, {"data": data})
            res = rt.driver.resolve_output("scan")
            ref = np.cumsum(data)
            if n <= 64:
                assert float(np.abs(res - ref).max()) == 0.0
            else:
                peak = float(np.abs(ref).max())
                assert float(np.abs(res - ref).max()) <= max(1e-4 * peak, 1e-4)
    finally:
        rt.driver.release()


# ============================================================================
# G3 - lifecycle: N calls -> 1 Runtime, 1 register_all (S21)
# ============================================================================

def test_g3_lifecycle_single_runtime(monkeypatch):
    from Runtime._lib.runtime import Runtime as RuntimeClass
    # Module Runtime.py: `import Runtime.Runtime as X` binds the package
    # attribute (the Runtime factory function), so take the module from
    # sys.modules directly.
    runtime_mod = sys.modules["Runtime.Runtime"]

    # Deterministic start: clean singleton + reset registration flag.
    monkeypatch.setattr(runtime_mod, "_runtime_instance", None)
    monkeypatch.setattr(svr, "_registered", False)

    calls = {"get": 0, "reg": 0, "inst": 0}
    real_get = runtime_mod._get_runtime

    def counting_get():
        calls["get"] += 1
        return real_get()

    monkeypatch.setattr(svr, "_get_runtime", counting_get)

    real_reg = svr.register_all

    def counting_reg(rt):
        calls["reg"] += 1
        return real_reg(rt)

    monkeypatch.setattr(svr, "register_all", counting_reg)

    orig_init = RuntimeClass.__init__

    def counting_init(self, *a, **k):
        calls["inst"] += 1
        return orig_init(self, *a, **k)

    monkeypatch.setattr(RuntimeClass, "__init__", counting_init)

    data = np.arange(32, dtype=np.float32)
    for _ in range(10):
        Operations.scan(data)

    # id is stable - one and the same singleton
    rt1 = runtime_mod._get_runtime()
    rt2 = runtime_mod._get_runtime()
    assert rt1 is rt2

    assert calls["get"] == 10, f"_get_runtime calls={calls['get']}"
    assert calls["inst"] == 1, f"Runtime instantiations={calls['inst']}"
    assert calls["reg"] == 1, f"register_all calls={calls['reg']}"
    assert len(runtime_mod._runtime_instance.kernel_table) > 0, \
        "kernel_table not rewritten / not empty"

    evidence = {
        "phase": 3,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": _versions(),
        "result": {
            "n_calls": 10,
            "runtime_instantiations": calls["inst"],
            "register_all_calls": calls["reg"],
            "get_runtime_calls": calls["get"],
            "kernel_table_entries": len(runtime_mod._runtime_instance.kernel_table),
            "result": "PASS",
        },
        "evidence_schema": "v1",
    }
    _write_evidence("lifecycle.json", evidence)


# ============================================================================
# G4 - D2 import disabled (import-check)
# ============================================================================

def test_g4_no_d2_import():
    data = (np.random.default_rng(SEED).standard_normal(32) * 5 + 30).astype(np.float32)
    nf.scan(data)  # force build + path B

    # D2 deleted in Phase 4 (S27, delete gate 8/8): no module, no file.
    assert "operations._lib.scan" not in sys.modules, \
        "operations._lib.scan imported by production code"

    spec = importlib.util.find_spec("operations._lib.scan")
    assert spec is None, "D2 module must not exist (deleted)"
    d2_path = pathlib.Path(Operations.__file__).parent / "_lib" / "scan.py"
    assert not d2_path.exists(), "D2 file must be deleted"

    # grep over Operations.py - no old import
    ops_src = pathlib.Path(Operations.__file__).read_text(encoding="utf-8")
    assert "from operations._lib.scan import" not in ops_src
    assert "scan_via_runtime" in ops_src


# ============================================================================
# G6 - old path disabled, D2 file deleted, nf.scan works without it
# ============================================================================

def test_g6_old_path_disabled():
    d2_path = pathlib.Path(Operations.__file__).parent / "_lib" / "scan.py"
    assert not d2_path.exists(), "D2 file must be deleted (Phase 4 S27)"

    rng = np.random.default_rng(SEED)
    for n in (64, 1000):
        data = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        res = nf.scan(data)
        ref = np.cumsum(data)
        if n <= 64:
            assert float(np.abs(res - ref).max()) == 0.0
        else:
            peak = float(np.abs(ref).max())
            assert float(np.abs(res - ref).max()) <= max(1e-4 * peak, 1e-4)


# ============================================================================
# G8 - semantics unchanged: scan_via_runtime vs contract oracle
# ============================================================================

def test_g8_semantics_unchanged():
    # D2 deleted in Phase 4 (S27): semantics asserted against the CONTRACT
    # oracle (np.cumsum, f32-precision), not against the legacy D2 path.
    rng = np.random.default_rng(SEED)
    for n in (64, 1000, 1_000_000):
        data = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        res_new = svr.scan(data)
        ref = np.cumsum(data)
        diff = float(np.abs(res_new - ref).max())
        if n <= 64:
            assert diff == 0.0, f"new<->oracle diff={diff} n={n}"
        else:
            peak = float(np.abs(ref).max())
            assert diff <= max(1e-4 * peak, 1e-4), \
                f"new<->oracle diff={diff} n={n} tol={max(1e-4*peak, 1e-4)}"