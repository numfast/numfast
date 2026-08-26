"""Canonical Scan (inclusive prefix sum) - path B: jobs -> Runtime singleton.

Phase 3 REFACTOR_AUDIT sec.4 / BUILDER_CONTRACT S19-S26 (S22 - canonical Scan
slice). One responsibility: run inclusive prefix scan (op=sum) through the
canonical path B - `Runtime._get_runtime()` singleton (S21), NOT creating its
own Runtime per call (route A / D2 - DISABLE in Phase 3).

Jobs rules (SEMANTIC_CONTRACTS Scan sec.1; Compute/_lib/scan/descriptor.py):
  N <= 64  -> [ScanLocal]                        (single block, exact f32)
  N > 64   -> [ScanLocal, ScanTotals, ScanFinal]  (3-phase chain)
op codes (F-046): 0=sum, 1=mul, 2=max, 3=min. Public contract of
Operations.scan is op=sum (0). Output names - from Phase 2 SCAN_CHAIN_JOBS
(tests/test_phase2_execution_contract.py:46-55).

use_gpu: IGNORED in v1 - the singleton is created without a driver, execute()
defaults to CpuDriver (AS-IS, Runtime/_lib/runtime.py:116-118; S21 "Driver:
v1: singleton without driver -> CpuDriver"). WebGPU injection for nf.scan is
a Phase 4 decision (driver selection). Conformance of path B on WebGpuDriver
was proven in Phase 2 separately (jobs path, EXECUTION_CONTRACT Step B).

Registration: `Compute.register_all` is called NO MORE than once per process
(S21: "N nf.scan calls -> 1 Runtime, 1 register_all"). If kernel_table is
already filled by bootstrap (_ensure_built -> Compute.setup) it is NOT called
again (last-wins, idempotent, S12).
"""

from Runtime.Runtime import _get_runtime
from Compute import register_all

# Job structures - exactly as in Phase 2 (test_phase2_execution_contract.py:46-55).
_SCANLOCAL_JOBS = [
    {"op": "ScanLocal", "inputs": ["data"], "params": {"op": 0},
     "out": ["scan", "_bsum"]},
]
_SCAN_CHAIN_JOBS = [
    {"op": "ScanLocal", "inputs": ["data"], "params": {"op": 0},
     "out": ["scan_local", "block_sum"]},
    {"op": "ScanTotals", "inputs": ["block_sum"], "params": {"op": 0},
     "out": "block_prefix"},
    {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"],
     "params": {"op": 0}, "out": "scan"},
]

# Module-level lifecycle flag: register_all at most once per process.
_registered = False


def _ensure_registered(rt):
    """Register Compute primitives once per process (S20/S21)."""
    global _registered
    if _registered:
        return
    if len(rt.kernel_table) == 0:
        register_all(rt)
    _registered = True


def scan(data, use_gpu=False):
    """Inclusive prefix sum (op=sum) via the canonical path B.

    Args:
        data: 1D array (host float64/float32/int; Scan contract: semantic f32,
            host storage float64).
        use_gpu: IGNORED in v1 (see module docstring; GPU injection for
            nf.scan is a Phase 4 decision). Conformance on WebGpuDriver was
            proven in Phase 2 separately (jobs path).

    Returns:
        prefix_sum: 1D float64 array, same length as data.
    """
    n = len(data)
    jobs = _SCANLOCAL_JOBS if n <= 64 else _SCAN_CHAIN_JOBS

    rt = _get_runtime()
    _ensure_registered(rt)
    tasks = rt.compile(jobs)
    rt.execute(tasks, {"data": data})
    return rt.driver.resolve_output("scan")