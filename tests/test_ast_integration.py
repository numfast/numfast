"""Integration test: AST -> ISA -> Runtime end-to-end.

Tests the full pipeline:
  Expression -> AST.parse -> ast_to_dict -> ISA.lower -> format_jobs
  -> Runtime.compile -> Runtime.execute -> result

Single test: SMA(data, 14) compared against NumPy reference.
"""

import sys
import os
import numpy as np

# Path setup: add numfast src dirs
_numfast_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
for _sub in ["", "core", "math"]:
    _p = os.path.join(_numfast_root, _sub) if _sub else _numfast_root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AST._lib.parser import parse
from AST._lib.ast_nodes import ast_to_dict
from ISA._lib.lowering import lower, format_jobs
from Runtime import Runtime
from Compute.Compute import register_all


def _numpy_sma(data, period):
    """Reference SMA implementation."""
    result = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1 : i + 1])
    return result


def test_sma_end_to_end():
    """Full pipeline: SMA(close, 14) -> result matches NumPy."""
    # Setup
    runtime = Runtime()
    register_all(runtime)

    # Expression -> AST -> dict -> ISA lowering -> Runtime jobs
    ast = parse("SMA(data, 14)")
    d = ast_to_dict(ast)
    graph, out = lower(d)
    jobs = format_jobs(graph, out)

    # Compile and execute
    tasks = runtime.compile(jobs)
    data_arr = np.arange(1.0, 31.0, dtype=np.float64)
    runtime.execute(tasks, {"data": data_arr})

    # Verify
    result = runtime.driver.resolve_output(out)
    expected = _numpy_sma(data_arr, 14)

    # Compare valid (computable) elements only
    valid = ~np.isnan(expected)
    assert np.allclose(result[valid], expected[valid], atol=0.01), (
        f"SMA mismatch. Max diff: {np.max(np.abs(result[valid] - expected[valid]))}"
    )
