"""Scan — inclusive parallel prefix scan, op-parameterized.

Multi-workgroup implementation via three chained kernels:
  ScanLocal:  local scan per block of 64 + block totals
  ScanTotals: exclusive scan of block totals (sp[0] = neutral)
  ScanFinal:  combines accumulated offset with each element

op: "sum" (0) / "mul" (1) / "max" (2) / "min" (3), default sum.
Нейтрали: sum -> 0.0, mul -> 1.0, max -> -f32max, min -> f32max.
Семантика: INCLUSIVE prefix — out[i] = op(x[0..i]).

Usage:
    jobs = [
        {"op": "ScanLocal", "inputs": ["data"], "params": {"op": "max"}},
        {"op": "ScanTotals", "inputs": ["block_sum"], "params": {"op": "max"}},
        {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"],
         "params": {"op": "max"}, "out": "scan_max"},
    ]
"""

from .descriptor import describe_local, describe_totals, describe_final
from .cpu import cpu_local, cpu_totals, cpu_final
from .wgsl import wgsl_local, wgsl_totals, wgsl_final

__all__ = ["describe_local", "describe_totals", "describe_final",
           "cpu_local", "cpu_totals", "cpu_final",
           "wgsl_local", "wgsl_totals", "wgsl_final"]
