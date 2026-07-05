"""Scan (Prefix Sum) — inclusive parallel prefix sum.

Multi-workgroup implementation via three chained kernels:
  ScanLocal:  local prefix sum per block of 64 + block totals
  ScanTotals: prefix sum of block totals (single thread, sequential)
  ScanFinal:  adds accumulated offset to each element

Usage:
    jobs = [
        {"op": "ScanLocal", "inputs": ["data"], "params": {}},
        {"op": "ScanTotals", "inputs": ["block_sum"], "params": {}},
        {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"], "params": {}},
    ]
"""

from .descriptor import describe_local, describe_totals, describe_final
from .cpu import cpu_local, cpu_totals, cpu_final
from .wgsl import wgsl_local, wgsl_totals, wgsl_final

__all__ = ["describe_local", "describe_totals", "describe_final",
           "cpu_local", "cpu_totals", "cpu_final",
           "wgsl_local", "wgsl_totals", "wgsl_final"]
