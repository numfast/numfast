"""MatMul — tiled matrix multiplication.

Usage:
    jobs = [
        {"op": "MatMul", "inputs": ["A", "B"], "params": {"M": 64, "N": 64, "K": 64}},
    ]
"""

from .descriptor import describe_matmul as describe
from .cpu import cpu_matmul as cpu
from .wgsl import wgsl_matmul as wgsl

__all__ = ["describe", "cpu", "wgsl"]
