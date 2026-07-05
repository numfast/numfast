"""Bitonic Sort — parallel comparison-based sort for GPU.

Usage:
    jobs = [
        {"op": "Sort", "inputs": ["data"], "params": {"stage": 1, "substage": 1}},
    ]
    # Called repeatedly by the CPU in nested loops over stage and substage
"""

from .descriptor import describe_sort
from .cpu import cpu_sort
from .wgsl import wgsl_sort

__all__ = ["describe_sort", "cpu_sort", "wgsl_sort"]
