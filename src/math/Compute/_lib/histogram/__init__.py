"""Histogram — bin counting with GPU atomics.

Usage:
    jobs = [
        {"op": "Histogram", "inputs": ["data"], "params": {"min_val": 0.0, "max_val": 100.0, "num_bins": 10}},
    ]
"""

from .descriptor import describe
from .cpu import cpu_histogram
from .wgsl import wgsl_histogram

__all__ = ["describe", "cpu_histogram", "wgsl_histogram"]
