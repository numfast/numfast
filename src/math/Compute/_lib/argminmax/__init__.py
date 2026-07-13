"""ArgMin/ArgMax — reduce with index tracking. Returns (value, index) pair.

Single-pass reduction within a workgroup. Two outputs:
  out_val: min/max value
  out_idx: index of that value
"""

from .descriptor import describe
from .cpu import cpu
from .wgsl import WGSL as wgsl

__all__ = ["describe", "cpu", "wgsl"]
