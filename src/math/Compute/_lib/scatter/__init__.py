"""Scatter — indexed write: out[index[i]] = value[i]. 
Inverse of Gather. Warning: conflicting indices = undefined result.
"""

from .descriptor import describe
from .cpu import cpu
from .wgsl import WGSL as wgsl

__all__ = ["describe", "cpu", "wgsl"]
