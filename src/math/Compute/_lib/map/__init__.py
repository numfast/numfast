"""Map — apply function to each element.

Supports: sin, cos, exp, sqrt, log, neg, abs
Checks: WGSL built-in functions, uniforms, read/write
"""

from .descriptor import describe
from .cpu import cpu
from .wgsl import WGSL as wgsl

__all__ = ["describe", "cpu", "wgsl"]
