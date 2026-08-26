"""Compare — element-wise comparison: out[i] = a[i] op b[i], result 0.0/1.0."""

from .descriptor import describe
from .cpu import cpu
from .wgsl import wgsl_generator as wgsl

__all__ = ["describe", "cpu", "wgsl"]
