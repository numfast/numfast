"""Mask — select: out[i] = cond[i] ? a[i] : b[i]."""

from .descriptor import describe
from .cpu import cpu
from .wgsl import wgsl

__all__ = ["describe", "cpu", "wgsl"]
