"""Clamp — ограничение значений: out = max(min(x, max_val), min_val)."""

from .descriptor import describe
from .cpu import cpu
from .wgsl import WGSL as wgsl

__all__ = ["describe", "cpu", "wgsl"]
