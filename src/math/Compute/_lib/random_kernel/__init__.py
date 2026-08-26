"""RandomKernel — counter-based PCG-u32 random creation без входных буферов."""

from .descriptor import describe
from .cpu import cpu
from .wgsl import WGSL as wgsl

__all__ = ["describe", "cpu", "wgsl"]
