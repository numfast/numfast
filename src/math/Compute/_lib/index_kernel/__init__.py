"""IndexKernel — creation primitives без входных буферов: const/arange/linspace."""

from .descriptor import describe
from .cpu import cpu
from .wgsl import wgsl

__all__ = ["describe", "cpu", "wgsl"]
