"""MapBinary — бинарные операции: + - * / max min, со скалярным режимом."""

from .descriptor import describe
from .cpu import cpu
from .wgsl import WGSL as wgsl

__all__ = ["describe", "cpu", "wgsl"]
