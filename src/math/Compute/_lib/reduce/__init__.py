"""Reduce — сумма элементов массива.

Использует workspace для промежуточных результатов workgroup.
"""

from .descriptor import describe
from .cpu import cpu
from .wgsl import WGSL as wgsl

__all__ = ["describe", "cpu", "wgsl"]
