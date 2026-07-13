"""Exponential Moving Average — CPU only.

GPU: use StateKernel_Single(mode=0, a=2/(period+1), b=1-2/(period+1))
"""

from .descriptor import describe
from .cpu import cpu

__all__ = ["describe", "cpu"]