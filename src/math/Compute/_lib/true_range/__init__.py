"""TrueRange — True Range: max(H-L, |H-Cp|, |L-Cp|)."""

from .descriptor import describe
from .cpu import cpu
from .wgsl import WGSL as wgsl

__all__ = ["describe", "cpu", "wgsl"]