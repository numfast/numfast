"""Combine — линейная комбинация 2-4 входов: out = sum(wi * xi) + bias."""

from .descriptor import describe
from .cpu import cpu
from .wgsl import wgsl

__all__ = ["describe", "cpu", "wgsl"]
