"""NumFast High-Level Operations API.

Usage:
    from numfast import scan, matmul, fft, sort, histogram

    y = scan(x)
    c = matmul(A, B, M=64, N=64, K=64)
    z = fft(x)
    s = sort(x)
    h = histogram(x, bins=10)
"""

from .scan import scan
from .matmul import matmul
from .fft import fft
from .sort import sort
from .histogram import histogram

__all__ = ["scan", "matmul", "fft", "sort", "histogram"]
