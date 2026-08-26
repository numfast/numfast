"""Operations Extension — user-facing computation functions.

Builder entry point.

Usage:
    from numfast import scan, matmul, fft, sort, histogram

    y = scan(x)
    c = matmul(A, B, M=64, N=64, K=64)
    z = fft(x)
    s = sort(x)
    h = histogram(x, bins=10)
"""

import numpy as np


def _to_array(x):
    """Convert list/tuple/Series to ndarray. Pass through if already ndarray."""
    return np.asarray(x, dtype=np.float64)


def scan(arr):
    from operations._lib.scan_via_runtime import scan as _scan
    return _scan(arr)


def matmul(A, B, M=64, N=64, K=64):
    from operations._lib.matmul import matmul as _matmul
    A = _to_array(A)
    B = _to_array(B)
    return _matmul(A, B, M, N, K)


def fft(x):
    from operations._lib.fft import fft as _fft
    x = _to_array(x)
    return _fft(x)


def sort(x):
    from operations._lib.sort import sort as _sort
    x = _to_array(x)
    return _sort(x)


def histogram(x, bins=10):
    from operations._lib.histogram import histogram as _histogram
    x = _to_array(x)
    return _histogram(x, bins=bins)


__all__ = ["scan", "matmul", "fft", "sort", "histogram"]