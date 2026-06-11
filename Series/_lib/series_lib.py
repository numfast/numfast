try:
    import cupy as xp
except ImportError:
    import numpy as xp


def _to_array(data: list | tuple) -> xp.ndarray:
    return xp.array(data, dtype=xp.float64)


def _rolling(arr: xp.ndarray, window: int) -> xp.ndarray:
    if window > len(arr):
        window = len(arr)
    indices = xp.arange(window)[None, :] + xp.arange(len(arr) - window + 1)[:, None]
    return arr[indices]


def _normalize(arr: xp.ndarray) -> xp.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn == 0:
        return xp.zeros_like(arr)
    return (arr - mn) / (mx - mn)
