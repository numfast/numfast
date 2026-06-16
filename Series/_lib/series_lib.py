# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from _core.backend import get_xp, get_active_name

xp = get_xp()


def _to_array(data: list | tuple) -> xp.ndarray:
    return xp.array(data, dtype=xp.float64)


def _rolling(arr: xp.ndarray, window: int) -> xp.ndarray:
    if window > len(arr):
        window = len(arr)
    indices = xp.arange(window)[None, :] + xp.arange(len(arr) - window + 1)[:, None]
    return arr[indices]


def _rolling_series(data: list, window: int) -> list:
    if get_active_name() == "wgpu":
        try:
            from _core.backend._wgpu import rolling_wgpu
            result = rolling_wgpu(data, window)
            n = len(data)
            num_windows = n - window + 1
            return [result[i * window:(i + 1) * window] for i in range(num_windows)]
        except Exception:
            pass
    arr = _to_array(data)
    result = _rolling(arr, window)
    return result.tolist()


def _normalize(arr: xp.ndarray) -> xp.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn == 0:
        return xp.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _normalize_series(data: list) -> list:
    if get_active_name() == "wgpu":
        try:
            from _core.backend._wgpu import normalize_wgpu, compute_moments_wgpu
            data_f32 = [float(v) for v in data]
            moments = compute_moments_wgpu(data_f32, len(data_f32))
            return normalize_wgpu(data_f32, moments["min"], moments["max"])
        except Exception:
            pass
    arr = _to_array(data)
    result = _normalize(arr)
    return result.tolist()
