# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import os
import math

os.environ["NUMFAST_BACKEND"] = "wgpu"

from _core.backend import set_active, get_active_name


def _wgpu_available() -> bool:
    try:
        set_active("wgpu")
        return get_active_name() == "wgpu"
    except Exception:
        return False


def test_wgpu_backend_can_activate():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU backend not available on this system")
    assert get_active_name() == "wgpu"


def test_compute_moments_basic():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU backend not available on this system")
    from _core.backend._wgpu import compute_moments_wgpu

    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = compute_moments_wgpu(data, len(data))
    assert result["count"] == 5
    assert abs(result["sum"] - 15.0) < 1e-4
    assert abs(result["min"] - 1.0) < 1e-4
    assert abs(result["max"] - 5.0) < 1e-4


def test_compute_moments_with_padding():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU backend not available on this system")
    from _core.backend._wgpu import compute_moments_wgpu

    # 500 elements padded to 512
    data = [float(i + 1) for i in range(500)] + [0.0] * 12
    result = compute_moments_wgpu(data, 500)
    assert result["count"] == 500
    assert abs(result["sum"] - 125250.0) < 0.1  # sum(1..500) = 125250
    assert abs(result["min"] - 1.0) < 1e-4
    assert abs(result["max"] - 500.0) < 1e-4


def test_compute_moments_negative_values():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU backend not available on this system")
    from _core.backend._wgpu import compute_moments_wgpu

    data = [-5.0, -3.0, -1.0, 0.0, 2.0, 4.0]
    result = compute_moments_wgpu(data, len(data))
    assert result["count"] == 6
    assert abs(result["sum"] - (-3.0)) < 1e-4
    assert abs(result["min"] - (-5.0)) < 1e-4
    assert abs(result["max"] - 4.0) < 1e-4


def test_compute_moments_single_element():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU backend not available on this system")
    from _core.backend._wgpu import compute_moments_wgpu

    data = [42.0]
    result = compute_moments_wgpu(data, 1)
    assert result["count"] == 1
    assert abs(result["sum"] - 42.0) < 1e-4
    assert abs(result["min"] - 42.0) < 1e-4
    assert abs(result["max"] - 42.0) < 1e-4


def test_compute_moments_empty():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU backend not available on this system")
    from _core.backend._wgpu import compute_moments_wgpu

    data = [0.0] * 256  # empty padded buffer
    result = compute_moments_wgpu(data, 0)
    assert result["count"] == 0
    assert result["sum"] == 0.0


def test_wgpu_matches_numpy():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU backend not available on this system")
    from _core.backend._wgpu import compute_moments_wgpu
    import numpy as np

    np.random.seed(42)
    arr = list(np.random.randn(499) * 100)

    # Pad to 512
    chunk_buf = arr + [0.0] * (512 - 499)

    wgpu_result = compute_moments_wgpu(chunk_buf, 499)

    np_arr = np.array(arr, dtype=np.float64)
    np_sum = float(np.sum(np_arr))
    np_sum_sq = float(np.sum(np_arr ** 2))
    np_min = float(np.min(np_arr))
    np_max = float(np.max(np_arr))

    assert wgpu_result["count"] == 499
    assert abs(wgpu_result["sum"] - np_sum) / max(1.0, abs(np_sum)) < 1e-4
    assert abs(wgpu_result["sum_sq"] - np_sum_sq) / max(1.0, abs(np_sum_sq)) < 1e-4
    assert abs(wgpu_result["min"] - np_min) < 1e-4
    assert abs(wgpu_result["max"] - np_max) < 1e-4


def test_stats_via_wgpu_backend():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU backend not available on this system")
    from _core import kernel
    from _core.context import create_context
    from _core.series import make_series
    from Stats.Stats import total, minimum, maximum, mean

    kernel.clear_all()
    kernel.configure(chunk_size=1000)
    ctx = create_context("test")
    s = make_series([1.0, 2.0, 3.0, 4.0, 5.0], ctx)

    assert abs(total(s) - 15.0) < 1e-4
    assert abs(minimum(s) - 1.0) < 1e-4
    assert abs(maximum(s) - 5.0) < 1e-4
    assert abs(mean(s) - 3.0) < 1e-4


def test_scaled_series_stats_wgpu():
    if not _wgpu_available():
        import pytest
        pytest.skip("WebGPU backend not available on this system")
    from _core import kernel
    from _core.context import create_context
    from _core.series import make_series
    from _core.compression import pack_scaled
    from Stats.Stats import total, minimum, maximum, mean, var
    import numpy as np

    kernel.clear_all()
    kernel.configure(chunk_size=1000)

    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    packed, meta = pack_scaled(data, "int16")
    comp = {"bits": 16, "scale": meta["scale"], "offset": meta["offset"]}
    ctx = create_context("test")
    s = make_series(data, ctx, compression=comp)

    np_ref = np.array(data, dtype=np.float64)
    assert abs(total(s) - float(np.sum(np_ref))) / float(np.sum(np_ref)) < 0.01
    assert abs(minimum(s) - float(np.min(np_ref))) < 0.01
    assert abs(maximum(s) - float(np.max(np_ref))) < 0.01
    assert abs(mean(s) - float(np.mean(np_ref))) < 0.01
    assert abs(var(s) - float(np.var(np_ref))) < 0.01
