# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: Series.returns / nf.returns as Shift+MapBinary(div/sub) (spec 02).

Contract: out[0] invalid (shift head, fill 0); out[i] = x[i]/x[i-1] - 1
for i>=1, no x100 factor (vs roc). NaN is a value (IEEE propagation);
invalid rows propagate via validity AND. Zeros in previous inherit
MapBinary div as-is (float: IEEE +-inf/NaN; int32: deterministic
INT_MIN-family cast, sub wraps -- verified, never invented).
Fused oracle parity: x[1:]/x[:-1]-1. Seed 42 where RNG is used.
"""

import math
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from harness import assert_float_close, load_profile

APP_DIR = str(Path(__file__).resolve().parents[2])
PROFILE = load_profile()


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _bufs(a, jobs):
    s = time.perf_counter()
    graph = a["optimize"](a["compile"](jobs))
    t_compile = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](graph["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    print(f"\nstages ms: compile+optimize={t_compile:.3f} execute={t_exec:.3f}")
    return bufs


def _isnan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _manual_jobs(a, vals, dtype, **kw):
    return [a["ir_series"]("s", np.ascontiguousarray(vals), dtype, **kw),
            a["ir_shift"]("h", "s", 1),
            a["ir_map"]("d", "s", "div", "h"),
            a["ir_map"]("r", "d", "sub", 1)]


@pytest.mark.fast
def test_returns_basic_float(kernel):
    a = kernel.alias
    for dtype in ("float32", "float64"):
        vals = np.array([10.0, 20.0, 10.0, 15.0], dtype=dtype)
        bufs = _bufs(a, _manual_jobs(a, vals, dtype))
        got = np.asarray(bufs["r"])
        assert [bool(v) for v in bufs["r#validity"]] == \
            [False, True, True, True]
        for g, w in zip(got[1:], [1.0, -0.5, 0.5]):
            assert_float_close(g, w, PROFILE, "f32", label=f"returns/{dtype}")


@pytest.mark.fast
def test_returns_int_exact(kernel):
    a = kernel.alias
    bufs = _bufs(a, _manual_jobs(a, [2, 4, 8, 4], "int32"))
    got = np.asarray(bufs["r"])
    assert got.dtype == np.dtype(np.int32)
    assert [bool(v) for v in bufs["r#validity"]] == \
        [False, True, True, True]
    assert [int(v) for v in got[1:]] == [1, 1, -1]


@pytest.mark.fast
def test_returns_head_invalid_nan(kernel):
    import numfast as nf

    s = nf.from_numpy(np.array([5.0, 10.0, 20.0], dtype=np.float64))
    g = s.returns()
    assert [bool(v) for v in g.validity] == [False, True, True]
    back = g.to_numpy()
    assert _isnan(back[0])  # invalid head restored as NaN on readback
    assert [float(v) for v in back[1:]] == pytest.approx([1.0, 1.0])


@pytest.mark.fast
def test_returns_nan_vs_validity(kernel):
    a = kernel.alias
    # NaN is a VALUE: IEEE propagation, never a validity signal.
    bufs = _bufs(a, _manual_jobs(
        a, np.array([1.0, np.nan, 4.0], dtype=np.float32), "float32"))
    got = list(np.asarray(bufs["r"]).tolist())
    assert _isnan(got[1]) and _isnan(got[2])
    assert [bool(v) for v in bufs["r#validity"]] == [False, True, True]
    # Invalid row (sidecar): stays invalid, downstream resumes.
    bufs = _bufs(a, _manual_jobs(
        a, np.array([8.0, 4.0, 2.0], dtype=np.float64), "float64",
        validity=[1, 0, 1]))
    # AND of shift-head-invalid and the input gap: all rows invalid.
    assert [bool(v) for v in bufs["r#validity"]] == [False, False, False]


@pytest.mark.fast
def test_returns_zero_previous_float_ieee(kernel):
    a = kernel.alias
    # x/0 = +-inf, 0/0 = NaN (inherited IEEE, no error, no invention).
    bufs = _bufs(a, _manual_jobs(
        a, np.array([1.0, 0.0, 3.0], dtype=np.float64), "float64"))
    got = np.asarray(bufs["r"])
    assert [bool(v) for v in bufs["r#validity"]] == [False, True, True]
    assert float(got[1]) == -1.0  # 0/1-1
    assert got[2] == np.inf  # 3/0-1


@pytest.mark.fast
def test_returns_zero_previous_int_map_semantics(kernel):
    a = kernel.alias
    # int follows the MapBinary div formula exactly (rint(f64 div)->i32,
    # sub wraps): replicate the formula independently, never hardcode.
    x = np.array([2, 0, 6], dtype=np.int32)
    bufs = _bufs(a, _manual_jobs(a, x, "int32"))
    got = np.asarray(bufs["r"])
    assert got.dtype == np.dtype(np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prev = np.zeros_like(x)
        prev[1:] = x[:-1]
        div = np.rint(x.astype(np.float64) / prev).astype(np.int32)
        ref = (div - np.int32(1)).astype(np.int32)
    assert [int(v) for v in got] == [int(v) for v in ref]
    assert [bool(v) for v in bufs["r#validity"]] == [False, True, True]


@pytest.mark.fast
def test_returns_empty(kernel):
    a = kernel.alias
    for dtype in ("int32", "float32", "float64"):
        bufs = _bufs(a, _manual_jobs(
            a, np.zeros(0, dtype=dtype), dtype))
        assert np.asarray(bufs["r"]).size == 0
        assert "r#validity" not in bufs


@pytest.mark.fast
def test_returns_vs_fused_oracle(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.01, size=60)
    x = (100.0 * np.exp(np.cumsum(rets))).astype(np.float32)
    (ref,) = a["fused_oracle"](x, [{"name": "ret", "op": "returns"}])
    bufs = _bufs(a, _manual_jobs(a, x, "float32"))
    got = np.asarray(bufs["r"])  # float64 composition dtype
    assert _isnan(ref[0])  # oracle head NaN; Series head invalid->NaN
    assert not bufs["r#validity"][0]
    for i in range(1, len(x)):
        assert_float_close(got[i], ref[i], PROFILE, "f32",
                           label=f"oracle[{i}]")


@pytest.mark.fast
def test_returns_vs_manual_composition(kernel):
    import numfast as nf

    rng = np.random.default_rng(42)
    x = (rng.normal(50, 5, size=25).astype(np.float64)).copy()
    x[7] = 0.0  # zero previous for row 8 (IEEE path)
    s = nf.from_numpy(np.ascontiguousarray(x))
    g = s.returns()
    a = kernel.alias
    bufs = _bufs(a, _manual_jobs(a, x, "float64"))
    ref_valid = list(np.asarray(bufs["r#validity"]))
    assert [bool(v) for v in g.validity] == [bool(v) for v in ref_valid]
    for gv, rv, vv in zip(g.to_numpy(), np.asarray(bufs["r"]), ref_valid):
        if not vv:
            continue
        if _isnan(rv):
            assert _isnan(gv)
        elif np.isinf(rv):
            assert gv == rv
        else:
            assert abs(float(gv) - float(rv)) <= 1e-9


@pytest.mark.fast
def test_series_and_nf_returns_api(kernel):
    import numfast as nf

    s = nf.from_numpy(np.array([4.0, 8.0, 4.0], dtype=np.float64))
    g = s.returns()
    assert g.dtype == "float64"
    assert [float(v) for v in np.asarray(g.to_numpy())[1:]] == \
        pytest.approx([1.0, -0.5])
    g2 = nf.returns(s)
    assert [float(v) for v in np.asarray(g2.to_numpy())[1:]] == \
        pytest.approx([1.0, -0.5])
    assert [bool(v) for v in g2.validity] == [False, True, True]
    si = nf.from_numpy(np.array([2, 4, 8], dtype=np.int32))
    gi = nf.returns(si)
    assert gi.dtype == "int32"
    assert [int(v) for v in np.asarray(gi.to_numpy())[1:]] == [1, 1]
    with pytest.raises(ValueError, match="returns"):
        nf.returns(nf.from_numpy(np.array([True, False], dtype=bool)))
    with pytest.raises(ValueError, match="returns"):
        nf.returns([1, 2, 3])


@pytest.mark.fast
def test_regression_shift_cumsum_rolling(kernel):
    """Regression: shift + cumsum + rolling_mean untouched by returns."""
    a = kernel.alias
    bufs = _bufs(a, [a["ir_series"]("s", [1, 2, 3, 4], "int32"),
                     a["ir_shift"]("h", "s", 1)])
    assert list(np.asarray(bufs["h"]).tolist()) == [0, 1, 2, 3]
    assert [bool(v) for v in bufs["h#validity"]] == [False, True, True, True]
    bufs = _bufs(a, [a["ir_series"]("s", [1, 2, 3, 4], "int32"),
                     a["ir_cumsum"]("h", "s")])
    assert list(np.asarray(bufs["h"]).tolist()) == [1, 3, 6, 10]
    bufs = _bufs(a, [a["ir_series"]("s", [1, 2, 3, 4, 5], "float32"),
                     a["ir_rolling_sum"]("rs", "s", 3),
                     a["ir_map"]("rm", "rs", "div", 3.0)])
    got = list(bufs["rm"])
    assert _isnan(got[0]) and _isnan(got[1])
    assert [float(v) for v in got[2:]] == pytest.approx([2.0, 3.0, 4.0],
                                                        abs=1e-5)
