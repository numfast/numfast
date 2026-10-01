# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: rolling_sum + rolling_mean as RollingSum+MapBinary(div) (spec 02).

int без NaN — exact; float — tolerance/ULP из conformance-profile.toml.
mean-композиция точна при min_periods == window (дефолт).
"""

import math
import time
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


@pytest.mark.fast
def test_rolling_sum_full_windows_float(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1, 2, 3, 4, 5], "float32"), a["ir_rolling_sum"]("r", "s", 3)]
    got = list(_bufs(a, jobs)["r"])
    assert _isnan(got[0]) and _isnan(got[1])
    for v, w in zip(got[2:], [6.0, 9.0, 12.0]):
        assert_float_close(v, w, PROFILE, "f32", label="rolling_sum")


@pytest.mark.fast
def test_rolling_sum_min_periods_int_exact(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1, 2, 3, 4]), a["ir_rolling_sum"]("r", "s", 2, min_periods=1)]
    got = list(_bufs(a, jobs)["r"])
    assert [int(v) for v in got] == [1, 3, 5, 7]


@pytest.mark.fast
def test_rolling_mean_is_sum_plus_div(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("s", [1, 2, 3, 4, 5], "float32"),
        a["ir_rolling_sum"]("rs", "s", 3),
        a["ir_map"]("rm", "rs", "div", 3.0),
    ]
    bufs = _bufs(a, jobs)
    got = list(bufs["rm"])
    assert _isnan(got[0]) and _isnan(got[1])
    for v, w in zip(got[2:], [2.0, 3.0, 4.0]):
        assert_float_close(v, w, PROFILE, "f32", label="rolling_mean")


@pytest.mark.fast
def test_rolling_window_one_identity(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [7, 8, 9]), a["ir_rolling_sum"]("r", "s", 1)]
    got = list(_bufs(a, jobs)["r"])
    assert [int(v) for v in got] == [7, 8, 9]


@pytest.mark.fast
def test_rolling_window_full_length(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [2, 4, 6], "float32"), a["ir_rolling_sum"]("r", "s", 3)]
    got = list(_bufs(a, jobs)["r"])
    assert _isnan(got[0]) and _isnan(got[1])
    assert_float_close(got[2], 12.0, PROFILE, "f32", label="full-window")


@pytest.mark.fast
def test_rolling_bad_params_rejected(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="window must be int >= 1"):
        a["ir_rolling_sum"]("r", "s", 0)
    with pytest.raises(ValueError, match="min_periods"):
        a["ir_rolling_sum"]("r", "s", 3, min_periods=4)


# -- M7a: rolling_sum was vectorised (sliding_window_view + .sum(axis=1)).
# The four semantics below are contractual and are pinned independently of
# the implementation: the oracle is the pre-M7a per-window Python loop, so the
# values must stay BIT-IDENTICAL, not merely close.


def _oracle_rolling_sum(values, window, min_periods, validity=None):
    """Pre-M7a per-window loop, verbatim. ANY-NaN poisons the whole window."""
    a = np.asarray(values, dtype=np.float64)
    if validity is not None:
        a = np.where(np.asarray(validity, dtype=bool), a, np.nan)
    n = a.size
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        cnt = i - lo + 1
        if cnt >= min_periods:
            w = a[lo:i + 1]
            if not np.isnan(w).any():
                out[i] = w.sum()
    src = np.asarray(values)
    if np.issubdtype(src.dtype, np.integer) and not np.isnan(out).any():
        out = np.rint(out).astype(src.dtype)
    return out


def _exact(got, ref, label):
    got, ref = np.asarray(got), np.asarray(ref)
    assert got.dtype == ref.dtype, f"{label}: dtype {got.dtype} != {ref.dtype}"
    gn = np.isnan(got.astype(np.float64))
    rn = np.isnan(ref.astype(np.float64))
    assert np.array_equal(gn, rn), f"{label}: NaN mask differs at {np.flatnonzero(gn != rn)}"
    m = ~rn
    assert np.array_equal(got[m], ref[m]), f"{label}: values differ"


@pytest.mark.fast
def test_m7a_rolling_sum_bit_identical_to_per_window_loop(kernel):
    """M7a: vectorised result is bit-identical to the old per-window loop.

    Grid: n x window x min_periods x dtype, seed 42, plus NaN/validity/inf rows.
    """
    a = kernel.alias
    rng = np.random.default_rng(42)
    checked = 0
    for dtype in ("int32", "int64", "float32", "float64"):
        for n in (1, 2, 7, 100, 1000):
            if np.issubdtype(np.dtype(dtype), np.integer):
                vals = rng.integers(-1000, 1000, size=n).astype(dtype)
            else:
                vals = (rng.standard_normal(n) * 100.0).astype(dtype)
            for window in (1, 2, 3, 7, 20, 64, 1000):
                for mp in sorted({1, window}):
                    jobs = [a["ir_series"]("s", np.ascontiguousarray(vals), dtype),
                            a["ir_rolling_sum"]("r", "s", window, min_periods=mp)]
                    got = np.asarray(_bufs(a, jobs)["r"])
                    _exact(got, _oracle_rolling_sum(vals, window, mp),
                           f"{dtype} n={n} w={window} mp={mp}")
                    checked += 1
    # 7 windows x min_periods in {1, window}; window == 1 collapses to one mp
    assert checked == 4 * 5 * (7 * 2 - 1)


@pytest.mark.fast
def test_m7a_rolling_sum_any_nan_in_window_poisons(kernel):
    """Rule 2: a NaN ANYWHERE in the window poisons it; clean windows after
    the NaN sum normally (no sticky/prefix poisoning)."""
    a = kernel.alias
    vals = np.array([1.0, np.nan, 3.0, 4.0, 5.0], dtype=np.float64)
    jobs = [a["ir_series"]("s", vals, "float64"),
            a["ir_rolling_sum"]("r", "s", 3, min_periods=1)]
    got = np.asarray(_bufs(a, jobs)["r"])
    assert float(got[0]) == 1.0
    assert _isnan(got[1]) and _isnan(got[2]) and _isnan(got[3])
    assert abs(float(got[4]) - 12.0) < 1e-9  # window [3,4,5] is clean again


@pytest.mark.fast
def test_m7a_rolling_sum_validity_row_poisons_its_window(kernel):
    """Rule 3: the validity sidecar injects NaN BEFORE the reduction, so an
    invalid row poisons every window that contains it."""
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1, 2, 3, 4, 5], "int32", validity=[1, 1, 0, 1, 1]),
            a["ir_rolling_sum"]("r", "s", 2, min_periods=1)]
    got = np.asarray(_bufs(a, jobs)["r"])
    ref = _oracle_rolling_sum([1, 2, 3, 4, 5], 2, 1, validity=[1, 1, 0, 1, 1])
    _exact(got, ref, "validity-gap w=2 mp=1")
    assert float(got[0]) == 1.0 and float(got[1]) == 3.0
    assert _isnan(got[2]) and _isnan(got[3])
    assert abs(float(got[4]) - 9.0) < 1e-9


@pytest.mark.fast
def test_m7a_rolling_sum_int_rint_only_when_no_nan(kernel):
    """Rule 4: rint -> src.dtype only when NO output is NaN; a single NaN
    keeps the whole array float64."""
    a = kernel.alias
    clean = [a["ir_series"]("s", [1, 2, 3, 4], "int32"),
             a["ir_rolling_sum"]("r", "s", 2, min_periods=1)]
    got = np.asarray(_bufs(a, clean)["r"])
    assert got.dtype == np.int32 and list(got) == [1, 3, 5, 7]
    # same int32 source, but the validity gap forces NaN -> stays float64
    gapped = [a["ir_series"]("s", [1, 2, 3, 4], "int32", validity=[1, 1, 0, 1]),
              a["ir_rolling_sum"]("r", "s", 2, min_periods=1)]
    got2 = np.asarray(_bufs(a, gapped)["r"])
    assert got2.dtype == np.float64
    assert _isnan(got2[2]) and _isnan(got2[3])
    _exact(got2, _oracle_rolling_sum([1, 2, 3, 4], 2, 1, validity=[1, 1, 0, 1]),
           "int+NaN")


@pytest.mark.fast
def test_m7a_rolling_sum_min_periods_gate(kernel):
    """Rule 1: cnt >= mp gate where cnt = i - max(0, i-window+1) + 1."""
    a = kernel.alias
    vals = np.array([1, 2, 3, 4, 5], dtype=np.int32)

    def run(mp):
        jobs = [a["ir_series"]("s", vals, "int32"),
                a["ir_rolling_sum"]("r", "s", 3, min_periods=mp)]
        return np.asarray(_bufs(a, jobs)["r"])

    full, loose, ones = run(3), run(2), run(1)
    # mp == window -> only the two full-window rows leave the NaN region
    _exact(full, _oracle_rolling_sum(vals, 3, 3), "mp=3")
    assert full.dtype == np.float64          # NaN present -> no int rint cast
    assert list(full[2:]) == [6.0, 9.0, 12.0]
    _exact(loose, _oracle_rolling_sum(vals, 3, 2), "mp=2")
    assert list(loose[1:]) == [3.0, 6.0, 9.0, 12.0]
    # mp == 1 -> prefix sums from row 0, no NaN at all -> int32 rint cast
    _exact(ones, _oracle_rolling_sum(vals, 3, 1), "mp=1")
    assert ones.dtype == np.int32 and list(ones) == [1, 3, 6, 9, 12]


@pytest.mark.fast
def test_m7a_rolling_sum_edges_one_and_window_gt_n(kernel):
    """Edges: window=1 identity, n=1, window > n (all NaN), window > n with
    min_periods=1 (prefix sums), and +/-inf propagation."""
    a = kernel.alias
    vals = np.array([7.0, 8.0, 9.0], dtype=np.float64)

    def run(v, window, min_periods=None):
        job = a["ir_rolling_sum"](
            "r", "s", window,
            **({} if min_periods is None else {"min_periods": min_periods}))
        return np.asarray(_bufs(a, [a["ir_series"]("s", v, "float64"), job])["r"])

    assert list(run(vals, 1)) == [7.0, 8.0, 9.0]
    assert list(run(vals[:1], 1)) == [7.0]
    over = run(vals, 9)                       # window > n, mp == window
    assert all(_isnan(v) for v in over)
    over_mp1 = run(vals, 9, min_periods=1)     # window > n, mp == 1 -> prefixes
    _exact(over_mp1, _oracle_rolling_sum(vals, 9, 1), "window>n mp=1")
    assert list(over_mp1) == [7.0, 15.0, 24.0]
    inf = np.array([1.0, np.inf, -np.inf, 2.0], dtype=np.float64)
    got = run(inf, 2, min_periods=1)
    _exact(got, _oracle_rolling_sum(inf, 2, 1), "inf pairs")
    assert float(got[0]) == 1.0 and float(got[1]) == np.inf
    assert _isnan(got[2]) and float(got[3]) == -np.inf
