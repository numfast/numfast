# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: ir_shift N->N CPU (v1, periods>=0 only).

Semantics: periods>0 right-shift, head invalid; 0 identity; negative
explicit error; |periods|>=N all invalid; empty->empty. Validity travels
with values; NaN is a value (rides as-is). Oracle: independent NumPy
(no numfast imports inside oracle fns). Seed 42 where RNG is used.
"""

import math
import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])


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


def oracle_shift(vals, periods, validity=None):
    """Independent oracle: (out_vals, out_valid) with fill 0 / False head."""
    vals = np.asarray(vals)
    n = vals.size
    out = np.zeros(n, dtype=vals.dtype)
    if validity is None:
        ov = np.zeros(n, dtype=bool)
        if periods < n:
            out[periods:] = vals[: n - periods]
            ov[periods:] = True
        return out, ov
    m = np.asarray(validity, dtype=bool)
    ov = np.zeros(n, dtype=bool)
    if periods < n and n:
        out[periods:] = vals[: n - periods]
        ov[periods:] = m[: n - periods]
    return out, ov


def oracle_rolling_sum(vals, window, min_periods=None):
    vals = np.asarray(vals, dtype=np.float64)
    mp = window if min_periods is None else min_periods
    n = vals.size
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        cnt = i - lo + 1
        if cnt >= mp and not np.isnan(vals[lo : i + 1]).any():
            out[i] = float(np.nansum(vals[lo : i + 1]))
    return out


def _assert_shift(got, got_valid, ref, ref_valid, label):
    assert list(np.asarray(got).tolist()) == list(np.asarray(ref).tolist()) or (
        np.asarray(got, dtype=np.float64) == np.asarray(ref, dtype=np.float64)
    ).all() if np.asarray(got).dtype.kind == "f" and _has_nan(ref) else \
        list(np.asarray(got).tolist()) == list(np.asarray(ref).tolist()), \
        f"{label}: values {list(got)!r} != {list(ref)!r}"
    assert [bool(v) for v in got_valid] == [bool(v) for v in ref_valid], \
        f"{label}: validity {list(got_valid)!r} != {list(ref_valid)!r}"


def _has_nan(a):
    try:
        return bool(np.isnan(np.asarray(a, dtype=np.float64)).any())
    except (TypeError, ValueError):
        return False


def _assert_nan_aware(got, ref, label):
    assert len(got) == len(ref), f"{label}: len {len(got)} != {len(ref)}"
    for i, (g, r) in enumerate(zip(got, ref)):
        if _isnan(r):
            assert _isnan(g), f"{label}[{i}]: expected NaN, got {g!r}"
        else:
            assert float(g) == float(r), f"{label}[{i}]: {g!r} != {r!r}"


@pytest.mark.fast
def test_shift_1_int32_exact(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1, 2, 3, 4], "int32"), a["ir_shift"]("h", "s", 1)]
    bufs = _bufs(a, jobs)
    assert list(np.asarray(bufs["h"]).tolist()) == [0, 1, 2, 3]
    assert [bool(v) for v in bufs["h#validity"]] == [False, True, True, True]
    assert bufs["h"].dtype == np.dtype(np.int32)


@pytest.mark.fast
def test_shift_2(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1, 2, 3, 4], "int32"), a["ir_shift"]("h", "s", 2)]
    bufs = _bufs(a, jobs)
    assert list(np.asarray(bufs["h"]).tolist()) == [0, 0, 1, 2]
    assert [bool(v) for v in bufs["h#validity"]] == [False, False, True, True]


@pytest.mark.fast
def test_shift_0_identity(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [5, 6, 7], "int32"), a["ir_shift"]("h", "s", 0)]
    bufs = _bufs(a, jobs)
    assert list(np.asarray(bufs["h"]).tolist()) == [5, 6, 7]
    assert "h#validity" not in bufs  # all-valid identity keeps no sidecar
    jobs = [a["ir_series"]("s", [5, 6, 7], "int32", validity=[1, 0, 1]),
            a["ir_shift"]("h", "s", 0)]
    bufs = _bufs(a, jobs)
    assert list(np.asarray(bufs["h"]).tolist()) == [5, 6, 7]
    assert [bool(v) for v in bufs["h#validity"]] == [True, False, True]


@pytest.mark.fast
def test_shift_ge_N_all_invalid(kernel):
    a = kernel.alias
    for p in (4, 7):
        jobs = [a["ir_series"]("s", [1, 2, 3, 4], "int32"),
                a["ir_shift"]("h", "s", p)]
        bufs = _bufs(a, jobs)
        assert list(np.asarray(bufs["h"]).tolist()) == [0, 0, 0, 0]
        assert [bool(v) for v in bufs["h#validity"]] == [False] * 4


@pytest.mark.fast
def test_shift_empty(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [], "int32"), a["ir_shift"]("h", "s", 1)]
    bufs = _bufs(a, jobs)
    assert np.asarray(bufs["h"]).size == 0
    assert bufs["h"].dtype == np.dtype(np.int32)


@pytest.mark.fast
def test_shift_validity_gaps(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [10, 20, 30, 40], "int32", validity=[1, 0, 1, 1]),
            a["ir_shift"]("h", "s", 1)]
    bufs = _bufs(a, jobs)
    ref, refv = oracle_shift([10, 20, 30, 40], 1, [1, 0, 1, 1])
    assert list(np.asarray(bufs["h"]).tolist()) == list(ref.tolist())
    assert [bool(v) for v in bufs["h#validity"]] == [bool(v) for v in refv]


@pytest.mark.fast
def test_shift_nan_vs_validity(kernel):
    a = kernel.alias
    # NaN is a VALUE: rides right like any float, stays valid.
    jobs = [a["ir_series"]("s", np.array([1.0, np.nan, 3.0], dtype=np.float32),
                           "float32"),
            a["ir_shift"]("h", "s", 1)]
    bufs = _bufs(a, jobs)
    got = list(np.asarray(bufs["h"]).tolist())
    assert got[0] == 0.0
    assert not _isnan(got[1]) and abs(float(got[1]) - 1.0) < 1e-6
    assert _isnan(got[2])
    assert [bool(v) for v in bufs["h#validity"]] == [False, True, True]
    # Invalid row (sidecar) vs NaN row: invalid head fill is 0.0 + False,
    # NaN row is NaN + True -- never conflated.
    jobs = [a["ir_series"]("s", np.array([1.0, 2.0, 3.0], dtype=np.float32),
                           "float32", validity=[1, 0, 1]),
            a["ir_shift"]("h", "s", 1)]
    bufs = _bufs(a, jobs)
    got = np.asarray(bufs["h"])
    assert [bool(v) for v in bufs["h#validity"]] == [False, True, False]
    assert float(got[1]) == 1.0 and float(got[2]) == 2.0


@pytest.mark.fast
def test_shift_dtypes_preserved(kernel):
    a = kernel.alias
    for dtype, vals in (("int32", np.array([1, 2, 3], dtype=np.int32)),
                        ("float32", np.array([1.5, 2.5, 3.5], dtype=np.float32)),
                        ("float64", np.array([1.5, 2.5, 3.5], dtype=np.float64))):
        jobs = [a["ir_series"]("s", vals, dtype), a["ir_shift"]("h", "s", 1)]
        bufs = _bufs(a, jobs)
        ref, refv = oracle_shift(vals, 1)
        assert bufs["h"].dtype == np.asarray(ref).dtype, dtype
        _assert_nan_aware(list(np.asarray(bufs["h"]).tolist()), list(ref.tolist()), dtype)
        assert [bool(v) for v in bufs["h#validity"]] == [bool(v) for v in refv]


@pytest.mark.fast
def test_shift_negative_rejected(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="negative"):
        a["ir_shift"]("h", "s", -1)
    with pytest.raises(ValueError, match="non-negative int"):
        a["ir_shift"]("h", "s", 1.5)
    with pytest.raises(ValueError, match="non-negative int"):
        a["ir_shift"]("h", "s", True)


@pytest.mark.fast
def test_series_and_nf_shift_api(kernel):
    import numfast as nf

    s = nf.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))
    g = s.shift(1)
    assert g.dtype == "int32"
    # to_numpy keeps int fill 0 at invalid head (ints have no NaN restore)
    assert list(np.asarray(g.to_numpy()).tolist()) == [0, 1, 2, 3]
    assert [bool(v) for v in g.validity] == [False, True, True, True]
    g2 = nf.shift(s, 2)
    assert list(np.asarray(g2.to_numpy()).tolist()) == [0, 0, 1, 2]
    assert [bool(v) for v in g2.validity] == [False, False, True, True]
    with pytest.raises(ValueError, match="periods>=0|negative"):
        s.shift(-1)


@pytest.mark.fast
def test_sort_rolling_shift_alignment_oracle(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    n = 25
    ts = rng.integers(0, 10, size=n)
    vals = rng.normal(100, 10, size=n).astype(np.float32)
    window, periods = 4, 2
    jobs = [a["ir_series"]("s_ts", np.ascontiguousarray(ts), "int32"),
            a["ir_series"]("s_px", np.ascontiguousarray(vals), "float32"),
            a["ir_sort"]("p", "s_ts"),
            a["ir_gather"]("g_ts", "s_ts", "p"),
            a["ir_gather"]("g_px", "s_px", "p"),
            a["ir_rolling_sum"]("r", "g_px", window),
            a["ir_shift"]("h", "r", periods)]
    bufs = _bufs(a, jobs)
    perm = np.argsort(ts, kind="stable")
    assert [int(i) for i in np.asarray(bufs["p"]).tolist()] == [int(i) for i in perm]
    sg = vals[perm]
    assert np.allclose(np.asarray(bufs["g_px"], dtype=np.float64),
                       np.asarray(sg, dtype=np.float64), atol=1e-6)
    ref_roll = oracle_rolling_sum(sg.astype(np.float64), window)
    _assert_nan_aware(list(np.asarray(bufs["r"]).tolist()), list(ref_roll.tolist()),
                      "rolling")
    ref_h, ref_hv = oracle_shift(ref_roll, periods)
    got_h = np.asarray(bufs["h"])
    got_hv = np.asarray(bufs["h#validity"])
    assert [bool(v) for v in got_hv] == [bool(v) for v in ref_hv]
    _assert_nan_aware([g if v else np.nan for g, v in
                       zip(got_h.tolist(), got_hv.tolist())],
                      [r if v else np.nan for r, v in
                       zip(ref_h.tolist(), ref_hv.tolist())], "shift-after-rolling")
    pd = pytest.importorskip("pandas")
    # Pandas parity at the value level: shift of the rolled values with
    # NaN head matches our data buffer where valid (head excluded).
    ps = pd.Series(np.asarray(bufs["r"], dtype=np.float64)).shift(periods)
    for g, r, v in zip(got_h.tolist(), ps.tolist(), got_hv.tolist()):
        if not bool(v):
            continue
        if _isnan(r):
            assert _isnan(g), f"pandas parity: {g!r} != {r!r}"
        else:
            assert abs(float(g) - float(r)) < 1e-9


@pytest.mark.fast
def test_shift_fuzz_N_periods(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    max_n = 0
    for trial in range(30):
        n = int(rng.integers(0, 40))
        max_n = max(max_n, n)
        vals = rng.integers(-50, 50, size=n).astype(np.int32)
        use_valid = bool(rng.integers(0, 2))
        valid = None
        if use_valid and n:
            valid = [bool(x) for x in rng.integers(0, 2, size=n)]
        p = int(rng.integers(0, n + 3)) if n else int(rng.integers(0, 3))
        kw = {"validity": [int(v) for v in valid]} if valid is not None else {}
        jobs = [a["ir_series"]("s", np.ascontiguousarray(vals), "int32", **kw),
                a["ir_shift"]("h", "s", p)]
        bufs = _bufs(a, jobs)
        ref, refv = oracle_shift(vals, p, None if valid is None else valid)
        got = np.asarray(bufs["h"])
        assert got.dtype == np.dtype(np.int32)
        assert list(got.tolist()) == list(ref.tolist()), f"t={trial} n={n} p={p}"
        if p == 0 and valid is None:
            assert "h#validity" not in bufs
        else:
            gotv = np.asarray(bufs["h#validity"]) if "h#validity" in bufs \
                else np.ones(n, dtype=bool)
            assert [bool(v) for v in gotv] == [bool(v) for v in refv], \
                f"t={trial} n={n} p={p}"
    print(f"\nfuzz trials=30 max_n={max_n} seed=42")


@pytest.mark.fast
def test_regression_rolling_and_sort(kernel):
    """Regression: rolling_sum + sort paths untouched by the shift patch."""
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1, 2, 3, 4, 5], "float32"),
            a["ir_rolling_sum"]("r", "s", 3)]
    got = list(_bufs(a, jobs)["r"])
    assert _isnan(got[0]) and _isnan(got[1])
    assert [float(v) for v in got[2:]] == pytest.approx([6.0, 9.0, 12.0], abs=1e-5)
    jobs = [a["ir_series"]("s", [30, 10, 20, 10], "int32"),
            a["ir_sort"]("p", "s"),
            a["ir_gather"]("g", "s", "p")]
    bufs = _bufs(a, jobs)
    assert [int(i) for i in np.asarray(bufs["p"]).tolist()] == [1, 3, 2, 0]
    assert list(np.asarray(bufs["g"]).tolist()) == [10, 10, 20, 30]
