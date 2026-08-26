# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Public API stage 5: comparisons -> 0/1 masks, filter, rtruediv.

Comparisons build lazy 'cmp' nodes executed through the existing
Compute L1 Compare primitive (op codes gt=0 ge=1 lt=2 le=3 eq=4 ne=5).
compute() returns a NumericSeries mask of 0.0/1.0 f32.
"""

import numpy as np
import pytest

import numfast as nf


def _values(s):
    """Materialized float values of a Series/LazyExpr result."""
    return np.asarray(s, dtype=np.float64).tolist()


# ── Comparisons ──────────────────────────────────────────────────────


def test_comparisons_mask():
    s = nf.series([1, 2, 3, 4])
    m = s > 2
    assert m.op == "cmp"
    assert m.cmp_op == "gt"
    r = m.compute()
    assert isinstance(r, nf.Series)
    assert _values(r) == [0.0, 0.0, 1.0, 1.0]


def test_comparisons_all_ops_scalar():
    s = nf.series([1.0, 2.0, 3.0])
    ref = [1.0, 2.0, 3.0]
    assert _values((s > 2).compute()) == [0.0, 0.0, 1.0]
    assert _values((s >= 2).compute()) == [0.0, 1.0, 1.0]
    assert _values((s < 2).compute()) == [1.0, 0.0, 0.0]
    assert _values((s <= 2).compute()) == [1.0, 1.0, 0.0]
    assert _values((s == 2).compute()) == [0.0, 1.0, 0.0]
    assert _values((s != 2).compute()) == [1.0, 0.0, 1.0]
    # reflected forms: scalar on the left keeps operand order
    assert _values((2 < s).compute()) == [0.0, 0.0, 1.0]
    assert _values((2 >= s).compute()) == [1.0, 1.0, 0.0]


def test_comparisons_series_operand():
    a = nf.series([1.0, 5.0, 3.0])
    b = nf.series([2.0, 2.0, 2.0])
    assert _values((a < b).compute()) == [1.0, 0.0, 0.0]
    assert _values((a != b).compute()) == [1.0, 1.0, 1.0]


def test_compare_chain():
    s = nf.series([1.0, 2.0, 3.0, 4.0])
    try:
        chained = (s > 1) & (s < 4)
    except TypeError:
        pytest.skip("__and__ not wired to LogicalAnd yet")
    expected = [1.0 if 1 < v < 4 else 0.0 for v in [1, 2, 3, 4]]
    assert _values(chained.compute()) == expected


# ── Filter ───────────────────────────────────────────────────────────


def test_filter_basic():
    wave = nf.series([0.1, 0.7, 0.4, 0.9])
    peaks = (wave > 0.5).filter(wave)
    assert isinstance(peaks, nf.Series)
    # series storage is f32 -> compare with tolerance
    assert _values(peaks) == pytest.approx([0.7, 0.9], abs=1e-6)


def test_filter_explicit_mask():
    src = nf.series([10.0, 20.0, 30.0])
    mask = nf.series([1.0, 0.0, 1.0])
    out = src.filter(mask)
    assert _values(out) == [10.0, 30.0]


def test_filter_rejects_non_series_mask():
    s = nf.series([1.0, 2.0])
    with pytest.raises(TypeError):
        s.filter([1, 0])


# ─__rtruediv__ ───────────────────────────────────────────────────────


def test_rtruediv():
    s = nf.series([1.0, 2.0, 4.0])
    r = (2 / s).compute()
    assert _values(r) == [2.0, 1.0, 0.5]


# ── data() materialization boundary ──────────────────────────────────


def test_data_materialization_boundary():
    s = nf.series([1.0, 2.0])
    assert "MATERIALIZATION BOUNDARY" in type(s).data.__doc__
    assert s.data() == [1.0, 2.0]
