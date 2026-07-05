# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import math

from _core.context import create_context
from Series._lib import NumericSeries
from Series._lib.expr import _LazyExpr
from Series._lib.math_ops import sin, cos, exp, log, sqrt, neg


def test_lazy_add():
    ctx = create_context("test_lazy_add")
    a = NumericSeries([1.0, 2.0, 3.0], ctx)
    b = NumericSeries([4.0, 5.0, 6.0], ctx)
    c = a + b
    assert isinstance(c, _LazyExpr)
    assert c.data() == [5.0, 7.0, 9.0]


def test_lazy_sub():
    ctx = create_context("test_lazy_sub")
    a = NumericSeries([10.0, 20.0, 30.0], ctx)
    b = NumericSeries([1.0, 2.0, 3.0], ctx)
    c = a - b
    assert isinstance(c, _LazyExpr)
    assert c.data() == [9.0, 18.0, 27.0]


def test_lazy_mul():
    ctx = create_context("test_lazy_mul")
    a = NumericSeries([2.0, 3.0, 4.0], ctx)
    b = NumericSeries([5.0, 6.0, 7.0], ctx)
    c = a * b
    assert isinstance(c, _LazyExpr)
    assert c.data() == [10.0, 18.0, 28.0]


def test_lazy_add_scalar():
    ctx = create_context("test_lazy_add_scalar")
    a = NumericSeries([1.0, 2.0, 3.0], ctx)
    c = a + 10
    assert c.data() == [11.0, 12.0, 13.0]


def test_lazy_radd_scalar():
    ctx = create_context("test_lazy_radd")
    a = NumericSeries([1.0, 2.0, 3.0], ctx)
    c = 10 + a
    assert c.data() == [11.0, 12.0, 13.0]


def test_lazy_sub_scalar():
    ctx = create_context("test_lazy_sub_scalar")
    a = NumericSeries([10.0, 20.0, 30.0], ctx)
    c = a - 5
    assert c.data() == [5.0, 15.0, 25.0]


def test_lazy_rsub_scalar():
    ctx = create_context("test_lazy_rsub")
    a = NumericSeries([1.0, 2.0, 3.0], ctx)
    c = 10 - a
    assert c.data() == [9.0, 8.0, 7.0]


def test_lazy_mul_scalar():
    ctx = create_context("test_lazy_mul_scalar")
    a = NumericSeries([1.0, 2.0, 3.0], ctx)
    c = a * 3
    assert c.data() == [3.0, 6.0, 9.0]


def test_lazy_rmul_scalar():
    ctx = create_context("test_lazy_rmul")
    a = NumericSeries([1.0, 2.0, 3.0], ctx)
    c = 3 * a
    assert c.data() == [3.0, 6.0, 9.0]


def test_neg():
    ctx = create_context("test_neg")
    a = NumericSeries([1.0, -2.0, 3.0], ctx)
    c = -a
    assert isinstance(c, _LazyExpr)
    assert c.data() == [-1.0, 2.0, -3.0]


def test_sin():
    ctx = create_context("test_sin")
    a = NumericSeries([0.0, math.pi / 2, math.pi], ctx)
    c = sin(a)
    assert isinstance(c, _LazyExpr)
    result = c.data()
    assert abs(result[0] - 0.0) < 1e-12
    assert abs(result[1] - 1.0) < 1e-12
    assert abs(result[2] - 0.0) < 1e-7


def test_cos():
    ctx = create_context("test_cos")
    a = NumericSeries([0.0, math.pi, 2 * math.pi], ctx)
    c = cos(a)
    result = c.data()
    assert abs(result[0] - 1.0) < 1e-12
    assert abs(result[1] - (-1.0)) < 1e-12
    assert abs(result[2] - 1.0) < 1e-12


def test_sin_cos_chain():
    ctx = create_context("test_sin_cos_chain")
    x = NumericSeries([0.0, 1.0, 2.0], ctx)
    y = sin(x) + cos(x) * 3
    assert isinstance(y, _LazyExpr)
    result = y.data()
    ref = [math.sin(v) + math.cos(v) * 3 for v in [0.0, 1.0, 2.0]]
    for r, ref_v in zip(result, ref):
        assert abs(r - ref_v) < 1e-12


def test_exp():
    ctx = create_context("test_exp")
    a = NumericSeries([0.0, 1.0, 2.0], ctx)
    c = exp(a)
    result = c.data()
    assert abs(result[0] - 1.0) < 1e-12
    assert abs(result[1] - math.e) < 1e-12
    assert abs(result[2] - math.e ** 2) < 1e-12


def test_log():
    ctx = create_context("test_log")
    a = NumericSeries([1.0, math.e, math.e ** 2], ctx)
    c = log(a)
    result = c.data()
    assert abs(result[0] - 0.0) < 1e-7
    assert abs(result[1] - 1.0) < 1e-7
    assert abs(result[2] - 2.0) < 1e-7


def test_sqrt():
    ctx = create_context("test_sqrt")
    a = NumericSeries([0.0, 4.0, 9.0], ctx)
    c = sqrt(a)
    result = c.data()
    assert abs(result[0] - 0.0) < 1e-12
    assert abs(result[1] - 2.0) < 1e-12
    assert abs(result[2] - 3.0) < 1e-12


def test_lazy_with_stats():
    ctx = create_context("test_lazy_stats")
    from Series._lib import make_series
    from Stats.Stats import total, mean
    a = make_series([1.0, 2.0, 3.0], ctx)
    b = make_series([4.0, 5.0, 6.0], ctx)
    c = a + b  # [5, 7, 9]
    assert abs(total(c) - 21.0) < 0.001
    assert abs(mean(c) - 7.0) < 0.001


def test_lazy_complex_expression():
    ctx = create_context("test_lazy_complex")
    x = NumericSeries([0.5, 1.0, 1.5], ctx)
    y = sin(x) ** 2 + cos(x) ** 2
    result = y.data()
    for v in result:
        assert abs(v - 1.0) < 1e-12


def test_lazy_chain():
    ctx = create_context("test_lazy_chain")
    a = NumericSeries([1.0, 2.0], ctx)
    b = NumericSeries([3.0, 4.0], ctx)
    c = NumericSeries([5.0, 6.0], ctx)
    r = (a + b) * c
    assert isinstance(r, _LazyExpr)
    assert r.data() == [20.0, 36.0]  # (1+3)*5=20, (2+4)*6=36


def test_numfast_import():
    import numfast as nf
    assert hasattr(nf, 'sin')
    assert hasattr(nf, 'cos')
    assert hasattr(nf, 'exp')
    assert hasattr(nf, 'log')
    assert hasattr(nf, 'sqrt')
    assert hasattr(nf, 'neg')
    assert hasattr(nf, 'NumericSeries')
    assert hasattr(nf, 'create_context')
    assert hasattr(nf, 'total')
    assert hasattr(nf, 'mean')


def test_numfast_usage():
    import numfast as nf
    ctx = nf.create_context("test_nf_usage")
    x = nf.NumericSeries([0.0, 1.0, 2.0], ctx)
    y = nf.sin(x) + nf.cos(x) * 3
    result = y.data()
    ref = [math.sin(v) + math.cos(v) * 3 for v in [0.0, 1.0, 2.0]]
    for r, ref_v in zip(result, ref):
        assert abs(r - ref_v) < 1e-12
