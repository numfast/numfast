# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import math

from _core.context import create_context
from Series._lib import NumericSeries, make_series
from Series._lib.expr import _LazyExpr
from Series._lib.math_ops import sin, cos, exp, log, sqrt
from Series._lib.executor import _plan, execute


def test_plan_simple_add():
    ctx = create_context("test_plan_add")
    a = make_series([1.0, 2.0], ctx)
    b = make_series([3.0, 4.0], ctx)
    expr = a + b
    plan = _plan(expr)
    assert len(plan) == 1  # one add op
    assert plan[0][0] == 'add'


def test_plan_chain():
    ctx = create_context("test_plan_chain")
    a = make_series([1.0, 2.0], ctx)
    b = make_series([3.0, 4.0], ctx)
    expr = (a + b) * a
    plan = _plan(expr)
    assert len(plan) == 2  # add then mul
    assert plan[0][0] == 'add'
    assert plan[1][0] == 'mul'


def test_plan_sin_cos():
    ctx = create_context("test_plan_sin_cos")
    x = make_series([0.0, 1.0], ctx)
    expr = sin(x) + cos(x)
    plan = _plan(expr)
    assert len(plan) == 3  # sin, cos, add
    names = [p[0] for p in plan]
    assert names == ['sin', 'cos', 'add']


def test_execute_simple_add():
    ctx = create_context("test_exec_add")
    a = make_series([1.0, 2.0, 3.0], ctx)
    b = make_series([4.0, 5.0, 6.0], ctx)
    expr = a + b
    result = execute(expr)
    assert result == [5.0, 7.0, 9.0]


def test_execute_sin_cos_chain():
    ctx = create_context("test_exec_sin_cos")
    x = make_series([0.0, 1.0, 2.0], ctx)
    expr = sin(x) + cos(x) * 3
    result = execute(expr)
    ref = [math.sin(v) + math.cos(v) * 3 for v in [0.0, 1.0, 2.0]]
    for r, ref_v in zip(result, ref):
        assert abs(r - ref_v) < 1e-6


def test_execute_complex():
    ctx = create_context("test_exec_complex")
    x = make_series([0.5, 1.0, 1.5], ctx)
    expr = sin(x) ** 2 + cos(x) ** 2
    result = execute(expr)
    for v in result:
        assert abs(v - 1.0) < 1e-6


def test_execute_with_scalar():
    ctx = create_context("test_exec_scalar")
    a = make_series([1.0, 2.0, 3.0], ctx)
    expr = a * 2 + 1
    result = execute(expr)
    assert result == [3.0, 5.0, 7.0]


def test_execute_scalar_radd():
    ctx = create_context("test_exec_radd")
    a = make_series([1.0, 2.0, 3.0], ctx)
    expr = 10 - a
    result = execute(expr)
    assert result == [9.0, 8.0, 7.0]


def test_compute_returns_numeric_series():
    ctx = create_context("test_compute_ns")
    a = make_series([1.0, 2.0], ctx)
    b = make_series([3.0, 4.0], ctx)
    expr = a + b
    result = expr.compute()
    assert isinstance(result, NumericSeries)
    assert result.data() == [4.0, 6.0]


def test_lazy_with_stats_via_executor():
    ctx = create_context("test_lazy_stats_exec")
    from Stats.Stats import total, mean
    a = make_series([1.0, 2.0, 3.0], ctx)
    b = make_series([4.0, 5.0, 6.0], ctx)
    c = a + b
    # Stats calls .data() via __array__ which uses Python loops
    t = total(c)
    assert abs(t - 21.0) < 0.001
    m = mean(c)
    assert abs(m - 7.0) < 0.001
