# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from _core.context import create_context, context_id, check_context_compatible
from Series._lib import make_series, series_len, series_add, series_sub, series_data_proxy
from _core.backend import set_active


def test_context_create_returns_dict():
    ctx = create_context("test")
    assert isinstance(ctx, dict)
    assert "_id" in ctx
    assert "_name" in ctx
    assert ctx["_name"] == "test"


def test_context_id_is_unique():
    ctx_a = create_context("a")
    ctx_b = create_context("b")
    assert context_id(ctx_a) != context_id(ctx_b)


def test_context_name():
    ctx = create_context("myctx")
    assert ctx["_name"] == "myctx"


def test_check_context_compatible_passes():
    set_active("numpy")
    ctx = create_context()
    s1 = make_series([1, 2], ctx)
    s2 = make_series([3, 4], ctx)
    check_context_compatible(s1, s2)


def test_check_context_compatible_raises():
    set_active("numpy")
    ctx_a = create_context("a")
    ctx_b = create_context("b")
    s1 = make_series([1], ctx_a)
    s2 = make_series([2], ctx_b)
    import pytest
    with pytest.raises(TypeError, match="Cannot mix"):
        check_context_compatible(s1, s2)


def test_same_context_series_add():
    set_active("numpy")
    ctx = create_context("main")
    s1 = make_series([1, 2, 3], ctx)
    s2 = make_series([10, 20, 30], ctx)
    r = series_add(s1, s2)
    assert series_len(r) == 3
    assert series_data_proxy(r) == [11, 22, 33]


def test_same_context_series_add_chained():
    set_active("numpy")
    ctx = create_context("main")
    s1 = make_series([1, 2, 3], ctx)
    s2 = make_series([10, 20, 30], ctx)
    s3 = make_series([100, 200, 300], ctx)
    r = series_add(series_add(s1, s2), s3)
    assert series_data_proxy(r) == [111, 222, 333]


def test_different_contexts_raise_on_add():
    set_active("numpy")
    ctx_a = create_context("alpha")
    ctx_b = create_context("beta")
    s1 = make_series([1, 2, 3], ctx_a)
    s2 = make_series([4, 5, 6], ctx_b)
    import pytest
    with pytest.raises(TypeError):
        series_add(s1, s2)


def test_different_contexts_raise_on_sub():
    set_active("numpy")
    ctx_a = create_context("alpha")
    ctx_b = create_context("beta")
    s1 = make_series([10, 20], ctx_a)
    s2 = make_series([1, 2], ctx_b)
    import pytest
    with pytest.raises(TypeError):
        series_sub(s1, s2)
