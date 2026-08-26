# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Math functions that return lazy expression nodes.

Usage::

    from numfast import sin, cos
    y = sin(x) + cos(x) * 3
    result = y.data()   # or pass to Stats, or .compute()
"""

from .expr import _LazyExpr


def sin(x):
    """Element-wise sine. Returns a lazy expression node."""
    return _LazyExpr('sin', x)


def cos(x):
    """Element-wise cosine. Returns a lazy expression node."""
    return _LazyExpr('cos', x)


def tan(x):
    """Element-wise tangent. Returns a lazy expression node."""
    return _LazyExpr('tan', x)


def exp(x):
    """Element-wise exponential. Returns a lazy expression node."""
    return _LazyExpr('exp', x)


def log(x):
    """Element-wise natural logarithm. Returns a lazy expression node."""
    return _LazyExpr('log', x)


def sqrt(x):
    """Element-wise square root. Returns a lazy expression node."""
    return _LazyExpr('sqrt', x)


def neg(x):
    """Element-wise negation. Returns a lazy expression node."""
    return _LazyExpr('neg', x)


def abs(x):
    """Element-wise absolute value. Returns a lazy expression node."""
    return _LazyExpr('abs', x)


def square(x):
    """Element-wise square (x * x). Returns a lazy expression node."""
    return _LazyExpr('square', x)
