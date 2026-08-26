# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Lazy expression tree for deferred computation.

``a + b``, ``nf.sin(a)`` — return LazyExpr nodes.
Evaluation happens on ``.data()``, ``.compute()``, or when passed to Stats.
"""

import math as _math


def _eval_operand(op):
    """Reduce operand to a Python list or scalar."""
    if isinstance(op, _LazyExpr):
        return op._eval()
    if isinstance(op, list):
        return op
    if isinstance(op, (int, float)):
        return op
    # NumericSeries or other object with .data()
    if hasattr(op, 'data') and callable(op.data):
        return op.data()
    raise TypeError(f"Cannot evaluate operand of type {type(op).__name__}")


# Comparison semantics mirror Compute L1 Compare primitive:
# gt/ge/lt/le/eq/ne -> 0.0/1.0 mask (see math/Compute/_lib/compare).
_CMP_PY = {
    'gt': lambda x, y: x > y,
    'ge': lambda x, y: x >= y,
    'lt': lambda x, y: x < y,
    'le': lambda x, y: x <= y,
    'eq': lambda x, y: x == y,
    'ne': lambda x, y: x != y,
}


def _execute_cmp(cmp_op, *args):
    """Execute a comparison on evaluated operands -> 0.0/1.0 list."""
    fn = _CMP_PY.get(cmp_op)
    if fn is None:
        raise ValueError(f"Unknown comparison: {cmp_op}")
    a, b = args
    if isinstance(a, (int, float)):
        return [1.0 if fn(a, v) else 0.0 for v in b]
    if isinstance(b, (int, float)):
        return [1.0 if fn(v, b) else 0.0 for v in a]
    return [1.0 if fn(x, y) else 0.0 for x, y in zip(a, b)]


def _execute_op(op, *args):
    """Execute a single operation on evaluated operands."""
    if op == 'add':
        a, b = args
        if isinstance(a, (int, float)):
            return [a + v for v in b]
        if isinstance(b, (int, float)):
            return [v + b for v in a]
        return [x + y for x, y in zip(a, b)]

    if op == 'sub':
        a, b = args
        if isinstance(a, (int, float)):
            return [a - v for v in b]
        if isinstance(b, (int, float)):
            return [v - b for v in a]
        return [x - y for x, y in zip(a, b)]

    if op == 'mul':
        a, b = args
        if isinstance(a, (int, float)):
            return [a * v for v in b]
        if isinstance(b, (int, float)):
            return [v * b for v in a]
        return [x * y for x, y in zip(a, b)]

    if op == 'truediv':
        a, b = args
        if isinstance(a, (int, float)):
            return [a / v for v in b]
        if isinstance(b, (int, float)):
            return [v / b for v in a]
        return [x / y for x, y in zip(a, b)]

    if op == 'mod':
        # C-style fmod (trunc semantics, WGSL % parity); b == 0 -> 0.0
        a, b = args
        if isinstance(a, (int, float)):
            return [_math.fmod(a, v) if v != 0 else 0.0 for v in b]
        if isinstance(b, (int, float)):
            return [_math.fmod(v, b) if b != 0 else 0.0 for v in a]
        return [_math.fmod(x, y) if y != 0 else 0.0
                for x, y in zip(a, b)]

    if op == 'pow':
        a, b = args
        if isinstance(a, (int, float)):
            return [a ** v for v in b]
        if isinstance(b, (int, float)):
            return [v ** b for v in a]
        return [x ** y for x, y in zip(a, b)]

    if op == 'neg':
        return [-v for v in args[0]]

    if op == 'abs':
        return [abs(v) for v in args[0]]

    if op == 'square':
        return [v * v for v in args[0]]

    if op in ('sin', 'cos', 'tan', 'exp', 'log', 'sqrt'):
        fn = getattr(_math, op)
        return [fn(v) for v in args[0]]

    raise ValueError(f"Unknown operation: {op}")


class _LazyExpr:
    """A node in the deferred expression DAG.

    Created by arithmetic operations (``+``, ``-``, ``*``, ``/``)
    and by math functions (``sin``, ``cos``, …).

    Args:
        op: operation name (``'add'``, ``'sin'``, …)
        *operands: child nodes or ``NumericSeries`` or scalars
    """

    def __init__(self, op, *operands, cmp_op=None):
        self.op = op
        self.operands = operands
        # For op == 'cmp': which comparison (gt/ge/lt/le/eq/ne).
        # Encoded as plan-level op name "cmp_<kind>" by the executor.
        self.cmp_op = cmp_op

    # ── Identity semantics preserved despite __eq__ ───────────────────
    __hash__ = object.__hash__

    # ── Arithmetic ────────────────────────────────────────────────────

    def __add__(self, other):
        if isinstance(other, _LazyExpr):
            return _LazyExpr('add', self, other)
        if hasattr(other, 'data') and callable(other.data):
            return _LazyExpr('add', self, other)
        if isinstance(other, (int, float)):
            return _LazyExpr('add', self, other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return _LazyExpr('add', other, self)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, _LazyExpr):
            return _LazyExpr('sub', self, other)
        if hasattr(other, 'data') and callable(other.data):
            return _LazyExpr('sub', self, other)
        if isinstance(other, (int, float)):
            return _LazyExpr('sub', self, other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return _LazyExpr('sub', other, self)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, _LazyExpr):
            return _LazyExpr('mul', self, other)
        if hasattr(other, 'data') and callable(other.data):
            return _LazyExpr('mul', self, other)
        if isinstance(other, (int, float)):
            return _LazyExpr('mul', self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return _LazyExpr('mul', other, self)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, _LazyExpr):
            return _LazyExpr('truediv', self, other)
        if hasattr(other, 'data') and callable(other.data):
            return _LazyExpr('truediv', self, other)
        if isinstance(other, (int, float)):
            return _LazyExpr('truediv', self, other)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return _LazyExpr('truediv', other, self)
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, _LazyExpr):
            return _LazyExpr('mod', self, other)
        if hasattr(other, 'data') and callable(other.data):
            return _LazyExpr('mod', self, other)
        if isinstance(other, (int, float)):
            return _LazyExpr('mod', self, other)
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return _LazyExpr('pow', self, other)
        return NotImplemented

    def __neg__(self):
        return _LazyExpr('neg', self)

    # ── Comparisons -> 0/1 mask expression (Compare primitive, L1) ────

    def _cmp(self, other, cmp_op):
        if isinstance(other, _LazyExpr):
            return _LazyExpr('cmp', self, other, cmp_op=cmp_op)
        if hasattr(other, 'data') and callable(other.data):
            return _LazyExpr('cmp', self, other, cmp_op=cmp_op)
        if isinstance(other, (int, float)):
            return _LazyExpr('cmp', self, other, cmp_op=cmp_op)
        return NotImplemented

    def __lt__(self, other):
        return self._cmp(other, 'lt')

    def __le__(self, other):
        return self._cmp(other, 'le')

    def __gt__(self, other):
        return self._cmp(other, 'gt')

    def __ge__(self, other):
        return self._cmp(other, 'ge')

    def __eq__(self, other):
        return self._cmp(other, 'eq')

    def __ne__(self, other):
        return self._cmp(other, 'ne')

    # ── Filter composition ────────────────────────────────────────────

    def filter(self, mask):
        """Condition-side filter: ``(s > 0.5).filter(s)``.

        Computes this comparison into a 0/1 mask, then keeps values of
        ``mask``'s series where the mask is nonzero (core Filter
        extension: Compare+Scan+Gather composition), stable order.
        """
        return mask.filter(self.compute())

    def __iter__(self):
        return iter(self.data())

    def __len__(self):
        return len(self.data())

    # ── Evaluation ────────────────────────────────────────────────────

    def data(self):
        """Evaluate the expression tree and return a Python list."""
        return self._eval()

    def compute(self):
        """Plan + execute via the Executor (numpy-vectorized), return NumericSeries.

        Unlike ``.data()`` which uses Python loops, ``.compute()``
        flattens the DAG and executes via vectorized numpy ops.
        """
        from .executor import execute
        from .numeric_series import NumericSeries
        result = execute(self)
        ctx_id = None
        for op in self.operands:
            if hasattr(op, '_proxy'):
                ctx_id = op._proxy.get("_context_id")
                break
            if hasattr(op, 'operands'):
                for sub in op.operands:
                    if hasattr(sub, '_proxy'):
                        ctx_id = sub._proxy.get("_context_id")
                        break
        from _core.context import create_context
        ctx = create_context("executor") if ctx_id is None else {"_id": ctx_id}
        return NumericSeries(result, ctx)

    def _eval(self):
        evaled = [_eval_operand(op) for op in self.operands]
        if self.op == 'cmp':
            return _execute_cmp(self.cmp_op or 'gt', *evaled)
        return _execute_op(self.op, *evaled)

    # ── Interop ───────────────────────────────────────────────────────

    def __array__(self, dtype=None):
        import numpy as np
        return np.array(self.data(), dtype=dtype)

    def __repr__(self):
        op = self.op
        n = len(self.operands)
        if n == 1:
            return f"<{op}({self.operands[0]})>"
        if n == 2:
            return f"<{self.operands[0]} {op} {self.operands[1]}>"
        return f"<{op}{self.operands}>"
