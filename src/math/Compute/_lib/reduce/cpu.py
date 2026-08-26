"""Reduce kernel — CPU реализация.

Сворачивает все элементы входного массива по op:
"sum" (0) / "min" (1) / "max" (2).
Принимает op как строку или числовой код (см. descriptor).
Reference implementation — чистый Python, без оптимизаций.

S62 (defense-in-depth): N=0 — ВСЕ op (sum/min/max) → ValueError
(тот же guard, что и в descriptor.output_size_fn, для прямых вызовов
cpu() вне Runtime; единая семантика, решение пользователя 2026-08-20).
S63: min/max → IEEE (np.minimum/np.maximum, по прецеденту S29 Scan /
S55 MapBinary); sum остаётся a + b.
"""

import numpy as np

from Runtime._lib.mod_iface import ExecutionContext

_EMPTY_MSG = (
    "Reduce on empty input (n=0) is undefined; "
    "use Fill or check input length")


def _max_ieee(a, b):
    """IEEE 754-2019 maximum: NaN propagate, max(-0.0, +0.0) = +0.0.

    np.maximum propagates NaN but returns the second operand's zero sign
    for mixed zeros (numpy 2.5.1: max(+0.0,-0.0) = -0.0). IEEE requires
    max of zeros to be +0.0 unless both are -0.0 - corrected here.
    """
    res = np.maximum(a, b)
    if res == 0.0 and a == 0.0 and b == 0.0:
        if np.signbit(a) and np.signbit(b):
            return -0.0
        return +0.0
    return res


def _min_ieee(a, b):
    """IEEE 754-2019 minimum: NaN propagate, min(-0.0, +0.0) = -0.0.

    np.minimum propagates NaN but returns the second operand's zero sign
    for mixed zeros (numpy 2.5.1: min(-0.0,+0.0) = +0.0). IEEE requires
    min of zeros to be -0.0 unless both are +0.0 - corrected here.
    """
    res = np.minimum(a, b)
    if res == 0.0 and a == 0.0 and b == 0.0:
        if np.signbit(a) or np.signbit(b):
            return -0.0
        return +0.0
    return res


_OPS = {
    "sum": lambda a, b: a + b,
    "min": _min_ieee,
    "max": _max_ieee,
}
_CODES = {0: "sum", 1: "min", 2: "max"}


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    n = src.length()

    op = ctx.uniforms.get("op", "sum")
    if not isinstance(op, str):
        op = _CODES.get(op)
        if op is None:
            raise ValueError(f"Unknown Reduce op code: {ctx.uniforms.get('op')}")
    if op not in _OPS:
        raise ValueError(f"Unknown Reduce op: {op}")

    if n == 0:
        raise ValueError(_EMPTY_MSG)

    func = _OPS[op]
    total = src.read(0)
    for i in range(1, n):
        total = func(total, src.read(i))

    dst.write(0, total)