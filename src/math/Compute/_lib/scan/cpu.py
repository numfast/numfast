"""Scan CPU references — one per kernel phase.

op из ctx.uniforms: строка ("sum"/"mul"/"max"/"min") или числовой код
(0 = sum, 1 = mul, 2 = max, 3 = min). Reference: inclusive prefix scan
(для одной задачи): out[i] = op(x[0..i]).
"""

from Runtime._lib.mod_iface import ExecutionContext

import numpy as np


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
    "mul": lambda a, b: a * b,
    "max": _max_ieee,
    "min": _min_ieee,
}
_CODES = {0: "sum", 1: "mul", 2: "max", 3: "min"}
_NEUTRAL = {"sum": 0.0, "mul": 1.0, "max": -3.4028235e38, "min": 3.4028235e38}


def _op_name(ctx: ExecutionContext) -> str:
    op = ctx.uniforms.get("op", "sum")
    if not isinstance(op, str):
        op = _CODES.get(op)
        if op is None:
            raise ValueError(f"Unknown Scan op code: {ctx.uniforms.get('op')}")
    if op not in _OPS:
        raise ValueError(f"Unknown Scan op: {op}")
    return op


def cpu_local(ctx: ExecutionContext):
    """Local scan per block of 64 (inclusive: первый элемент = сам вход).

    Output 0: local scans (same length as input)
    Output 1: block totals (one per block of 64)
    """
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    sums = ctx.outputs[1].view
    func = _OPS[_op_name(ctx)]
    n = src.length()
    BLOCK = 64

    num_full_blocks = n // BLOCK
    for b in range(num_full_blocks):
        start = b * BLOCK
        total = src.read(start)
        dst.write(start, total)
        for i in range(start + 1, start + BLOCK):
            total = func(total, src.read(i))
            dst.write(i, total)
        sums.write(b, total)

    # Partial last block
    remainder = n % BLOCK
    if remainder > 0:
        start = num_full_blocks * BLOCK
        total = src.read(start)
        dst.write(start, total)
        for i in range(start + 1, n):
            total = func(total, src.read(i))
            dst.write(i, total)
        sums.write(num_full_blocks, total)


def cpu_totals(ctx: ExecutionContext):
    """Exclusive scan of block totals: sp[t] = op(sp[t-1], last[t-1]), sp[0] = neutral."""
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    op = _op_name(ctx)
    func = _OPS[op]
    n = src.length()
    if n == 0:
        return
    acc = _NEUTRAL[op]
    dst.write(0, acc)
    for i in range(1, n):
        acc = func(acc, src.read(i - 1))
        dst.write(i, acc)


def cpu_final(ctx: ExecutionContext):
    """Combine accumulated block offset with each local scan.

    final[i] = op(local[i], sp[block_id]) при block_id > 0, иначе local[i].
    """
    local = ctx.inputs[1].view
    prefix = ctx.inputs[2].view
    dst = ctx.outputs[0].view
    func = _OPS[_op_name(ctx)]
    n = local.length()
    BLOCK = 64

    for i in range(n):
        block_id = i // BLOCK
        if block_id > 0:
            dst.write(i, func(local.read(i), prefix.read(block_id)))
        else:
            dst.write(i, local.read(i))
