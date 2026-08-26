"""MapBinary — CPU reference.

Input count matches descriptor: 1 input when one operand is scalar, 2 when both are arrays.
Scalar mode ignores the corresponding array value and uses uniform.

dtype-семантика (решение владельца): uniforms int_mode=1 -> целочисленный путь
i32: add/sub/mul WRAP (two's complement), mod = C-trunc % guard b==0 -> 0,
скаляры — i32 (trunc). Векторизовано через BlockView._raw (bulk-доступ,
прецедент walkforward_full_fused/cpu).
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


def _fmod_c(a, b):
    """C-style fmod: a - trunc(a/b)*b (WGSL % semantics), NOT floor-mod.

    np.fmod is trunc semantics (matches WGSL %); np.mod is floor semantics.
    Guard mirrors div (S50): b == 0 -> exact +0.0 (WGSL % by zero is NaN).
    """
    if b == 0:
        return 0.0
    return float(np.fmod(a, b))


_OPS = {
    0: lambda a, b: a + b,
    1: lambda a, b: a - b,
    2: lambda a, b: a * b,
    3: lambda a, b: a / b if b != 0 else 0.0,
    4: _max_ieee,
    5: _min_ieee,
    6: _fmod_c,
}


def _wrap_i32(x):
    """Wrap integer array/value to int32 domain (two's complement, WGSL parity)."""
    m = x & 0xFFFFFFFF
    return np.where(m >= 0x80000000, m - 0x100000000, m).astype(np.int32)


def _cpu_int32(ctx: ExecutionContext):
    dst = ctx.outputs[0].view
    use_sa = int(ctx.uniforms.get("use_scalar_a", 0))
    use_sb = int(ctx.uniforms.get("use_scalar_b", 0))
    scalar_a = int(ctx.uniforms.get("scalar_a", 0))
    scalar_b = int(ctx.uniforms.get("scalar_b", 0))
    op_code = int(ctx.uniforms.get("op", 0))
    if op_code not in (0, 1, 2, 6):
        raise ValueError(f"MapBinary int32: unsupported op {op_code}")

    input_idx = 0
    a_view = None if use_sa else ctx.inputs[input_idx].view
    if not use_sa:
        input_idx += 1
    b_view = None if use_sb else ctx.inputs[input_idx].view

    n = dst.length()
    if not use_sa and a_view.length() != n:
        raise ValueError(
            f"MapBinary: broadcast of 'a' (len {a_view.length()}) not supported; "
            f"n={n}; only 'b' broadcasts (len 1)"
        )

    # Exact i32 sources: raw containers hold int values exactly.
    if use_sa:
        a = np.full(n, scalar_a, dtype=np.int64)
    else:
        a = np.asarray(a_view._raw[:n], dtype=np.int64)
    if use_sb:
        b = np.full(n, scalar_b, dtype=np.int64)
    elif b_view.length() == 1:
        b = np.full(n, int(b_view._raw[0]), dtype=np.int64)
    else:
        b = np.asarray(b_view._raw[:n], dtype=np.int64)

    if op_code == 0:
        res = a + b          # WRAP via _wrap_i32 below
    elif op_code == 1:
        res = a - b          # WRAP via _wrap_i32 below
    elif op_code == 2:
        res = a * b          # WRAP via _wrap_i32 below
    else:
        # C-trunc modulo: r = a - trunc(a/b)*b, sign of dividend;
        # guard mirrors GPU path: b == 0 -> exact 0.
        bb = np.where(b == 0, 1, b)
        q = a // bb
        corr = ((a % bb) != 0) & ((a < 0) != (bb < 0))
        res = np.where(b == 0, 0, a - bb * (q + corr))

    wrapped = _wrap_i32(res)
    dst._raw[:n] = wrapped.astype(dst._raw.dtype)


def cpu(ctx: ExecutionContext):
    if int(ctx.uniforms.get("int_mode", 0)):
        return _cpu_int32(ctx)

    dst = ctx.outputs[0].view
    use_sa = int(ctx.uniforms.get("use_scalar_a", 0))
    use_sb = int(ctx.uniforms.get("use_scalar_b", 0))
    scalar_a = ctx.uniforms.get("scalar_a", 0.0)
    scalar_b = ctx.uniforms.get("scalar_b", 0.0)
    op_code = int(ctx.uniforms.get("op", 0))

    # Map input index based on scalar flags
    input_idx = 0
    a_view = None if use_sa else ctx.inputs[input_idx].view
    if not use_sa:
        input_idx += 1
    b_view = None if use_sb else ctx.inputs[input_idx].view

    func = _OPS.get(op_code)
    if func is None:
        raise ValueError(f"Unknown MapBinary op code: {op_code}")
    n = dst.length()
    if not use_sa and a_view.length() != dst.length():
        raise ValueError(
            f"MapBinary: broadcast of 'a' (len {a_view.length()}) not supported; "
            f"n={n}; only 'b' broadcasts (len 1)"
        )
    for i in range(n):
        a_val = scalar_a if use_sa else a_view.read(i)
        b_val = scalar_b if use_sb else b_view.read(i if b_view.length() > 1 else 0)
        dst.write(i, func(a_val, b_val))
