"""MapBinary — CPU reference.

Input count matches descriptor: 1 input when one operand is scalar, 2 when both are arrays.
Scalar mode ignores the corresponding array value and uses uniform.
"""

from Runtime._lib.mod_iface import ExecutionContext

_OPS = {
    0: lambda a, b: a + b,
    1: lambda a, b: a - b,
    2: lambda a, b: a * b,
    3: lambda a, b: a / b if b != 0 else 0.0,
    4: lambda a, b: max(a, b),
    5: lambda a, b: min(a, b),
}


def cpu(ctx: ExecutionContext):
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
    for i in range(n):
        a_val = scalar_a if use_sa else a_view.read(i)
        b_val = scalar_b if use_sb else b_view.read(i)
        dst.write(i, func(a_val, b_val))
