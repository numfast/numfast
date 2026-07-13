"""MapScalar — CPU reference: out[i] = data[i] op scalar."""

from Runtime._lib.mod_iface import ExecutionContext

_OPS = {
    0: lambda x, s: x + s,   # add
    1: lambda x, s: x - s,   # sub
    2: lambda x, s: x * s,   # mul
    3: lambda x, s: x / s if s != 0 else 0.0,  # div
    4: lambda x, s: max(x, s),  # max
    5: lambda x, s: min(x, s),  # min
}


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    op_code = int(ctx.uniforms.get("op", 0))
    scalar = ctx.uniforms.get("scalar", 0.0)
    func = _OPS.get(op_code)
    if func is None:
        raise ValueError(f"Unknown MapScalar op code: {op_code}")
    n = src.length()
    for i in range(n):
        dst.write(i, func(src.read(i), scalar))
