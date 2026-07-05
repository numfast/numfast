"""Momentum kernel — price change over period."""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    period = ctx.uniforms["period"]
    n = src.length()

    for i in range(n):
        if i < period:
            dst.write(i, 0.0)
        else:
            dst.write(i, src.read(i) - src.read(i - period))
