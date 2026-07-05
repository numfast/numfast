"""EMA kernel — CPU implementation."""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    period = ctx.uniforms["period"]
    multiplier = 2.0 / (period + 1.0)
    for i in range(src.length()):
        if i == 0:
            dst.write(i, src.read(i))
        else:
            prev = dst.read(i - 1)
            val = (src.read(i) - prev) * multiplier + prev
            dst.write(i, val)
