"""ROC kernel — Rate of Change (%)."""

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
            prev = src.read(i - period)
            if prev != 0:
                dst.write(i, (src.read(i) - prev) / prev * 100.0)
            else:
                dst.write(i, 0.0)
