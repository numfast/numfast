"""Rolling Sum -- CPU reference."""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    period = int(ctx.uniforms["period"])
    n = src.length()

    running_sum = 0.0
    for i in range(n):
        running_sum += src.read(i)
        if i >= period:
            running_sum -= src.read(i - period)
        if i >= period - 1:
            dst.write(i, running_sum)
        else:
            dst.write(i, 0.0)
