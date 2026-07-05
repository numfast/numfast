"""SMA kernel — CPU implementation."""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    period = ctx.uniforms["period"]

    window = []
    for i in range(src.length()):
        window.append(src.read(i))
        if len(window) > period:
            window.pop(0)
        if i < period - 1:
            dst.write(i, src.read(i))
        else:
            dst.write(i, sum(window) / len(window))
