"""Gather — CPU reference: out[i] = src[index[i]]."""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    idx = ctx.inputs[1].view
    dst = ctx.outputs[0].view
    n = dst.length()
    for i in range(n):
        ix = int(idx.read(i))
        if 0 <= ix < src.length():
            dst.write(i, src.read(ix))
        else:
            dst.write(i, 0.0)