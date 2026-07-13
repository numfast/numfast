"""LogicalAnd -- CPU: out[i] = a[i] & b[i] (uint32)."""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    a = ctx.inputs[0].view
    b = ctx.inputs[1].view
    dst = ctx.outputs[0].view
    n = dst.length()
    for i in range(n):
        dst.write(i, a.read(i) & b.read(i))
