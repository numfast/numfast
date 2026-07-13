"""Scatter — CPU reference: out[index[i]] = value[i].

Warning: conflicting indices (same index written twice) = last write wins.
"""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    value = ctx.inputs[0].view
    idx = ctx.inputs[1].view
    dst = ctx.outputs[0].view
    n = value.length()
    dst_n = dst.length()
    for i in range(n):
        ix = int(idx.read(i))
        if 0 <= ix < dst_n:
            dst.write(ix, value.read(i))
