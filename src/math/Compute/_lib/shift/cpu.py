"""Shift -- CPU reference: out[i] = in[i - offset]."""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    offset = int(ctx.uniforms.get("offset", 1))
    n = src.length()
    for i in range(n):
        j = i - offset
        if 0 <= j < n:
            dst.write(i, src.read(j))
        else:
            dst.write(i, 0.0)