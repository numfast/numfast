"""Clamp — CPU reference: out = max(min(x, max_val), min_val)."""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    min_val = ctx.uniforms.get("min_val", 0.0)
    max_val = ctx.uniforms.get("max_val", 1.0)
    n = src.length()
    for i in range(n):
        x = src.read(i)
        if x < min_val:
            dst.write(i, min_val)
        elif x > max_val:
            dst.write(i, max_val)
        else:
            dst.write(i, x)
