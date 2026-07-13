"""Fill — CPU reference: out[i] = scalar."""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    dst = ctx.outputs[0].view
    scalar = ctx.uniforms.get("scalar", 0.0)
    n = dst.length()
    for i in range(n):
        dst.write(i, scalar)
