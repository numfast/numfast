"""Mask — CPU reference: out[i] = cond[i] ? a[i] : b[i].

cond is uint32 array. a and b can be arrays or scalars.
"""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    cond = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    use_sa = int(ctx.uniforms.get("use_scalar_a", 0))
    use_sb = int(ctx.uniforms.get("use_scalar_b", 0))
    scalar_a = ctx.uniforms.get("scalar_a", 0.0)
    scalar_b = ctx.uniforms.get("scalar_b", 0.0)

    # Dynamic input index: a at idx 1 if array, b follows
    idx = 1
    a_view = None if use_sa else ctx.inputs[idx].view
    if not use_sa:
        idx += 1
    b_view = None if use_sb else ctx.inputs[idx].view

    n = dst.length()
    for i in range(n):
        a_val = scalar_a if use_sa else a_view.read(i)
        b_val = scalar_b if use_sb else b_view.read(i)
        dst.write(i, a_val if cond.read(i) != 0 else b_val)