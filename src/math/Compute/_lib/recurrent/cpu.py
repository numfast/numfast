"""Recurrent — CPU reference.

Mode 0 (single): out[i] = a*in[i] + b*out[i-1]
Mode 1 (dual):   out0[i] = a*in0[i] + b*out0[i-1]
                 out1[i] = a*in1[i] + b*out1[i-1]
"""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    mode = int(ctx.uniforms.get("mode", 0))
    a = ctx.uniforms.get("a", 1.0)
    b = ctx.uniforms.get("b", 0.0)

    if mode == 0:
        src = ctx.inputs[0].view
        dst = ctx.outputs[0].view
        state = ctx.workspace[0].view
        n = src.length()
        prev = 0.0
        for i in range(n):
            val = a * src.read(i) + b * prev
            dst.write(i, val)
            state.write(i, val)
            prev = val

    elif mode == 1:
        in0 = ctx.inputs[0].view
        in1 = ctx.inputs[1].view
        out0 = ctx.outputs[0].view
        out1 = ctx.outputs[1].view
        state0 = ctx.workspace[0].view
        state1 = ctx.workspace[1].view
        n = in0.length()
        prev0 = 0.0
        prev1 = 0.0
        for i in range(n):
            val0 = a * in0.read(i) + b * prev0
            val1 = a * in1.read(i) + b * prev1
            out0.write(i, val0)
            out1.write(i, val1)
            state0.write(i, val0)
            state1.write(i, val1)
            prev0 = val0
            prev1 = val1
