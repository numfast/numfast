"""Combine — CPU reference: linear combination of N inputs."""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    n = len(ctx.inputs)
    views = [inp.view for inp in ctx.inputs]
    dst = ctx.outputs[0].view
    weights = [ctx.uniforms.get(f"w{i}", 0.0) for i in range(4)]
    bias = ctx.uniforms.get("bias", 0.0)
    length = views[0].length()
    
    for i in range(length):
        acc = bias
        for j in range(n):
            acc += weights[j] * views[j].read(i)
        dst.write(i, acc)
