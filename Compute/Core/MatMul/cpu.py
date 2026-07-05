"""MatMul CPU — naive triple loop."""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu_matmul(ctx: ExecutionContext):
    """C = A × B (row-major flatten).

    Input: A (M×K), B (K×N)
    Output: C (M×N)
    Uniforms: M, N, K
    """
    A = ctx.inputs[0].view
    B = ctx.inputs[1].view
    C = ctx.outputs[0].view
    M = ctx.uniforms.get("M", 1)
    N = ctx.uniforms.get("N", 1)
    K = ctx.uniforms.get("K", 1)

    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += A.read(i * K + k) * B.read(k * N + j)
            C.write(i * N + j, acc)
