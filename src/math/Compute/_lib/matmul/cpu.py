"""MatMul CPU — naive triple loop."""

from Runtime._lib.mod_iface import ExecutionContext


def cpu_matmul(ctx: ExecutionContext):
    """C = A × B (row-major flatten).

    Input: A (M×K), B (K×N)
    Output: C (M×N)
    Uniforms: M, N, K
    """
    M, N, K = int(ctx.uniforms["M"]), int(ctx.uniforms["N"]), int(ctx.uniforms["K"])
    if M < 1 or N < 1 or K < 1:
        raise ValueError(f"MatMul M/N/K must be >= 1; got M={M}, N={N}, K={K}")
    A = ctx.inputs[0].view
    B = ctx.inputs[1].view
    C = ctx.outputs[0].view

    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += A.read(i * K + k) * B.read(k * N + j)
            C.write(i * N + j, acc)
