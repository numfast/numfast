"""Bitonic Sort CPU — one substage pass."""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu_sort(ctx: ExecutionContext):
    """One substage of bitonic merge.

    Each thread pair (i, i ^ substage) compares and swaps if the
    ordering is wrong for the current stage direction.

    Input: data array
    Output: same array, partially sorted
    Uniforms: stage, substage
    """
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    n = src.length()
    stage = ctx.uniforms.get("stage", 1)
    substage = ctx.uniforms.get("substage", 1)

    # Copy input to output
    for i in range(n):
        dst.write(i, src.read(i))

    # Bitonic merge
    dist = 1 << (substage - 1)      # compare distance
    mask = 1 << stage                # direction mask

    for i in range(n):
        j = i ^ dist  # XOR with distance bit
        if j > i and j < n:
            a = dst.read(i)
            b = dst.read(j)
            ascending = (i & mask) == 0
            if ascending:
                if a > b:
                    dst.write(i, b)
                    dst.write(j, a)
            else:
                if a < b:
                    dst.write(i, b)
                    dst.write(j, a)
