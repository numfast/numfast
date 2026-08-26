"""FFT CPU — iterative Cooley-Tukey reference."""

from Runtime._lib.mod_iface import ExecutionContext
import math


def _bit_rev(x, bits):
    """Reverse bits of x."""
    y = 0
    for _ in range(bits):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


def cpu_bitreverse(ctx: ExecutionContext):
    """Bit-reversal permutation of complex array.

    Input: complex interleaved (2N floats)
    Output: bit-reversed order
    """
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    N = ctx.uniforms.get("N", 1)
    bits = int(math.log2(N))

    for i in range(N):
        j = _bit_rev(i, bits)
        if j > i:
            # Swap real + imag
            ri = src.read(2 * i)
            ii = src.read(2 * i + 1)
            rj = src.read(2 * j)
            ij = src.read(2 * j + 1)
            dst.write(2 * i, rj)
            dst.write(2 * i + 1, ij)
            dst.write(2 * j, ri)
            dst.write(2 * j + 1, ii)
        elif j == i:
            dst.write(2 * i, src.read(2 * i))
            dst.write(2 * i + 1, src.read(2 * i + 1))


def cpu_stage(ctx: ExecutionContext):
    """One Cooley-Tukey butterfly stage.

    Input: complex array (2N floats)
    Output: complex array after one butterfly stage
    Uniforms: stage (0..log2N-1), N
    """
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    N = ctx.uniforms.get("N", 1)
    stage = ctx.uniforms.get("stage", 0)

    stride = 1 << (stage + 1)    # butterfly span
    half = 1 << stage            # elements per group half

    total_butterflies = N // 2

    for bf in range(total_butterflies):
        group = bf // half
        k_in_group = bf % half
        k = group * stride + k_in_group
        j = k + half

        # Twiddle factor: exp(-2πi * k_in_group / stride)
        angle = -2.0 * math.pi * k_in_group / stride
        w_re = math.cos(angle)
        w_im = math.sin(angle)

        # Read
        a_re = src.read(2 * k)
        a_im = src.read(2 * k + 1)
        b_re = src.read(2 * j)
        b_im = src.read(2 * j + 1)

        # Butterfly: a + w*b, a - w*b
        wb_re = w_re * b_re - w_im * b_im
        wb_im = w_re * b_im + w_im * b_re

        dst.write(2 * k, a_re + wb_re)
        dst.write(2 * k + 1, a_im + wb_im)
        dst.write(2 * j, a_re - wb_re)
        dst.write(2 * j + 1, a_im - wb_im)
