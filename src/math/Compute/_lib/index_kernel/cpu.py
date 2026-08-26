"""IndexKernel CPU oracle — np.full / np.arange / np.linspace / np.tile / np.repeat.

dtype-семантика (решение владельца): uniform-флаг int_mode=1 -> целочисленный
путь: значения в домене i32 (ТОЧНО до 2^31, лимит f32 2^24 не применяется),
арифметика i32 WRAP (two's complement), linspace — C-trunc деление.
Контейнер выхода (float64/int64 raw от builder) хранит i32-значения точно.
"""

import math

import numpy as np

from Runtime._lib.mod_iface import ExecutionContext


def _wrap_i32(x):
    """Wrap integer array/value to int32 domain (two's complement, WGSL parity)."""
    m = x & 0xFFFFFFFF
    return np.where(m >= 0x80000000, m - 0x100000000, m).astype(np.int32)


def _cpu_int32(ctx: ExecutionContext):
    dst = ctx.outputs[0].view
    n = int(ctx.uniforms["n"])
    mode = int(ctx.uniforms["mode"])
    p0 = int(ctx.uniforms["p0"])
    p1 = int(ctx.uniforms["p1"])
    p2 = int(ctx.uniforms["p2"])
    idx64 = np.arange(n, dtype=np.int64)

    if mode == 0:
        values = np.full(n, p0, dtype=np.int64)
    elif mode == 1:
        # arange(start=p0, step=p1) in exact int64 then wrap to i32 (WRAP).
        values = p0 + p1 * idx64
    elif mode in (3, 4):
        # tile/repeat: pattern values truncate to i32 (host-cast parity).
        pat = np.asarray([ctx.inputs[0].view.read(j)
                          for j in range(int(ctx.uniforms["k"]))],
                         dtype=np.float32).astype(np.int64)
        if mode == 3:
            values = np.tile(pat, math.ceil(n / max(len(pat), 1)))[:n]
        else:
            rep = int(ctx.uniforms["rep"])
            values = np.repeat(pat, rep)[:n]
    else:
        # linspace integer-trunc semantics: q = trunc(num/den), C-style.
        num = (p1 - p0) * idx64
        q = num // p2
        corr = (num % p2 != 0) & ((num < 0) != (p2 < 0))
        values = p0 + q + corr

    wrapped = _wrap_i32(values)
    # Bulk write into output container (walkforward_full_fused/cpu precedent):
    # container (float64/int64) holds i32 values exactly.
    dst._raw[:n] = wrapped.astype(dst._raw.dtype)


def cpu(ctx: ExecutionContext):
    if int(ctx.uniforms.get("int_mode", 0)):
        return _cpu_int32(ctx)

    dst = ctx.outputs[0].view
    n = int(ctx.uniforms["n"])
    mode = int(ctx.uniforms["mode"])
    p0 = ctx.uniforms["p0"]
    p1 = ctx.uniforms["p1"]
    p2 = ctx.uniforms["p2"]

    if mode == 0:
        # const(p0)
        values = np.full(n, p0, dtype=np.float32)
    elif mode == 1:
        # arange(start=p0, step=p1)
        values = np.arange(p0, p0 + p1 * n, p1, dtype=np.float32)[:n]
        if values.shape[0] != n:
            values = (p0 + p1 * np.arange(n, dtype=np.float64)).astype(np.float32)
    elif mode == 3:
        # tile(pattern, n): out[i] = pat[i % k]; pattern — входной буфер
        k = int(ctx.uniforms["k"])
        pat = np.asarray([ctx.inputs[0].view.read(j) for j in range(k)],
                         dtype=np.float32)
        values = np.tile(pat, math.ceil(n / k))[:n]
    elif mode == 4:
        # repeat(pattern, rep): out[i] = pat[i // rep]; pattern — входной буфер
        k = int(ctx.uniforms["k"])
        rep = int(ctx.uniforms["rep"])
        pat = np.asarray([ctx.inputs[0].view.read(j) for j in range(k)],
                         dtype=np.float32)
        values = np.repeat(pat, rep)[:n]
    else:
        # linspace(a=p0, b=p1, denom=p2); стандартный случай denom == num-1
        if p2 == float(max(n - 1, 1)):
            values = np.linspace(p0, p1, num=n, endpoint=True, dtype=np.float32)
        else:
            idx = np.arange(n, dtype=np.float64)
            values = (p0 + (p1 - p0) * idx / p2).astype(np.float32)

    for i in range(values.shape[0]):
        dst.write(i, float(values[i]))
