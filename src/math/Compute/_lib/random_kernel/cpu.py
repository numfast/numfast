"""RandomKernel CPU oracle — тот же PCG-u32 хеш на np.uint32.

uniform/integers — бит-в-бит с WGSL (арифметика хеша u32 WRAP, значения f32).
normal — тот же Box-Muller в float32 (transcendentals: tolerance-домен §5).
"""

import numpy as np

from Runtime._lib.mod_iface import ExecutionContext

_C1 = np.uint32(747796405)
_C2 = np.uint32(2891336453)
_C3 = np.uint32(277803737)
_SH = np.uint32(28)
_ADD = np.uint32(4)
_TWO_POW_NEG32 = np.float32(2.3283064365386963e-10)        # 2^-32
_TWO_POW_NEG33 = np.float32(1.16415321826934814453125e-10)  # 2^-33
_TWO_PI = np.float32(6.2831853071795864769)


def _pcg_hash(state):
    """3 раунда XOR-shift + multiply, всё в np.uint32 (WRAP)."""
    for _ in range(3):
        state = (state >> ((state >> _SH) + _ADD)) ^ (state * _C3)
    return state


def _hash_stream(counters, seed):
    """counters (np.uint32 array) -> f32 u01 in [0, 1)."""
    state = np.uint32(seed) ^ (counters * _C1 + _C2)
    h = _pcg_hash(state)
    return h.astype(np.float32) * _TWO_POW_NEG32


def cpu(ctx: ExecutionContext):
    dst = ctx.outputs[0].view
    n = int(ctx.uniforms["n"])
    mode = int(ctx.uniforms["mode"])
    seed = int(ctx.uniforms["seed"]) & 0xFFFFFFFF
    p0 = np.float32(ctx.uniforms["p0"])
    p1 = np.float32(ctx.uniforms["p1"])

    counters = np.arange(n, dtype=np.uint32)
    u1 = _hash_stream(counters, seed)

    if mode == 0:
        # uniform(low=p0, high=p1): p0 + (p1-p0)*u01
        values = p0 + (p1 - p0) * u1
    elif mode == 1:
        # int_range [p0, p1): floor(...), clamp в [p0, max(p0, p1-1)]
        raw = np.floor(p0 + (p1 - p0) * u1).astype(np.float32)
        hi = np.maximum(p0, p1 - np.float32(1.0))
        values = np.clip(raw, p0, hi)
    else:
        # normal(loc=p0, scale=p1): Box-Muller из хешей счётчиков (i, i+n)
        u2 = _hash_stream((counters + np.uint32(n)).astype(np.uint32), seed)
        u1g = np.where(u1 <= np.float32(0.0), _TWO_POW_NEG33, u1)
        r = np.sqrt(np.float32(-2.0) * np.log(u1g))
        theta = _TWO_PI * u2
        z = r * np.cos(theta)
        values = p0 + p1 * z

    for i in range(values.shape[0]):
        dst.write(i, float(values[i]))
