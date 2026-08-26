"""Creation fuzz: >=1000 cases vs numpy PCG-u32 oracle (master seed 42).

Grid: 5 param rows x SHAPES(10) x OPS(8) x SEEDS{42,7,1337} = 1200 cases.
uniform/integers: bit-exact vs oracle (u32 WRAP hash, f32 arithmetic).
normal: Box-Muller transcendental domain, rel<=1e-5 abs<=1e-6.
"""

import numpy as np
import pytest

from core.Creation._lib.api import (arange, full, linspace, ones,
                                    random_integers, random_normal,
                                    random_uniform, zeros)

_F = np.float32
_U32 = np.uint32

SHAPES = [0, 1, 2, 3, 63, 64, 65, 1000, 65536, 1_000_000]
SEEDS = [42, 7, 1337]
PARAMS = [
    dict(full=-2.5, ar=(0.0, 1.0), ls=(-5.0, 5.0), uni=(-1.0, 1.0),
         nor=(0.0, 1.0), ints=(-100, 100)),
    dict(full=17.75, ar=(-3.0, 2.0), ls=(0.0, 1000.0), uni=(10.0, 20.0),
         nor=(5.0, 2.0), ints=(0, 7)),
    dict(full=0.125, ar=(99.0, -5.0), ls=(-1.0, 1.0), uni=(-100.0, -1.0),
         nor=(-50.0, 10.0), ints=(-1000, 1000)),
    dict(full=-0.0, ar=(5.0, 0.25), ls=(10.0, -10.0), uni=(-0.5, 0.5),
         nor=(1e3, 1e-3), ints=(-3, 4)),
    dict(full=1234.5, ar=(-1.0, 0.125), ls=(0.0, 1.0), uni=(0.0, 1.0),
         nor=(-1.0, 3.0), ints=(17, 18)),
]
OPS = ("zeros", "ones", "full", "arange", "linspace",
       "uniform", "normal", "integers")

_C1, _C2, _C3 = _U32(747796405), _U32(2891336453), _U32(277803737)
_SH, _AD = _U32(28), _U32(4)
_2M32 = _F(2.3283064365386963e-10)        # 2^-32
_2M33 = _F(1.16415321826934814453125e-10)  # 2^-33
_2PI = _F(6.2831853071795864769)


def _hash_stream(counters, seed):
    s = _U32(seed & 0xFFFFFFFF) ^ (counters * _C1 + _C2)
    for _ in range(3):
        s = (s >> ((s >> _SH) + _AD)) ^ (s * _C3)
    return s.astype(_F) * _2M32


def _oracle(op, n, p, seed):
    """(expected_f32, bit_exact_required)."""
    if op == "zeros":
        return np.zeros(n, _F), True
    if op == "ones":
        return np.ones(n, _F), True
    if op == "full":
        return np.full(n, _F(p["full"]), _F), True
    if op == "arange":
        s0, st = p["ar"]
        i = np.arange(n, dtype=_F)
        return (np.float32(s0) + np.float32(st) * i).astype(_F), True
    if op == "linspace":
        a, b = p["ls"]
        den = _F((n - 1) if n > 1 else 1)
        i = np.arange(n, dtype=_F)
        return (np.float32(a) + (np.float32(b) - np.float32(a)) * i / den), False
    u1 = _hash_stream(np.arange(n, dtype=_U32), seed)
    if op == "uniform":
        lo, hi = p["uni"]
        return np.float32(lo) + (np.float32(hi) - np.float32(lo)) * u1, True
    if op == "integers":
        lo, hi = p["ints"]
        raw = np.floor(np.float32(lo)
                       + (np.float32(hi) - np.float32(lo)) * u1).astype(_F)
        hi_c = np.maximum(np.float32(lo), np.float32(hi) - _F(1.0))
        return np.clip(raw, np.float32(lo), hi_c).astype(np.int32), True
    loc, scale = p["nor"]
    u2 = _hash_stream((np.arange(n, dtype=_U32) + _U32(n)).astype(_U32), seed)
    u1g = np.where(u1 <= _F(0.0), _2M33, u1)
    r = np.sqrt(_F(-2.0) * np.log(u1g))
    z = r * np.cos(_2PI * u2)
    return np.float32(loc) + np.float32(scale) * z, False


def _call(op, shape, p, seed):
    if op == "zeros":
        return zeros(shape)
    if op == "ones":
        return ones(shape)
    if op == "full":
        return full(shape, p["full"])
    if op == "arange":
        s0, st = p["ar"]
        return arange(s0, s0 + st * shape, st)
    if op == "linspace":
        a, b = p["ls"]
        return linspace(a, b, shape, endpoint=True)
    if op == "uniform":
        lo, hi = p["uni"]
        return random_uniform(lo, hi, shape=(shape,), seed=seed)
    if op == "normal":
        loc, scale = p["nor"]
        return random_normal(loc, scale, shape=(shape,), seed=seed)
    lo, hi = p["ints"]
    return random_integers(lo, hi, shape=(shape,), seed=seed)


def test_fuzz_grid_1200_cases():
    mismatches = []
    total = 0
    for r, p in enumerate(PARAMS):
        for op in OPS:
            for shape in SHAPES:
                for seed in SEEDS:
                    total += 1
                    got = _call(op, shape, p, seed)
                    assert len(got) == shape
                    exp, exact = _oracle(op, shape, p, seed)
                    val = got.to_numpy()
                    ok = (np.array_equal(val, exp.astype(val.dtype))
                          if exact else
                          np.allclose(val, exp, rtol=1e-5, atol=1e-6))
                    if not ok:
                        mismatches.append(f"r{r}/{op}/n={shape}/seed={seed}")
                    got.free()
    assert total >= 1000, f"grid too small: {total}"
    assert mismatches == [], f"n_mismatches={len(mismatches)}: {mismatches[:8]}"
