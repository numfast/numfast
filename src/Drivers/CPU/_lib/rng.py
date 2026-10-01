# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""RNG reference core: bit-exact Python mirror of numfast-native/src/rng.rs.

STANDALONE module: numpy only, no Builder/Extension
imports (NO INTERNAL IMPORTS rule). Every function is the correctness
reference for the native FFI symbol of the same name; native-available
never changes outputs, only speed (same contract as native_cpu.py).

CORE = SplitMix64-seeded Philox4x32-10, u32 ops only (WGSL-portable);
chunkable via the (seed, stream, offset) counter domain: lane i always
uses counter offset + i, so chunked == unchunked bit-exact.
R-COMPAT = R 4.3.x MT19937 (Ripley init + twist/temper + fixup +
rejection chunks), strict sequential, CPU-only (chunkable=false).
"""

import numpy as np

_M32 = np.uint32(0xFFFFFFFF)
_M64 = 0xFFFFFFFFFFFFFFFF

PHILOX_M0 = np.uint32(0xD2511F53)
PHILOX_M1 = np.uint32(0xCD9E8D57)
PHILOX_W0 = np.uint32(0x9E3779B9)
PHILOX_W1 = np.uint32(0xBB67AE85)

POW10_F64 = np.array([10.0 ** d for d in range(16)], dtype=np.float64)

ERR_OK = 0
ERR_NULL = -1
ERR_RANGE = -2
ERR_GEOM = -3


def _splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & _M64
    z = state
    z = (((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64)
    z = (((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64)
    return state, (z ^ (z >> 31)) & _M64


def _derive_key(seed, stream):
    st = ((int(seed) & _M64) + 0x9E3779B97F4A7C15
          + ((int(stream) & _M64) * 0xBF58476D1CE4E5B9 & _M64)) & _M64
    st, k = _splitmix64(st)
    return np.uint32(k & 0xFFFFFFFF), np.uint32((k >> 32) & 0xFFFFFFFF)


def _mul_hi_u32_vec(a, b):
    """Vector u32 high-mul via 16-bit halves (wraps like Rust/WGSL)."""
    a = a.astype(np.uint32, copy=False)
    b = b.astype(np.uint32, copy=False)
    a0 = a & np.uint32(0xFFFF)
    a1 = a >> np.uint32(16)
    b0 = b & np.uint32(0xFFFF)
    b1 = b >> np.uint32(16)
    p0 = a0 * b0
    p1 = a0 * b1
    p2 = a1 * b0
    p3 = a1 * b1
    mid = (p0 >> np.uint32(16)) + (p1 & np.uint32(0xFFFF)) + (p2 & np.uint32(0xFFFF))
    return p3 + (p1 >> np.uint32(16)) + (p2 >> np.uint32(16)) + (mid >> np.uint32(16))


def _philox_block(counters, k0, k1):
    """counters: (4, n) uint32; returns (4, n) uint32 after 10 rounds."""
    a0, a1, a2, a3 = (np.ascontiguousarray(c, dtype=np.uint32) for c in counters)
    kk0, kk1 = int(k0) & 0xFFFFFFFF, int(k1) & 0xFFFFFFFF
    with np.errstate(over="ignore"):
        kk0n, kk1n = np.uint32(kk0), np.uint32(kk1)
        for _ in range(10):
            hi0 = _mul_hi_u32_vec(a0, PHILOX_M0)
            lo0 = a0 * PHILOX_M0
            hi1 = _mul_hi_u32_vec(a2, PHILOX_M1)
            lo1 = a2 * PHILOX_M1
            a0, a1, a2, a3 = (hi1 ^ a1 ^ kk0n), lo1, (hi0 ^ a3 ^ kk1n), lo0
            kk0 = (kk0 + 0x9E3779B9) & 0xFFFFFFFF
            kk1 = (kk1 + 0xBB67AE85) & 0xFFFFFFFF
            kk0n, kk1n = np.uint32(kk0), np.uint32(kk1)
    return a0, a1, a2, a3


def _draw_rows(n, seed, stream, offset):
    """Raw Philox (r0..r3): (4, n) uint32, lane i = counter offset + i."""
    g = (np.arange(n, dtype=np.uint64) + (np.uint64(int(offset) & _M64))) & np.uint64(_M64)
    c0 = (g & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    c1 = ((g >> np.uint64(32)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    s = int(stream) & _M64
    c2 = np.full(n, np.uint32((s & 0xFFFFFFFF) ^ 0xC0FFEE00), dtype=np.uint32)
    c3 = np.full(n, np.uint32(((s >> 32) & 0xFFFFFFFF) ^ 0x9E3779B9), dtype=np.uint32)
    k0, k1 = _derive_key(int(seed) & _M64, s)
    return _philox_block((c0, c1, c2, c3), k0, k1)


def validate_seed(seed):
    """rng_seed contract: explicit error unless uint64 int (bool rejected)."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"rng_seed must be an int in [0, 2**64), got {seed!r}. "
            "Fix: pass seed=42 (or any uint64 int). "
            "See specs/02-semantic-ir.md"
        )
    if not 0 <= seed < 2 ** 64:
        raise ValueError(
            f"rng_seed must be an int in [0, 2**64), got {seed!r}. "
            "Fix: pass seed=42 (or any uint64 int). "
            "See specs/02-semantic-ir.md"
        )
    return int(seed)


def fill_i32(n, seed, stream=0, offset=0, lo=0, hi=100, mode=0):
    """CORE fill i32 [lo, hi). Returns (rc, array)."""
    if mode != 0:
        return ERR_GEOM, None
    if not hi > lo:
        return ERR_RANGE, None
    width = int(hi) - int(lo)
    if width <= 0:
        return ERR_RANGE, None
    if n == 0:
        return ERR_OK, np.empty(0, dtype=np.int32)
    if width == 1:
        return ERR_OK, np.full(n, np.int32(lo), dtype=np.int32)
    r0, _, _, _ = _draw_rows(n, seed, stream, offset)
    v = ((r0.astype(np.uint64) * np.uint64(width)) >> np.uint64(32)).astype(np.int64)
    return ERR_OK, np.ascontiguousarray((v + np.int64(lo)).astype(np.int32))


def fill_f64(n, seed, stream=0, offset=0, lo=0.0, hi=1.0):
    """CORE fill f64 [lo, hi). Returns (rc, array)."""
    import math as _math
    lo_f, hi_f = float(lo), float(hi)
    if not (_math.isfinite(lo_f) and _math.isfinite(hi_f)) or not lo_f < hi_f:
        return ERR_RANGE, None
    if n == 0:
        return ERR_OK, np.empty(0, dtype=np.float64)
    r0, r1, _, _ = _draw_rows(n, seed, stream, offset)
    u = ((r0 >> np.uint32(5)).astype(np.uint64) << np.uint64(26)) | (
        (r1 >> np.uint32(6)).astype(np.uint64))
    f = u.astype(np.float64) / float(2 ** 53)
    return ERR_OK, np.ascontiguousarray(lo_f + f * (hi_f - lo_f))


def map_round(x, validity=None, ndigits=0):
    """Half-even round, banker's. Returns (rc, out, out_valid)."""
    if not isinstance(ndigits, int) or isinstance(ndigits, bool):
        return ERR_RANGE, None, None
    if not 0 <= ndigits <= 15:
        return ERR_RANGE, None, None
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    n = x.size
    if validity is None:
        valid = np.ones(n, dtype=np.uint8)
    else:
        valid = np.ascontiguousarray(np.asarray(validity, dtype=np.uint8))
        if valid.size != n:
            return ERR_RANGE, None, None
    out = np.empty(n, dtype=np.float64)
    ov = np.ones(n, dtype=np.uint8)
    p = float(POW10_F64[ndigits])
    is_valid = valid != 0
    out[~is_valid] = 0.0
    ov[~is_valid] = 0
    # Non-finite passthrough (NaN stays NaN, +-Inf stays +-Inf).
    nonfin = is_valid & ~np.isfinite(x)
    out[nonfin] = x[nonfin]
    # Zeros keep their signbit (-0.0 -> -0.0).
    is_zero = is_valid & (x == 0.0)
    out[is_zero] = x[is_zero]
    # Regular lanes: scaled-int ticks comparison (exact: |y| <= 2^53).
    y = x * p
    regular = (is_valid & np.isfinite(x) & (x != 0.0)
               & np.isfinite(y) & (np.abs(y) <= 9.007199254740992e15))
    fl = np.floor(y[regular])
    d = y[regular] - fl
    tick = fl.astype(np.int64)
    t = np.where(d < 0.5, fl,
                 np.where(d > 0.5, fl + 1.0,
                          np.where((tick & np.int64(1)) == 0, fl, fl + 1.0)))
    r = t / p
    neg_zero = (t == 0.0) & np.signbit(x[regular])
    r[neg_zero] = -0.0
    out[regular] = r
    # Scaled overflow lanes -> invalid (0 + valid 0).
    overflow = (is_valid & np.isfinite(x) & (x != 0.0)
                & ~(np.isfinite(y) & (np.abs(y) <= 9.007199254740992e15)))
    out[overflow] = 0.0
    ov[overflow] = 0
    return ERR_OK, np.ascontiguousarray(out), np.ascontiguousarray(ov)


def _draw_scalars(count, seed, stream, offset):
    """Sequential scalar draws r0 (chunked-vector exact, small counts)."""
    if count == 0:
        return np.empty(0, dtype=np.uint32)
    r0, _, _, _ = _draw_rows(count, seed, stream, offset)
    return np.ascontiguousarray(r0)


def _yates_swaps(pool, draws, upto):
    """pool[i], pool[j] swaps for i in 0..upto (draw j from draws).

    NATIVE-ALL: canonical exact lane (integer arithmetic identical to the
    retired JIT lane for n < 2**31; draws come from _draw_rows, order
    fixed by contract).
    """
    n = pool.size
    for i in range(upto):
        span = n - i
        j = i + int((int(draws[i]) * span) >> 32)
        pool[i], pool[j] = pool[j], pool[i]


def sample_no_replace(n, k, seed, stream=0, offset=0):
    """Fisher-Yates first-k (draw order). Returns (rc/ng, array)."""
    if k > n:
        return ERR_RANGE, None
    if n > 2 ** 31 - 1:
        return ERR_GEOM, None
    if n == 0:
        return 0, np.empty(0, dtype=np.int32)
    pool = np.arange(n, dtype=np.int32)
    draws = _draw_scalars(k, seed, stream, offset)
    _yates_swaps(pool, draws, k)
    return k, np.ascontiguousarray(pool[:k].copy())


def permutation(n, seed, stream=0, offset=0):
    """Full Yates permutation of 0..n. Returns (rc/ng, array)."""
    if n > 2 ** 31 - 1:
        return ERR_GEOM, None
    if n == 0:
        return 0, np.empty(0, dtype=np.int32)
    pool = np.arange(n, dtype=np.int32)
    draws = _draw_scalars(n, seed, stream, offset)
    _yates_swaps(pool, draws, n)
    return n, np.ascontiguousarray(pool)


# ---------------- R-COMPAT MT19937 (1:1 with scratch/r_rng_py.py) ------------

_MT_N = 624
_MT_M = 397
_MATRIX_A = np.uint32(0x9908B0DF)
_UPPER = np.uint32(0x80000000)
_LOWER = np.uint32(0x7FFFFFFF)
_I2_32M1 = 2.3283064365386963e-10


def _s32(x):
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x


def _imul(a, b):
    return _s32((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF))


class RCompat:
    """R 4.3.x RNG: Ripley init + MT twist/temper + fixup + rejection."""

    def __init__(self, seed):
        s = _s32(int(seed))
        for _ in range(50):
            s = _s32(_imul(69069, s) + 1)
        s = _s32(_imul(69069, s) + 1)
        self.mt = [0] * _MT_N
        for j in range(_MT_N):
            s = _s32(_imul(69069, s) + 1)
            self.mt[j] = s & 0xFFFFFFFF
        self.mti = _MT_N

    def _twist(self):
        mt = self.mt
        for kk in range(_MT_N - _MT_M):
            y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7FFFFFFF)
            mt[kk] = (mt[kk + _MT_M] ^ (y >> 1)
                      ^ (0x9908B0DF if y & 1 else 0)) & 0xFFFFFFFF
        for kk in range(_MT_N - _MT_M, _MT_N - 1):
            y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7FFFFFFF)
            mt[kk] = (mt[kk + (_MT_M - _MT_N)] ^ (y >> 1)
                      ^ (0x9908B0DF if y & 1 else 0)) & 0xFFFFFFFF
        y = (mt[_MT_N - 1] & 0x80000000) | (mt[0] & 0x7FFFFFFF)
        mt[_MT_N - 1] = (mt[_MT_M - 1] ^ (y >> 1)
                         ^ (0x9908B0DF if y & 1 else 0)) & 0xFFFFFFFF
        self.mti = 0

    def genrand_u32(self):
        if self.mti >= _MT_N:
            self._twist()
        y = self.mt[self.mti]
        self.mti += 1
        y = (y ^ (y >> 11)) & 0xFFFFFFFF
        y = (y ^ (((y << 7) & 0xFFFFFFFF) & 0x9D2C5680)) & 0xFFFFFFFF
        y = (y ^ (((y << 15) & 0xFFFFFFFF) & 0xEFC60000)) & 0xFFFFFFFF
        y = (y ^ (y >> 18)) & 0xFFFFFFFF
        return y

    @staticmethod
    def _fixup(x):
        if x <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - x <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return x

    def unif(self):
        return self._fixup(self.genrand_u32() * _I2_32M1)

    def _rbits(self, bits):
        import math as _math
        v = 0.0
        n = 0
        while n <= bits:
            v = 65536.0 * v + _math.floor(self.unif() * 65536.0)
            n += 16
        return int(v) % (2 ** bits)

    def unif_index(self, dn):
        import math as _math
        if not dn > 0:
            return 0
        bits = int(_math.ceil(_math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if not dn <= dv:
                return dv


def compat_runif(n, seed, lo=0.0, hi=1.0):
    """R-compat runif. Returns (rc, array)."""
    import math as _math
    lo_f, hi_f = float(lo), float(hi)
    if not (_math.isfinite(lo_f) and _math.isfinite(hi_f)) or not lo_f < hi_f:
        return ERR_RANGE, None
    r = RCompat(_s32(int(seed)))
    span = hi_f - lo_f
    return ERR_OK, np.ascontiguousarray(
        np.array([lo_f + span * r.unif() for _ in range(n)], dtype=np.float64))


def compat_sample(n, m, seed):
    """R-compat index draws, 0-based lanes. Returns (rc, array)."""
    if not int(n) > 0:
        return ERR_RANGE, None
    r = RCompat(_s32(int(seed)))
    return ERR_OK, np.ascontiguousarray(
        np.array([r.unif_index(int(n)) for _ in range(m)], dtype=np.int32))
