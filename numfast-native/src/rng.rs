// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Counter-based RNG core (CORE) + R-compat MT (R-COMPAT), additive.
//!
//! CORE: SplitMix64-seeded Philox4x32-10. HOT per-lane code uses ONLY
//! u32 mul/xor/shift/add (WGSL-portable by construction: the same rounds
//! run verbatim in the WGSL `rng_fill` kernel; `mul_hi_u32` is emulated
//! via 16-bit halves, never a u64 multiply). SplitMix64 (u64 wrapping
//! ops) runs COLD once per call on the host to derive the 64-bit key
//! from (seed, stream); it is NOT part of the per-lane WGSL path.
//!
//! R-COMPAT: R 4.3.x-compatible MT19937 (Ripley LCG init + twist/temper
//! + fixup + rejection chunks), 1:1 port of the `scratch/r_rng_py.py`
//! reference. Strict sequential, CPU-only, chunkable=false.
//!
//! Error codes (this surface only): 0 ok, -1 null pointer,
//! -2 bad range (hi<=lo / non-finite bounds / k>n / bad ndigits),
//! -3 bad geometry (unknown mode / n out of int32 range).
//! Lane-returning symbols return ng (>= 0) like the sorted/carry family;
//! -2 null-state convention is reused for those (see lib.rs wraps).

/// Cold key derivation: SplitMix64, host-side only (NOT WGSL per-lane).
#[inline]
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// u32 high-mul via 16-bit halves: u32 mul/add/shift ONLY (WGSL-portable).
#[inline(always)]
fn mul_hi_u32(a: u32, b: u32) -> u32 {
    let a0 = a & 0xFFFF;
    let a1 = a >> 16;
    let b0 = b & 0xFFFF;
    let b1 = b >> 16;
    let p0 = a0.wrapping_mul(b0);
    let p1 = a0.wrapping_mul(b1);
    let p2 = a1.wrapping_mul(b0);
    let p3 = a1.wrapping_mul(b1);
    // hi = p3 + (p1 >> 16) + (p2 >> 16) + carry from low halves
    let mid = (p0 >> 16).wrapping_add(p1 & 0xFFFF).wrapping_add(p2 & 0xFFFF);
    p3.wrapping_add(p1 >> 16)
        .wrapping_add(p2 >> 16)
        .wrapping_add(mid >> 16)
}

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

/// One Philox4x32 round, u32 ops only.
#[inline(always)]
fn philox_round(
    c0: u32,
    c1: u32,
    c2: u32,
    c3: u32,
    k0: u32,
    k1: u32,
) -> (u32, u32, u32, u32) {
    let hi0 = mul_hi_u32(c0, PHILOX_M0);
    let lo0 = c0.wrapping_mul(PHILOX_M0);
    let hi1 = mul_hi_u32(c2, PHILOX_M1);
    let lo1 = c2.wrapping_mul(PHILOX_M1);
    (hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0)
}

/// Philox4x32-10: counter (c0..c3) + key (k0,k1) -> 4 u32 outputs.
#[inline]
fn philox4x32_10(c0: u32, c1: u32, c2: u32, c3: u32, k0: u32, k1: u32) -> [u32; 4] {
    let (mut a0, mut a1, mut a2, mut a3) = (c0, c1, c2, c3);
    let (mut kk0, mut kk1) = (k0, k1);
    for _ in 0..10 {
        let (b0, b1, b2, b3) = philox_round(a0, a1, a2, a3, kk0, kk1);
        a0 = b0;
        a1 = b1;
        a2 = b2;
        a3 = b3;
        kk0 = kk0.wrapping_add(PHILOX_W0);
        kk1 = kk1.wrapping_add(PHILOX_W1);
    }
    [a0, a1, a2, a3]
}

/// Derive (k0,k1) from (seed, stream): two SplitMix64 draws over a
/// domain-separated state. COLD, host-side.
#[inline]
fn derive_key(seed: u64, stream: u64) -> (u32, u32) {
    let mut st = seed
        .wrapping_add(0x9E3779B97F4A7C15)
        .wrapping_add(stream.wrapping_mul(0xBF58476D1CE4E5B9));
    let k = splitmix64_next(&mut st);
    (k as u32, (k >> 32) as u32)
}

/// Per-lane draw: global counter g = offset + i, stream mixed into c2/c3.
#[inline]
fn draw4(seed: u64, stream: u64, g: u64) -> [u32; 4] {
    let (k0, k1) = derive_key(seed, stream);
    let c0 = g as u32;
    let c1 = (g >> 32) as u32;
    let c2 = (stream as u32) ^ 0xC0FF_EE00;
    let c3 = ((stream >> 32) as u32) ^ 0x9E37_79B9;
    philox4x32_10(c0, c1, c2, c3, k0, k1)
}

/// CORE fill i32: out[i] = lo + (draw * width >> 32), [lo, hi).
/// width==1 -> constant lo (counter still advances per lane, so chunked
/// and unchunked agree trivially). mode 0 = 'bits'; else -3.
pub fn rng_fill_i32(
    out: &mut [i32],
    seed: u64,
    stream: u64,
    offset: u64,
    lo: i32,
    hi: i32,
    mode: u32,
) -> i32 {
    if mode != 0 {
        return crate::core::errors::MALFORMED;
    }
    if hi <= lo {
        return crate::core::errors::BAD_RANGE;
    }
    let width = (hi as i64 - lo as i64) as u64;
    if width == 0 {
        return crate::core::errors::BAD_RANGE;
    }
    if width == 1 {
        for o in out.iter_mut() {
            *o = lo;
        }
        // Counter advances implicitly: lane i always uses offset + i,
        // so chunked (offset-adjusted) == unchunked bit-exact.
        return crate::core::errors::OK;
    }
    for (i, o) in out.iter_mut().enumerate() {
        let r = draw4(seed, stream, offset.wrapping_add(i as u64))[0];
        let v = ((r as u64 * width) >> 32) as i64;
        *o = (lo as i64 + v) as i32;
    }
    crate::core::errors::OK
}

/// CORE fill f64: 53-bit mantissa draws mapped to [lo, hi).
/// lo>=hi or non-finite bounds -> -2.
pub fn rng_fill_f64(
    out: &mut [f64],
    seed: u64,
    stream: u64,
    offset: u64,
    lo: f64,
    hi: f64,
) -> i32 {
    if !(lo.is_finite() && hi.is_finite()) || !(lo < hi) {
        return crate::core::errors::BAD_RANGE;
    }
    const TWO53: f64 = 9007199254740992.0;
    let span = hi - lo;
    for (i, o) in out.iter_mut().enumerate() {
        let r = draw4(seed, stream, offset.wrapping_add(i as u64));
        let u = (((r[0] >> 5) as u64) << 26) | ((r[1] >> 6) as u64);
        // u < 2^53 exactly; f in [0, 1).
        let f = (u as f64) / TWO53;
        *o = lo + f * span;
    }
    crate::core::errors::OK
}

pub const POW10_F64: [f64; 16] = [
    1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14,
    1e15,
];

/// map_round (half-even, banker's): out[i] = round_half_even(x[i], d).
/// NaN -> NaN, +-Inf -> +-Inf (validity preserved); invalid rows
/// (valid[i]==0) -> 0.0 + out_valid 0; -0.0 sign preserved.
/// ndigits > 15 -> -2 (range). Overflow of scaled ticks (|x|*10^d >
/// i64::MAX, non-finite excluded above) -> invalid row (0 + valid 0).
pub fn map_round(
    x: &[f64],
    valid: &[u8],
    ndigits: u32,
    out: &mut [f64],
    out_valid: &mut [u8],
) -> i32 {
    if ndigits > 15 {
        return crate::core::errors::BAD_RANGE;
    }
    let p = POW10_F64[ndigits as usize];
    for i in 0..x.len() {
        if valid[i] == 0 {
            out[i] = 0.0;
            out_valid[i] = 0;
            continue;
        }
        let v = x[i];
        if !v.is_finite() {
            out[i] = v; // NaN stays NaN, +-Inf stays +-Inf
            out_valid[i] = 1;
            continue;
        }
        if v == 0.0 {
            // Preserve -0.0 signbit through the no-op round.
            out[i] = v;
            out_valid[i] = 1;
            continue;
        }
        let y = v * p;
        if !y.is_finite() || y.abs() > 9.007199254740992e15 {
            out[i] = 0.0;
            out_valid[i] = 0;
            continue;
        }
        // Scaled-int ticks comparison: fl is exactly representable
        // (|y| <= 2^53 checked above), parity decided on the i64 tick.
        let f = y.floor();
        let d = y - f;
        let t: f64 = if d < 0.5 {
            f
        } else if d > 0.5 {
            f + 1.0
        } else {
            let tick = f as i64;
            if tick & 1 == 0 { f } else { f + 1.0 }
        };
        out[i] = t / p;
        // Restore -0.0: t == 0 and v < 0 (or v was -0, handled above).
        if t == 0.0 && v.is_sign_negative() {
            out[i] = -0.0;
        }
        out_valid[i] = 1;
    }
    crate::core::errors::OK
}

/// Fisher-Yates first-k (partial): pool[0..n] must be caller scratch
/// len >= n; out[0..k] receives the sample IN DRAW ORDER.
/// k > n -> -2; n > i32::MAX or k < 0 -> -3. n == 0 && k == 0 -> 0.
pub fn sample_no_replace(
    n: usize,
    k: usize,
    seed: u64,
    stream: u64,
    offset: u64,
    pool: &mut [i32],
    out: &mut [i32],
) -> i64 {
    if k > n {
        return crate::core::errors::BAD_RANGE as i64;
    }
    if n > i32::MAX as usize {
        return crate::core::errors::MALFORMED as i64;
    }
    if n == 0 {
        return 0;
    }
    for (i, s) in pool[..n].iter_mut().enumerate() {
        *s = i as i32;
    }
    let mut draws: u64 = 0;
    for i in 0..k {
        let span = (n - i) as u64;
        let r = draw4(seed, stream, offset.wrapping_add(draws))[0] as u64;
        draws += 1;
        let j = i + ((r * span) >> 32) as usize;
        pool.swap(i, j);
        out[i] = pool[i];
    }
    k as i64
}

/// Full Yates permutation: out[i] is the permuted position (permutation
/// of 0..n). Same codes as [`sample_no_replace`].
pub fn permutation(
    n: usize,
    seed: u64,
    stream: u64,
    offset: u64,
    pool: &mut [i32],
    out: &mut [i32],
) -> i64 {
    if n > i32::MAX as usize {
        return crate::core::errors::MALFORMED as i64;
    }
    if n == 0 {
        return 0;
    }
    for (i, s) in pool[..n].iter_mut().enumerate() {
        *s = i as i32;
    }
    let mut draws: u64 = 0;
    for i in 0..n {
        let span = (n - i) as u64;
        let r = draw4(seed, stream, offset.wrapping_add(draws))[0] as u64;
        draws += 1;
        let j = i + ((r * span) >> 32) as usize;
        pool.swap(i, j);
    }
    out[..n].copy_from_slice(&pool[..n]);
    n as i64
}

// ---------------- R-COMPAT MT19937 (R 4.3.x, 1:1 with scratch/r_rng_py.py) ---

const MT_N: usize = 624;
const MT_M: usize = 397;
const MATRIX_A: u32 = 0x9908B0DF;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7FFFFFFF;
const I2_32M1: f64 = 2.3283064365386963e-10; // 2^-32, same literal as R port

#[inline(always)]
fn imul_js(a: i32, b: i32) -> i32 {
    (a as u32).wrapping_mul(b as u32) as i32
}

pub struct RngCompat {
    pub mt: [u32; MT_N],
    pub mti: usize,
}

impl RngCompat {
    /// Ripley init: LCG(69069) x50 warmup + 624 fills (i32 semantics).
    pub fn new(seed: i32) -> Self {
        let mut mt = [0u32; MT_N];
        let mut s = seed;
        for _ in 0..50 {
            s = imul_js(69069, s).wrapping_add(1);
        }
        s = imul_js(69069, s).wrapping_add(1); // i_seed[0], FixupSeeds overwrite
        for j in 0..MT_N {
            s = imul_js(69069, s).wrapping_add(1);
            mt[j] = s as u32;
        }
        Self { mt, mti: MT_N }
    }

    fn twist(&mut self) {
        for kk in 0..MT_N - MT_M {
            let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
            self.mt[kk] = self.mt[kk + MT_M] ^ (y >> 1) ^ (if y & 1 == 1 { MATRIX_A } else { 0 });
        }
        for kk in MT_N - MT_M..MT_N - 1 {
            let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
            self.mt[kk] =
                self.mt[kk - (MT_N - MT_M)] ^ (y >> 1) ^ (if y & 1 == 1 { MATRIX_A } else { 0 });
        }
        let y = (self.mt[MT_N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
        self.mt[MT_N - 1] = self.mt[MT_M - 1] ^ (y >> 1) ^ (if y & 1 == 1 { MATRIX_A } else { 0 });
        self.mti = 0;
    }

    pub fn genrand_u32(&mut self) -> u32 {
        if self.mti >= MT_N {
            self.twist();
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C5680;
        y ^= (y << 15) & 0xEFC60000;
        y ^= y >> 18;
        y
    }

    #[inline]
    fn fixup(x: f64) -> f64 {
        if x <= 0.0 {
            0.5 * I2_32M1
        } else if 1.0 - x <= 0.0 {
            1.0 - 0.5 * I2_32M1
        } else {
            x
        }
    }

    /// R unif: fixup(genrand * 2^-32), strictly inside (0, 1).
    pub fn unif(&mut self) -> f64 {
        Self::fixup((self.genrand_u32() as f64) * I2_32M1)
    }

    fn rbits(&mut self, bits: u32) -> u64 {
        let mut v: f64 = 0.0;
        let mut n: u32 = 0;
        while n <= bits {
            v = 65536.0 * v + (self.unif() * 65536.0).floor();
            n += 16;
        }
        (v as u64) % (1u64 << bits)
    }

    /// R unif_index: rejection chunks (strict sequential draws).
    pub fn unif_index(&mut self, dn: i64) -> i64 {
        if dn <= 0 {
            return 0;
        }
        let mut bits: u32 = 0;
        let mut p = 1u64;
        while p < dn as u64 {
            p <<= 1;
            bits += 1;
        }
        loop {
            let dv = self.rbits(bits) as i64;
            if dn > dv {
                return dv;
            }
        }
    }
}

/// R-compat runif: out[i] = min + span * unif(), strict sequential.
pub fn compat_runif(out: &mut [f64], seed: i32, lo: f64, hi: f64) -> i32 {
    if !(lo.is_finite() && hi.is_finite()) || !(lo < hi) {
        return crate::core::errors::BAD_RANGE;
    }
    let span = hi - lo;
    let mut r = RngCompat::new(seed);
    for o in out.iter_mut() {
        *o = lo + span * r.unif();
    }
    crate::core::errors::OK
}

/// R-compat sample (0-based NumFast lanes; draw order = R draw order):
/// out[i] = unif_index(n), strict sequential. k > n allowed
/// (with-replacement draws, like R sample with replace=TRUE).
/// n <= 0 -> -2.
pub fn compat_sample(n: i64, out: &mut [i32], seed: i32) -> i32 {
    if n <= 0 {
        return crate::core::errors::BAD_RANGE;
    }
    let mut r = RngCompat::new(seed);
    for o in out.iter_mut() {
        *o = r.unif_index(n) as i32;
    }
    crate::core::errors::OK
}
