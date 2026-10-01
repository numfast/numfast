// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `map_kernel` mechanics: elementwise IR `map` over i32/f32/f64 lanes.
//!
//! Bit-exact mirror of the CPU driver (`src/Drivers/CPU/_lib/cpu.py`,
//! `op == "map"` branch), which is the frozen public semantic:
//! - `add/sub/mul`: NumPy lanes, output dtype == input lane dtype
//!   (i32 wraps mod 2^32; float scalars demote to the lane first).
//! - `div/pow`: float64 round-trip. i32 lanes rint (ties-even) then
//!   NumPy `astype(int32)` (out-of-range/Inf/NaN -> INT_MIN, x86
//!   `cvttsd2si` rule); f32 lanes widen to f64 output; f64 direct.
//! - `floor_div/mod`: i32 Python-floor semantics with sign correction
//!   (`//0`/`%0` -> 0, NumPy rule, never a trap); float lanes follow
//!   the CPython `divmod` algorithm (`fmod` + sign fix + signed-zero
//!   from the divisor), which is what NumPy implements (verified:
//!   `inf//2 -> nan`, `-1.0//inf -> -1.0`, `0.0%-2.0 -> -0.0`).
//!
//! Validity sidecars travel host-side (AND of inputs, same as CPU
//! `_valid_mask`); this kernel never sees validity. `pow` with an
//! array exponent is rejected at the IR layer (scalar-exp only).
//!
//! Same no-alloc `core`-only discipline as the other kernels.

/// IR `map` fn codes, shared by every `nf_map_*` symbol.
pub const ADD: u32 = 0;
pub const SUB: u32 = 1;
pub const MUL: u32 = 2;
pub const DIV: u32 = 3;
pub const POW: u32 = 4;
pub const FLOOR_DIV: u32 = 5;
pub const MOD: u32 = 6;

/// NumPy-compatible `pow` (measured full special-value grid, bases
/// x exps in {-inf,-2,-1,-0,0,1,2,inf,nan}): identical to `powf`
/// except `(-inf) ** positive-finite-fractional-exp`, which is NaN
/// (negative-base fractional powers are NaN, like `sqrt(-inf)`),
/// while C99 `powf` yields `+inf` for infinite bases. Integer-valued
/// exponents (incl. `(-inf)**2 -> +inf`, `(-inf)**3 -> -inf`) and all
/// other cells already agree bit-for-bit in tolerance terms.
#[inline]
pub fn pow_npy(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY && b.is_finite() && b > 0.0 && b.fract() != 0.0 {
        f64::NAN
    } else {
        a.powf(b)
    }
}
/// NumPy `astype(int32)` from float64 (x86 `cvttsd2si` rule, measured):
/// NaN, +-Inf and anything outside `[i32::MIN, 2^31)` -> INT_MIN,
/// in-range values truncate toward zero.
#[inline]
pub fn npy_i32_from_f64(v: f64) -> i32 {
    if v.is_nan() || v >= 2_147_483_648.0 || v < -2_147_483_648.0 {
        i32::MIN
    } else {
        v as i32
    }
}

/// Python-floor `(q, r)` for i32 lanes (NumPy `floor_divide`/`remainder`).
/// Zero divisor -> `(0, 0)` (NumPy rule, never a trap); `INT_MIN / -1`
/// short-circuits (NumPy yields `(MIN, 0)`).
#[inline]
pub fn floor_divmod_i32(a: i32, b: i32) -> (i32, i32) {
    if b == 0 {
        return (0, 0);
    }
    if a == i32::MIN && b == -1 {
        return (i32::MIN, 0);
    }
    let q = a / b; // trunc, cannot overflow (MIN/-1 excluded)
    let r = a.wrapping_sub(q.wrapping_mul(b));
    if r != 0 && ((r < 0) != (b < 0)) {
        (q.wrapping_sub(1), r.wrapping_add(b))
    } else {
        (q, r)
    }
}

/// NumPy float `floor_div` quotient lane (measured, 0 mismatches/5000):
/// zero divisor -> IEEE `a/b`; non-finite dividend -> NaN; infinite
/// divisor (finite dividend) -> CPython-`divmod` adjust in-lane;
/// otherwise `floor(a/b)` in-lane (f64) — NOT `(a-fmod)/b`, which
/// drifts 1 ulp (e.g. `2551.34//41.90 -> 60.00000000000001`).
#[inline]
pub fn q_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return a / b;
    }
    if !a.is_finite() {
        return f64::NAN;
    }
    if !b.is_finite() {
        let m = a % b; // == a (|a| < inf), exact
        let mut d = (a - m) / b; // signed zero
        if m != 0.0 && (b < 0.0) != (m < 0.0) {
            d -= 1.0;
        }
        return d;
    }
    (a / b).floor()
}

/// NumPy float `remainder` lane (measured, 0 mismatches/5000): the
/// CPython `float_divmod` remainder (`fmod` + sign fix, signed zero
/// takes the divisor sign). Zero divisor or non-finite dividend -> NaN.
#[inline]
pub fn r_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return f64::NAN;
    }
    if !a.is_finite() {
        return f64::NAN;
    }
    let m = a % b; // LLVM `frem` == C `fmod`, exact
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0_f64.copysign(b)
    }
}

/// f32 `floor_div` quotient: NumPy upcasts (measured, 0 mismatches/5000:
/// `f32(floor(f64(a)/f64(b)))`); in-lane `divmod` drifts 1 ulp
/// (e.g. `-> 12.000001` instead of `12`). Infinite-divisor lanes stay
/// in f32 (exact: `|a| < inf`).
#[inline]
pub fn q_f32(a: f32, b: f32) -> f32 {
    if b == 0.0 {
        return a / b;
    }
    if !a.is_finite() {
        return f32::NAN;
    }
    if !b.is_finite() {
        let m = a % b;
        let mut d = (a - m) / b;
        if m != 0.0 && (b < 0.0) != (m < 0.0) {
            d -= 1.0;
        }
        return d;
    }
    ((a as f64) / (b as f64)).floor() as f32
}

/// f32 `remainder` lane: in-lane `fmod` + sign fix (measured exact).
#[inline]
pub fn r_f32(a: f32, b: f32) -> f32 {
    if b == 0.0 {
        return f32::NAN;
    }
    if !a.is_finite() {
        return f32::NAN;
    }
    let m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0_f32.copysign(b)
    }
}

/// One array-array lane for i32 outputs (add/sub/mul/floor/div/pow/mod).
#[inline]
pub fn lane_i32(op: u32, a: i32, b: i32) -> Option<i32> {
    match op {
        ADD => Some(a.wrapping_add(b)),
        SUB => Some(a.wrapping_sub(b)),
        MUL => Some(a.wrapping_mul(b)),
        DIV => Some(npy_i32_from_f64(
            ((a as f64) / (b as f64)).round_ties_even(),
        )),
        POW => Some(npy_i32_from_f64(
            (a as f64).powf(b as f64).round_ties_even(),
        )),
        FLOOR_DIV => Some(floor_divmod_i32(a, b).0),
        MOD => Some(floor_divmod_i32(a, b).1),
        _ => None,
    }
}

/// One int-scalar lane for i32 outputs (NumPy value-based casting keeps
/// a Python-int scalar in-lane: add/sub/mul wrap; div/pow ride the same
/// f64+rint round-trip as the array path; floor/mod stay integer-exact).
#[inline]
pub fn lane_iscalar_i32(op: u32, a: i32, s: i32) -> Option<i32> {
    match op {
        ADD => Some(a.wrapping_add(s)),
        SUB => Some(a.wrapping_sub(s)),
        MUL => Some(a.wrapping_mul(s)),
        DIV => Some(npy_i32_from_f64(
            ((a as f64) / (s as f64)).round_ties_even(),
        )),
        POW => Some(npy_i32_from_f64(
            (a as f64).powf(s as f64).round_ties_even(),
        )),
        FLOOR_DIV => Some(floor_divmod_i32(a, s).0),
        MOD => Some(floor_divmod_i32(a, s).1),
        _ => None,
    }
}

/// One float-scalar lane for i32 outputs (`s` is the f64 scalar, never
/// pre-truncated: `3 * 2.5 -> 7`, matching NumPy `astype` after the op).
#[inline]
pub fn lane_scalar_i32(op: u32, a: i32, s: f64) -> Option<i32> {
    match op {
        ADD => Some(npy_i32_from_f64(a as f64 + s)),
        SUB => Some(npy_i32_from_f64(a as f64 - s)),
        MUL => Some(npy_i32_from_f64(a as f64 * s)),
        DIV => Some(npy_i32_from_f64(((a as f64) / s).round_ties_even())),
        POW => Some(npy_i32_from_f64((a as f64).powf(s).round_ties_even())),
        // Float-scalar floor/mod widen to float64 (NumPy promotion):
        // the wrapper routes them to `lane_scalar_f64` after an exact
        // i32->f64 conversion, so these lanes are unreachable there.
        // They stay `None` (BAD_RANGE) rather than a wrong-dtype value.
        FLOOR_DIV | MOD => None,
        _ => None,
    }
}

/// One array-array lane for f32 outputs (add/sub/mul/floor/mod only;
/// div/pow widen to f64 and live in [`lane_divpow_f32`]).
#[inline]
pub fn lane_f32(op: u32, a: f32, b: f32) -> Option<f32> {
    match op {
        ADD => Some(a + b),
        SUB => Some(a - b),
        MUL => Some(a * b),
        FLOOR_DIV => Some(q_f32(a, b)),
        MOD => Some(r_f32(a, b)),
        _ => None,
    }
}

/// f32 div/pow widen to f64 output (CPU `astype(f64)` round-trip).
#[inline]
pub fn lane_divpow_f32(op: u32, a: f32, b: f32) -> Option<f64> {
    match op {
        DIV => Some((a as f64) / (b as f64)),
        POW => Some(pow_npy(a as f64, b as f64)),
        _ => None,
    }
}

/// One scalar lane for f32 outputs (`s` demotes to f32 first, NumPy rule;
/// div/pow widen to f64 in [`lane_scalar_divpow_f32`]).
#[inline]
pub fn lane_scalar_f32(op: u32, a: f32, s: f64) -> Option<f32> {
    let b = s as f32;
    match op {
        ADD => Some(a + b),
        SUB => Some(a - b),
        MUL => Some(a * b),
        FLOOR_DIV => Some(q_f32(a, b)),
        MOD => Some(r_f32(a, b)),
        _ => None,
    }
}

/// f32 scalar div/pow widen to f64 output.
#[inline]
pub fn lane_scalar_divpow_f32(op: u32, a: f32, s: f64) -> Option<f64> {
    match op {
        DIV => Some((a as f64) / s),
        POW => Some(pow_npy(a as f64, s)),
        _ => None,
    }
}

/// One array-array lane for f64 outputs (all seven fns).
#[inline]
pub fn lane_f64(op: u32, a: f64, b: f64) -> Option<f64> {
    match op {
        ADD => Some(a + b),
        SUB => Some(a - b),
        MUL => Some(a * b),
        DIV => Some(a / b),
        POW => Some(pow_npy(a, b)),
        FLOOR_DIV => Some(q_f64(a, b)),
        MOD => Some(r_f64(a, b)),
        _ => None,
    }
}

/// One scalar lane for f64 outputs (all seven fns).
#[inline]
pub fn lane_scalar_f64(op: u32, a: f64, s: f64) -> Option<f64> {
    match op {
        ADD => Some(a + s),
        SUB => Some(a - s),
        MUL => Some(a * s),
        DIV => Some(a / s),
        POW => Some(pow_npy(a, s)),
        FLOOR_DIV => Some(q_f64(a, s)),
        MOD => Some(r_f64(a, s)),
        _ => None,
    }
}
