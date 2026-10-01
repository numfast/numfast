// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! numfast-native: feasibility crate — one native core, buffers in/out.
//! No Runtime/Planner classes, no legacy copies, zero external dependencies.
//!
//! Layout: `core/` (buffers, checks, errors, validity, numeric, compact,
//! reserved hash atoms) + `groupby/` (dense, fused, sorted, reserved hash
//! facade) + `transform/` (pack, pattern) + `series/` (scan/reduce/select
//! atoms) + `result/` (carry). Hot loops stay flat and specialized in their
//! kernels; sharing happens through tiny inline atoms (see REUSE.md).
//!
//! Core loops are `core`-only (no_std-compatible by construction: no alloc,
//! no collections, no syscalls); crate stays `std` for ctypes ergonomics.

pub mod adjacency;
pub mod bounded_select;
pub mod core;
pub mod cost;
pub mod groupby;
pub mod join;
pub mod join_i64;
pub mod pair_insert;
pub mod result;
pub mod rng;
pub mod router;
pub mod rowmin;
pub mod rowmin_kway;
pub mod rowmin_kway_proto;
pub mod segmented;
pub mod series;
pub mod sort;
pub mod sssp;
pub mod text;
pub mod transform;
pub mod unique;

// Internal import compat: previous flat module paths.
pub use adjacency::{adjacency_gather, adjacency_slice};
pub use cost::{cost_intern_rows, cost_travel_batch};
pub use groupby::dense::{dense_scatter, fused_sum_count};
pub use groupby::fused::{fused_multi_aggregate_soa, fused_scatter_soa};
pub use groupby::hash_grouped::{
    GH_GEOM, GH_PART_MAX, gh_count, gh_fp, gh_occ_count, gh_occ_fill, gh_pins, gh_scatter,
};
pub use groupby::mixed::{count_only, mixed_sum_only_i32_f64, owner_2i32_1f64};
pub use groupby::pack_fused::{pack_fused_f64, pack_fused_i32};
pub use groupby::scaled::{
    checked_scatter_i32, checked_scatter_i64, dense_scatter_i64, sorted_dense_f64,
    sorted_dense_i32, sorted_dense_i64,
};
pub use groupby::sorted::{sorted_run_aggregate_f64, sorted_run_aggregate_i64, sorted_runs};
pub use groupby::variants::{V_CHECKED, V_DENSE, V_SORTED, dispatch_f64, dispatch_i32, dispatch_i64};
pub use segmented::{
    MAX_DISPATCH_N, SEG_MAX, SEG_MIN, SEG_SUM, segment_count, segment_reduce_f32,
    segment_reduce_i32,
};
pub use result::carry::{
    carry_build_f64, carry_build_i64, carry_result_build_f64, carry_result_build_i64,
};
pub use rowmin::rowwise_min4_argmin_gather;
pub use rowmin::{rowwise_min4_time_argmin, rowwise_min4_time_argmin_gather};
pub use rowmin_kway::rowwise_kway_time_argmin_gather;
pub use series::select::{
    mask_combine_and, mask_combine_or, mask_negate, select_count, select_scatter,
};
pub use transform::pack::{pack_codes, pack_i32_direct};
pub use transform::pattern::{pattern_encode, pattern_scan_parse};

use core::buffers::{borrow, borrow_mut};
use core::checks::non_null;
use core::errors::{BAD_RANGE, MALFORMED, NULL_OR_ABORT, NULL_STATE, OK};

/// FFI: dense fused group-by accumulate.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `keys[0..n]`: int32 dense codes, each in `[0, g)`
/// - `values[0..n]`: float64
/// - `sums[0..g]`, `counts[0..g]`: outputs, overwritten (zeroed first)
/// - returns 0 on success, -1 on null pointer.
///
/// Out-of-range key = precondition violation (safe Rust traps, never UB).
/// Same symbol is exported from the wasm32 module; on wasm32 the
/// `usize` params lower to i32 (n < 2^31 assumed).
#[no_mangle]
pub unsafe extern "C" fn nf_group_sum_count(
    keys: *const i32,
    values: *const f64,
    n: usize,
    sums: *mut f64,
    counts: *mut i64,
    g: usize,
) -> i32 {
    if !non_null(keys) || !non_null(values) || !non_null(sums) || !non_null(counts) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let values = unsafe { borrow(values, n) };
    let sums = unsafe { borrow_mut(sums, g) };
    let counts = unsafe { borrow_mut(counts, g) };
    dense_scatter(keys, values, g, sums, counts);
    OK
}

/// FFI: fused SoA multi-aggregate (shared counts + per-column sums).
///
/// Contract (caller-owned buffers, all contiguous):
/// - `keys[0..n]`: int32 dense codes, each in `[0, g)`
/// - `values[0..ncols*n]`: SoA columns, column c at `[c*n..(c+1)*n]`
/// - `sums[0..ncols*g]`: outputs, column c at `[c*g..(c+1)*g]`, overwritten
/// - `counts[0..g]`: shared output, overwritten
/// - returns 0 ok, -1 null pointer, -2 out-of-range key (no panic via FFI).
///
/// Same symbol is exported from the wasm32 module; on wasm32 the
/// `usize` params lower to i32 (n < 2^31 assumed).
#[no_mangle]
pub unsafe extern "C" fn nf_group_multi_sum_count(
    keys: *const i32,
    values: *const f64,
    n: usize,
    ncols: usize,
    sums: *mut f64,
    counts: *mut i64,
    g: usize,
) -> i32 {
    if !non_null(keys) || !non_null(values) || !non_null(sums) || !non_null(counts) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let values = unsafe { borrow(values, ncols * n) };
    let sums = unsafe { borrow_mut(sums, ncols * g) };
    let counts = unsafe { borrow_mut(counts, g) };
    fused_scatter_soa(keys, values, ncols, g, sums, counts)
}

/// FFI (P2, accepted): single-pass mixed sums over `n_i32` int32 columns
/// + `n_f64` float64 columns, WITHOUT the counts lane (occupancy comes
/// from one [`nf_group_count_only`] pass). Generic over lane counts,
/// never over queries: Q5 = (2, 1), Q3 = (1, 1). int64 columns are NOT
/// covered (caller falls back, never a silent lossy route).
///
/// Contract (caller-owned buffers, all contiguous, SoA like
/// `nf_group_multi_sum_count`):
/// - `keys[0..n]`: int32 dense codes, each in `[0, g)`
/// - `i32_vals[0..n_i32*n]`, `f64_vals[0..n_f64*n]`: SoA value columns
/// - `i32_sums[0..n_i32*g]`, `f64_sums[0..n_f64*g]`: outputs, overwritten
/// - returns 0 ok, -1 null pointer, -2 out-of-range key, -4 int32 overflow.
#[no_mangle]
pub unsafe extern "C" fn nf_group_mixed_sum_only(
    keys: *const i32,
    i32_vals: *const i32,
    f64_vals: *const f64,
    n: usize,
    n_i32: usize,
    n_f64: usize,
    i32_sums: *mut i32,
    f64_sums: *mut f64,
    g: usize,
) -> i32 {
    if !non_null(keys) {
        return NULL_OR_ABORT;
    }
    if (n_i32 > 0 && (!non_null(i32_vals) || !non_null(i32_sums)))
        || (n_f64 > 0 && (!non_null(f64_vals) || !non_null(f64_sums)))
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let i32_vals = unsafe { borrow(i32_vals, n_i32 * n) };
    let f64_vals = unsafe { borrow(f64_vals, n_f64 * n) };
    let i32_sums = unsafe { borrow_mut(i32_sums, n_i32 * g) };
    let f64_sums = unsafe { borrow_mut(f64_sums, n_f64 * g) };
    mixed_sum_only_i32_f64(
        keys, i32_vals, f64_vals, n_i32, n_f64, g, i32_sums, f64_sums,
    )
}

/// FFI (P2, accepted): keys-only count pass (occupancy companion
/// for sum-only end-to-end). Contract: `keys[0..n]`, `counts[0..g]`
/// overwritten. Returns 0 ok, -1 null, -2 out-of-range key.
#[no_mangle]
pub unsafe extern "C" fn nf_group_count_only(
    keys: *const i32,
    n: usize,
    counts: *mut i64,
    g: usize,
) -> i32 {
    if !non_null(keys) || !non_null(counts) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let counts = unsafe { borrow_mut(counts, g) };
    count_only(keys, g, counts)
}

/// FFI (Q5-P3, experimental): owner-shard 2xi32 + 1xf64 sums WITH the
/// counts lane. Each thread owns a disjoint group range `[lo, hi)` of
/// SHARED dense state (no private states, no merge, no atomics); every
/// owner scans all rows, skipping out-of-range rows (no partition pass,
/// no materialization). Specialized by lane signature (2, 1) + large-M
/// dispatch on the caller side, never by query. Contract (caller-owned
/// buffers, all contiguous):
/// - `keys/a/b/c[0..n]`: int32 codes in `[0, g)` + int32/int32/float64
///   value lanes; `s1/s2[0..g]` int32, `s3[0..g]` float64,
///   `counts[0..g]` int64 outputs; ONLY lanes `[lo, hi)` are written
///   (disjoint across concurrent callers; `lo <= hi <= g` required).
/// - returns 0 ok, -1 null pointer, -2 out-of-range key, -4 int32 overflow.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_group_owner_2i32_1f64(
    keys: *const i32,
    a: *const i32,
    b: *const i32,
    c: *const f64,
    n: usize,
    lo: usize,
    hi: usize,
    s1: *mut i32,
    s2: *mut i32,
    s3: *mut f64,
    counts: *mut i64,
    g: usize,
) -> i32 {
    if !non_null(keys)
        || !non_null(a)
        || !non_null(b)
        || !non_null(c)
        || !non_null(s1)
        || !non_null(s2)
        || !non_null(s3)
        || !non_null(counts)
        || lo > hi
        || hi > g
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let a = unsafe { borrow(a, n) };
    let b = unsafe { borrow(b, n) };
    let c = unsafe { borrow(c, n) };
    let s1 = unsafe { borrow_mut(s1, g) };
    let s2 = unsafe { borrow_mut(s2, g) };
    let s3 = unsafe { borrow_mut(s3, g) };
    let counts = unsafe { borrow_mut(counts, g) };
    owner_2i32_1f64(keys, a, b, c, lo, hi, g, s1, s2, s3, counts)
}

/// FFI: int32-direct mixed-radix composite out[i] = k1[i]*m2 + k2[i].
///
/// Contract (caller-owned buffers, all contiguous):
/// - `k1[0..n]`, `k2[0..n]`: int32 codes (nonnegative)
/// - `out[0..n]`: int32 output, overwritten
/// - returns 0 ok, -1 null pointer, -2 negative input / int32 overflow.
///
/// Same symbol is exported from the wasm32 module; on wasm32 the
/// `usize` param lowers to i32 (n < 2^31 assumed).
#[no_mangle]
pub unsafe extern "C" fn nf_pack_i32_direct(
    k1: *const i32,
    k2: *const i32,
    m2: i32,
    n: usize,
    out: *mut i32,
) -> i32 {
    if !non_null(k1) || !non_null(k2) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let k1 = unsafe { borrow(k1, n) };
    let k2 = unsafe { borrow(k2, n) };
    let out = unsafe { borrow_mut(out, n) };
    pack_codes(k1, k2, m2, out)
}

/// FFI (P4, experimental): fused pack + i32 sum+count, no packed buffer.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `k1[0..n]`, `k2[0..n]`: int32 codes (nonnegative)
/// - `vals[0..n]`: int32 values
/// - `sums[0..g]`, `counts[0..g]`: outputs, overwritten
/// - returns 0 ok, -1 null pointer, -2 pack/key range, -4 i32 overflow.
#[no_mangle]
pub unsafe extern "C" fn nf_pack_sum_count_i32(
    k1: *const i32,
    k2: *const i32,
    m2: i32,
    vals: *const i32,
    n: usize,
    sums: *mut i32,
    counts: *mut i64,
    g: usize,
) -> i32 {
    if !non_null(k1) || !non_null(k2) || !non_null(vals) || !non_null(sums) || !non_null(counts)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let k1 = unsafe { borrow(k1, n) };
    let k2 = unsafe { borrow(k2, n) };
    let vals = unsafe { borrow(vals, n) };
    let sums = unsafe { borrow_mut(sums, g) };
    let counts = unsafe { borrow_mut(counts, g) };
    pack_fused_i32(k1, k2, m2, vals, g, sums, counts)
}

/// FFI (P4, experimental): fused pack + f64 sum+count, no packed buffer.
///
/// Same contract as [`nf_pack_sum_count_i32`] with float64
/// `vals[0..n]` / `sums[0..g]`; no overflow code (inf on extremes,
/// same as the existing float kernels).
#[no_mangle]
pub unsafe extern "C" fn nf_pack_sum_count_f64(
    k1: *const i32,
    k2: *const i32,
    m2: i32,
    vals: *const f64,
    n: usize,
    sums: *mut f64,
    counts: *mut i64,
    g: usize,
) -> i32 {
    if !non_null(k1) || !non_null(k2) || !non_null(vals) || !non_null(sums) || !non_null(counts)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let k1 = unsafe { borrow(k1, n) };
    let k2 = unsafe { borrow(k2, n) };
    let vals = unsafe { borrow(vals, n) };
    let sums = unsafe { borrow_mut(sums, g) };
    let counts = unsafe { borrow_mut(counts, g) };
    pack_fused_f64(k1, k2, m2, vals, g, sums, counts)
}

/// FFI: prefix+int string encode -> int32 codes + validity + width.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `data[0..total]`: concatenated row bytes; `offs[0..n+1]`: int32 row
///   boundaries, nondecreasing, inside data; `prefix[0..prefix_len]`
/// - `codes[0..n]`, `valid[0..n]` (u8 0/1): outputs, overwritten
/// - `width_out`, `err_row_out`: scalar metadata (width=-1 if none valid)
/// - returns 0 ok, -1 null, -2 int32 overflow (err_row = first bad row;
///   rows before it written, rest untouched), -3 malformed offsets.
///
/// Same symbol is exported from the wasm32 module; on wasm32 the
/// `usize` params lower to i32 (sizes < 2^31 assumed).
#[no_mangle]
pub unsafe extern "C" fn nf_pattern_encode(
    data: *const u8,
    total: usize,
    offs: *const i32,
    n: usize,
    prefix: *const u8,
    prefix_len: usize,
    codes: *mut i32,
    valid: *mut u8,
    width_out: *mut i32,
    err_row_out: *mut i32,
) -> i32 {
    if !non_null(data)
        || !non_null(offs)
        || !non_null(prefix)
        || !non_null(codes)
        || !non_null(valid)
        || !non_null(width_out)
        || !non_null(err_row_out)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let data = unsafe { borrow(data, total) };
    let offs = unsafe { borrow(offs, n + 1) };
    let prefix = unsafe { borrow(prefix, prefix_len) };
    let codes = unsafe { borrow_mut(codes, n) };
    let valid = unsafe { borrow_mut(valid, n) };
    let width_out = unsafe { &mut *width_out };
    let err_row_out = unsafe { &mut *err_row_out };
    let rc = pattern_scan_parse(data, offs, prefix, codes, valid, width_out, err_row_out);
    // Kernel already returns the frozen codes; assert the mapping here.
    debug_assert!(rc == OK || rc == BAD_RANGE || rc == MALFORMED);
    rc
}

/// FFI: sorted-run aggregation over nondecreasing int32 keys.
///
/// Contract (caller-owned buffers, all contiguous, len >= n):
/// - `keys[0..n]`: int32; `vals[0..n]`: int64 (`_i64`) / float64 (`_f64`)
/// - `ukeys/sums/counts[0..ng]`: outputs (ng <= n)
/// - returns ng (>= 0; 0 for empty input), -1 unsorted (speculative abort),
///   -2 null pointer. n == 0 returns 0 without dereferencing buffers.
///
/// Same symbols are exported from the wasm32 module; on wasm32 the
/// `usize` param lowers to i32 and the i64 return surfaces as BigInt.
#[no_mangle]
pub unsafe extern "C" fn nf_sorted_run_i64(
    keys: *const i32,
    vals: *const i64,
    n: usize,
    ukeys: *mut i64,
    sums: *mut i64,
    counts: *mut i64,
) -> i64 {
    if n == 0 {
        return 0;
    }
    if !non_null(keys) || !non_null(vals) || !non_null(ukeys) || !non_null(sums) || !non_null(counts)
    {
        return NULL_STATE;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let vals = unsafe { borrow(vals, n) };
    let ukeys = unsafe { borrow_mut(ukeys, n) };
    let sums = unsafe { borrow_mut(sums, n) };
    let counts = unsafe { borrow_mut(counts, n) };
    sorted_run_aggregate_i64(keys, vals, ukeys, sums, counts)
}

/// FFI: float variant of [`nf_sorted_run_i64`]; same contract.
#[no_mangle]
pub unsafe extern "C" fn nf_sorted_run_f64(
    keys: *const i32,
    vals: *const f64,
    n: usize,
    ukeys: *mut i64,
    sums: *mut f64,
    counts: *mut i64,
) -> i64 {
    if n == 0 {
        return 0;
    }
    if !non_null(keys) || !non_null(vals) || !non_null(ukeys) || !non_null(sums) || !non_null(counts)
    {
        return NULL_STATE;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let vals = unsafe { borrow(vals, n) };
    let ukeys = unsafe { borrow_mut(ukeys, n) };
    let sums = unsafe { borrow_mut(sums, n) };
    let counts = unsafe { borrow_mut(counts, n) };
    sorted_run_aggregate_f64(keys, vals, ukeys, sums, counts)
}

/// FFI: columnar result carry — compact (ukeys, counts, sums) from M-sized
/// dense aggregation state, one forward pass, ascending code order.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `counts_m[0..m]`: int64 dense counts (0 = absent)
/// - `sums_m[0..m]`: int64 (`_i64`) / float64 (`_f64`) dense sums
/// - `ukeys/counts/sums[0..ng]`: compact outputs (caller allocates m lanes)
/// - returns ng (>= 0; 0 for empty input), -2 null pointer.
///   m == 0 returns 0 without dereferencing buffers.
///
/// Same symbols are exported from the wasm32 module; on wasm32 the
/// `usize` param lowers to i32 and the i64 return surfaces as BigInt.
#[no_mangle]
pub unsafe extern "C" fn nf_carry_build_i64(
    counts_m: *const i64,
    sums_m: *const i64,
    m: usize,
    ukeys: *mut i64,
    counts: *mut i64,
    sums: *mut i64,
) -> i64 {
    if m == 0 {
        return 0;
    }
    if !non_null(counts_m)
        || !non_null(sums_m)
        || !non_null(ukeys)
        || !non_null(counts)
        || !non_null(sums)
    {
        return NULL_STATE;
    }
    // SAFETY: upheld by caller contract above.
    let counts_m = unsafe { borrow(counts_m, m) };
    let sums_m = unsafe { borrow(sums_m, m) };
    let ukeys = unsafe { borrow_mut(ukeys, m) };
    let counts = unsafe { borrow_mut(counts, m) };
    let sums = unsafe { borrow_mut(sums, m) };
    carry_build_i64(counts_m, sums_m, ukeys, counts, sums)
}

/// FFI: float variant of [`nf_carry_build_i64`]; same contract.
#[no_mangle]
pub unsafe extern "C" fn nf_carry_build_f64(
    counts_m: *const i64,
    sums_m: *const f64,
    m: usize,
    ukeys: *mut i64,
    counts: *mut i64,
    sums: *mut f64,
) -> i64 {
    if m == 0 {
        return 0;
    }
    if !non_null(counts_m)
        || !non_null(sums_m)
        || !non_null(ukeys)
        || !non_null(counts)
        || !non_null(sums)
    {
        return NULL_STATE;
    }
    // SAFETY: upheld by caller contract above.
    let counts_m = unsafe { borrow(counts_m, m) };
    let sums_m = unsafe { borrow(sums_m, m) };
    let ukeys = unsafe { borrow_mut(ukeys, m) };
    let counts = unsafe { borrow_mut(counts, m) };
    let sums = unsafe { borrow_mut(sums, m) };
    carry_build_f64(counts_m, sums_m, ukeys, counts, sums)
}

// --- Research-only multi-variant dispatch (Task 2). Production ABI
// above is frozen and untouched; these 3 symbols are additive and live
// in ONE DLL with the 8 frozen symbols. Variant ids: 0 V_DENSE,
// 1 V_CHECKED (f64/i64 only), 2 V_SORTED (needs nondecreasing keys).
// Return codes (this ABI only): 0 ok, -1 null pointer, -2
// out-of-range key, -3 unknown variant (outputs untouched),
// -4 integer overflow (checked_add failed, never wraps),
// -5 unsorted input on V_SORTED (speculative abort).

/// FFI (research): f64 group-by variant select.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `keys[0..n]`: int32 dense codes, each in `[0, g)`
/// - `values[0..n]`: float64
/// - `variant`: 0 dense / 1 checked / 2 sorted-dense
/// - `sums[0..g]`, `counts[0..g]`: outputs, overwritten
/// - returns 0 ok, -1 null, -2 out-of-range (V_CHECKED),
///   -3 unknown variant, -5 unsorted (V_SORTED).
#[no_mangle]
pub unsafe extern "C" fn nf_group_variant_f64(
    keys: *const i32,
    values: *const f64,
    n: usize,
    variant: u32,
    sums: *mut f64,
    counts: *mut i64,
    g: usize,
) -> i32 {
    if !non_null(keys) || !non_null(values) || !non_null(sums) || !non_null(counts) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let values = unsafe { borrow(values, n) };
    let sums = unsafe { borrow_mut(sums, g) };
    let counts = unsafe { borrow_mut(counts, g) };
    dispatch_f64(keys, values, g, variant, sums, counts)
}

/// FFI (research): int64-tick (scaled-integer) group-by variant select.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `keys[0..n]`: int32 dense codes, each in `[0, g)`
/// - `ticks[0..n]`: int64 physical ticks (`round((logical-offset)*M)`)
/// - `variant`: 0 dense / 1 checked / 2 sorted-dense
/// - `sums[0..g]` int64, `counts[0..g]`: outputs, overwritten
/// - returns 0 ok, -1 null, -2 out-of-range (V_CHECKED),
///   -3 unknown variant, -4 overflow, -5 unsorted (V_SORTED).
#[no_mangle]
pub unsafe extern "C" fn nf_group_variant_i64(
    keys: *const i32,
    ticks: *const i64,
    n: usize,
    variant: u32,
    sums: *mut i64,
    counts: *mut i64,
    g: usize,
) -> i32 {
    if !non_null(keys) || !non_null(ticks) || !non_null(sums) || !non_null(counts) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let ticks = unsafe { borrow(ticks, n) };
    let sums = unsafe { borrow_mut(sums, g) };
    let counts = unsafe { borrow_mut(counts, g) };
    dispatch_i64(keys, ticks, g, variant, sums, counts)
}

/// FFI (research): int32 group-by variant select (overflow-demo lanes).
///
/// Contract: same as [`nf_group_variant_i64`] with int32 `ticks`/`sums`.
/// `variant`: 0 dense (range-checked) / 2 sorted-dense; 1 and unknown
/// return -3. Mass-scale ticks overflow int32 by design — the symbol
/// exists to prove the explicit-overflow contract.
#[no_mangle]
pub unsafe extern "C" fn nf_group_variant_i32(
    keys: *const i32,
    ticks: *const i32,
    n: usize,
    variant: u32,
    sums: *mut i32,
    counts: *mut i64,
    g: usize,
) -> i32 {
    if !non_null(keys) || !non_null(ticks) || !non_null(sums) || !non_null(counts) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let ticks = unsafe { borrow(ticks, n) };
    let sums = unsafe { borrow_mut(sums, g) };
    let counts = unsafe { borrow_mut(counts, g) };
    dispatch_i32(keys, ticks, g, variant, sums, counts)
}

// --- Select/mask (IR `filter` / `mask` nodes). Additive symbols; frozen
// groupby/pack/pattern ABI above untouched. Scatter returns `m` (rows
// written, >= 0) like the sorted/carry symbols return `ng`; mask symbols
// return 0 ok / -1 null like the other buffer-style symbols.

/// FFI: count kept rows in a u8 mask (`0` = drop, nonzero = keep).
///
/// Contract (caller-owned buffers, all contiguous):
/// - `mask[0..n]`: u8 lanes
/// - returns `m` (>= 0); -2 on null pointer.
///
/// Same symbol is exported from the wasm32 module; on wasm32 the
/// `usize` param lowers to i32 (n < 2^31 assumed).
#[no_mangle]
pub unsafe extern "C" fn nf_select_count(mask: *const u8, n: usize) -> i64 {
    if n == 0 {
        return 0;
    }
    if !non_null(mask) {
        return NULL_STATE;
    }
    // SAFETY: upheld by caller contract above.
    let mask = unsafe { borrow(mask, n) };
    select_count(mask)
}

macro_rules! select_scatter_ffi {
    ($name:ident, $ty:ty, $doc:expr) => {
        #[doc = $doc]
        ///
        /// Contract (caller-owned buffers, all contiguous):
        /// - `src[0..n]`, `mask[0..n]` (u8 lanes, `0` = drop)
        /// - `out[0..n]`: output, first `m` lanes overwritten, order preserved
        /// - returns `m` (rows written, >= 0); -2 on null pointer.
        ///
        /// Same symbol is exported from the wasm32 module; on wasm32 the
        /// `usize` param lowers to i32 (n < 2^31 assumed).
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            src: *const $ty,
            mask: *const u8,
            n: usize,
            out: *mut $ty,
        ) -> i64 {
            if n == 0 {
                return 0;
            }
            if !non_null(src) || !non_null(mask) || !non_null(out) {
                return NULL_STATE;
            }
            // SAFETY: upheld by caller contract above.
            let src = unsafe { borrow(src, n) };
            let mask = unsafe { borrow(mask, n) };
            let out = unsafe { borrow_mut(out, n) };
            select_scatter(src, mask, out)
        }
    };
}

select_scatter_ffi!(
    nf_select_scatter_i32,
    i32,
    "FFI: boolean selection over int32 lanes (IR `filter`)."
);
select_scatter_ffi!(
    nf_select_scatter_i64,
    i64,
    "FFI: boolean selection over int64 lanes (IR `filter`, exact BIGINTs)."
);
select_scatter_ffi!(
    nf_select_scatter_f32,
    f32,
    "FFI: boolean selection over float32 lanes (IR `filter`)."
);
select_scatter_ffi!(
    nf_select_scatter_f64,
    f64,
    "FFI: boolean selection over float64 lanes (IR `filter`)."
);
select_scatter_ffi!(
    nf_select_scatter_u8,
    u8,
    "FFI: boolean selection over u8/bool lanes (IR `filter`)."
);

macro_rules! shift_ffi {
    ($name:ident, $ty:ty, $doc:expr) => {
        #[doc = $doc]
        ///
        /// Contract (caller-owned buffers, all contiguous):
        /// - `src[0..n]`: input lanes (read-only, never mutated)
        /// - `periods`: right-shift amount (IR-validated `>= 0`);
        ///   `periods == 0` copies, `periods >= n` zero-fills
        /// - `out[0..n]`: output, overwritten (head `min(periods, n)`
        ///   lanes zero, tail is a bit-exact copy, NaN/Inf ride as-is)
        /// - validity sidecar travels host-side (same carry rule as
        ///   CPU `_shift_ref` / GPU `shift_take`); this symbol never
        ///   sees validity
        /// - returns 0 ok, -1 null pointer (`n == 0` returns 0
        ///   without dereferencing buffers)
        ///
        /// Same symbol is exported from the wasm32 module; on wasm32 the
        /// `usize` params lower to i32 (n < 2^31 assumed).
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            src: *const $ty,
            n: usize,
            periods: usize,
            out: *mut $ty,
        ) -> i32 {
            if n == 0 {
                return OK;
            }
            if !non_null(src) || !non_null(out) {
                return NULL_OR_ABORT;
            }
            // SAFETY: upheld by caller contract above.
            let src = unsafe { borrow(src, n) };
            let out = unsafe { borrow_mut(out, n) };
            crate::series::shift::shift_scatter(src, periods, out);
            OK
        }
    };
}

shift_ffi!(
    nf_shift_i32,
    i32,
    "FFI: positional right-shift over int32 lanes (IR `shift`)."
);
shift_ffi!(
    nf_shift_f32,
    f32,
    "FFI: positional right-shift over float32 lanes (IR `shift`)."
);
shift_ffi!(
    nf_shift_f64,
    f64,
    "FFI: positional right-shift over float64 lanes (IR `shift`)."
);

// --- IR `cumsum` (inclusive prefix sum N->N, cumsum only) ---
// Data-only scan; validity travels host-side (0-fill + per-row carry,
// same rule as CPU `_cumsum_ref`); these symbols never see validity.
// int32 wraps mod 2**32 (two's complement, never saturate/trap — same
// wrap policy as the GPU scan lanes); floats accumulate in their own
// dtype (f32 stays f32, f64 stays f64), sequential order = the CPU
// oracle order (existing numerical contract, never a new tolerance).

macro_rules! cumsum_ffi {
    ($name:ident, $ty:ty, $kernel:path, $doc:expr) => {
        #[doc = $doc]
        ///
        /// Contract (caller-owned buffers, all contiguous):
        /// - `src[0..n]`: input lanes (read-only, never mutated)
        /// - `out[0..n]`: output, overwritten with the inclusive
        ///   prefix (`out[i] = sum(src[0..=i])`; NaN/Inf ride
        ///   IEEE-forward, never a validity signal)
        /// - validity sidecar travels host-side (same carry rule as
        ///   CPU `_cumsum_ref`); this symbol never sees validity
        /// - returns 0 ok, -1 null pointer (`n == 0` returns 0
        ///   without dereferencing buffers)
        ///
        /// Same symbol is exported from the wasm32 module; on wasm32 the
        /// `usize` params lower to i32 (n < 2^31 assumed).
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            src: *const $ty,
            n: usize,
            out: *mut $ty,
        ) -> i32 {
            if n == 0 {
                return OK;
            }
            if !non_null(src) || !non_null(out) {
                return NULL_OR_ABORT;
            }
            // SAFETY: upheld by caller contract above.
            let src = unsafe { borrow(src, n) };
            let out = unsafe { borrow_mut(out, n) };
            $kernel(src, out);
            OK
        }
    };
}

cumsum_ffi!(
    nf_cumsum_i32,
    i32,
    crate::series::cumsum::cumsum_scatter_i32,
    "FFI: inclusive prefix sum over int32 lanes (IR `cumsum`, wrapping)."
);
cumsum_ffi!(
    nf_cumsum_f32,
    f32,
    crate::series::cumsum::cumsum_scatter_f32,
    "FFI: inclusive prefix sum over float32 lanes (IR `cumsum`)."
);
cumsum_ffi!(
    nf_cumsum_f64,
    f64,
    crate::series::cumsum::cumsum_scatter_f64,
    "FFI: inclusive prefix sum over float64 lanes (IR `cumsum`)."
);

// --- IR `map` (elementwise add/sub/mul/div/pow/floor_div/mod) ---
// Op codes (series::map::{ADD,..}): 0 add, 1 sub, 2 mul, 3 div, 4 pow,
// 5 floor_div, 6 mod. Array-array lanes require both inputs in the lane
// dtype (mixed dtypes stay on the NumPy reference, never cast here).
// Scalar lanes split by scalar kind (NumPy value-based casting):
// int scalars stay in-lane (wrapping), float scalars ride float64.
// f32 div/pow widen to float64 output (CPU `astype(f64)` round-trip),
// hence the dedicated `divpow` symbols. Unknown op -> BAD_RANGE (-2).

macro_rules! map_ffi {
    ($name:ident, $lane:ident, $aty:ty, $bty:ty, $oty:ty, $doc:expr) => {
        #[doc = $doc]
        ///
        /// Contract (caller-owned buffers, all contiguous):
        /// - `a[0..n]`, `b[0..n]`: input lanes (read-only, never mutated)
        /// - `op`: map fn code (0 add, 1 sub, 2 mul, 3 div, 4 pow,
        ///   5 floor_div, 6 mod); lanes outside the symbol coverage
        ///   (e.g. div/pow on plain `f32`) return -2, never garbage
        /// - `out[0..n]`: output, overwritten, CPU bit-exact (integer
        ///   lanes) / CPU numerical contract (float lanes)
        /// - validity sidecars travel host-side (AND of inputs, same as
        ///   CPU `_valid_mask`); this symbol never sees validity
        /// - returns 0 ok, -1 null pointer, -2 unknown op for this
        ///   symbol (`n == 0` returns 0 without dereferencing buffers)
        ///
        /// Same symbol is exported from the wasm32 module; on wasm32 the
        /// `usize` params lower to i32 (n < 2^31 assumed).
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            a: *const $aty,
            b: *const $bty,
            n: usize,
            op: u32,
            out: *mut $oty,
        ) -> i32 {
            if n == 0 {
                if op > crate::series::map::MOD {
                    return BAD_RANGE;
                }
                return OK;
            }
            if !non_null(a) || !non_null(b) || !non_null(out) {
                return NULL_OR_ABORT;
            }
            // SAFETY: upheld by caller contract above.
            let a = unsafe { borrow(a, n) };
            let b = unsafe { borrow(b, n) };
            let out = unsafe { borrow_mut(out, n) };
            for i in 0..n {
                match crate::series::map::$lane(op, a[i], b[i]) {
                    Some(v) => out[i] = v,
                    None => return BAD_RANGE,
                }
            }
            OK
        }
    };
}

macro_rules! map_scalar_ffi {
    ($name:ident, $lane:ident, $aty:ty, $sty:ty, $oty:ty, $doc:expr) => {
        #[doc = $doc]
        ///
        /// Contract (caller-owned buffers, all contiguous):
        /// - `a[0..n]`: input lanes (read-only, never mutated)
        /// - `s`: scalar of the documented kind (int scalars stay
        ///   in-lane/wrapping, float scalars ride float64 -- NumPy
        ///   value-based casting, never pre-truncated)
        /// - `op`: map fn code (see array-array symbols); lanes outside
        ///   the symbol coverage return -2, never garbage
        /// - `out[0..n]`: output, overwritten, CPU bit-exact (integer
        ///   lanes) / CPU numerical contract (float lanes)
        /// - validity sidecars travel host-side; this symbol never sees
        ///   validity
        /// - returns 0 ok, -1 null pointer, -2 unknown op for this
        ///   symbol (`n == 0` returns 0 without dereferencing buffers)
        ///
        /// Same symbol is exported from the wasm32 module; on wasm32 the
        /// `usize` params lower to i32 (n < 2^31 assumed).
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            a: *const $aty,
            n: usize,
            s: $sty,
            op: u32,
            out: *mut $oty,
        ) -> i32 {
            if n == 0 {
                if op > crate::series::map::MOD {
                    return BAD_RANGE;
                }
                return OK;
            }
            if !non_null(a) || !non_null(out) {
                return NULL_OR_ABORT;
            }
            // SAFETY: upheld by caller contract above.
            let a = unsafe { borrow(a, n) };
            let out = unsafe { borrow_mut(out, n) };
            for i in 0..n {
                match crate::series::map::$lane(op, a[i], s) {
                    Some(v) => out[i] = v,
                    None => return BAD_RANGE,
                }
            }
            OK
        }
    };
}

map_ffi!(
    nf_map_i32,
    lane_i32,
    i32,
    i32,
    i32,
    "FFI: elementwise IR `map` over int32 lanes (array-array, all 7 fns)."
);
map_scalar_ffi!(
    nf_map_scalar_i32,
    lane_iscalar_i32,
    i32,
    i32,
    i32,
    "FFI: elementwise IR `map` over int32 lanes (int scalar, all 7 fns)."
);
map_scalar_ffi!(
    nf_map_fscalar_i32,
    lane_scalar_i32,
    i32,
    f64,
    i32,
    "FFI: elementwise IR `map` over int32 lanes (float scalar, all 7 fns)."
);
map_ffi!(
    nf_map_f32,
    lane_f32,
    f32,
    f32,
    f32,
    "FFI: elementwise IR `map` over float32 lanes (add/sub/mul/floor_div/mod)."
);
map_scalar_ffi!(
    nf_map_scalar_f32,
    lane_scalar_f32,
    f32,
    f64,
    f32,
    "FFI: elementwise IR `map` over float32 lanes, f64 scalar demoted (add/sub/mul/floor_div/mod)."
);
map_ffi!(
    nf_map_f32_divpow,
    lane_divpow_f32,
    f32,
    f32,
    f64,
    "FFI: float32 div/pow widened to float64 output (CPU round-trip contract)."
);
map_scalar_ffi!(
    nf_map_scalar_f32_divpow,
    lane_scalar_divpow_f32,
    f32,
    f64,
    f64,
    "FFI: float32 scalar div/pow widened to float64 output."
);
map_ffi!(
    nf_map_f64,
    lane_f64,
    f64,
    f64,
    f64,
    "FFI: elementwise IR `map` over float64 lanes (array-array, all 7 fns)."
);
map_scalar_ffi!(
    nf_map_scalar_f64,
    lane_scalar_f64,
    f64,
    f64,
    f64,
    "FFI: elementwise IR `map` over float64 lanes (scalar, all 7 fns)."
);


/// FFI: elementwise mask AND, output normalized `0/1` (IR `mask`).
///
/// Contract (caller-owned buffers, all contiguous):
/// - `a[0..n]`, `b[0..n]`: u8 lanes; `out[0..n]`: output, overwritten
/// - returns 0 ok, -1 null pointer.
///
/// Same symbol is exported from the wasm32 module; on wasm32 the
/// `usize` param lowers to i32 (n < 2^31 assumed).
#[no_mangle]
pub unsafe extern "C" fn nf_mask_and(a: *const u8, b: *const u8, n: usize, out: *mut u8) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(a) || !non_null(b) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let a = unsafe { borrow(a, n) };
    let b = unsafe { borrow(b, n) };
    let out = unsafe { borrow_mut(out, n) };
    mask_combine_and(a, b, out);
    OK
}

/// FFI: elementwise mask OR, output normalized `0/1` (IR `mask`).
/// Same contract as [`nf_mask_and`].
#[no_mangle]
pub unsafe extern "C" fn nf_mask_or(a: *const u8, b: *const u8, n: usize, out: *mut u8) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(a) || !non_null(b) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let a = unsafe { borrow(a, n) };
    let b = unsafe { borrow(b, n) };
    let out = unsafe { borrow_mut(out, n) };
    mask_combine_or(a, b, out);
    OK
}

/// FFI: elementwise mask NOT, output normalized `0/1` (IR `mask`).
/// Same contract as [`nf_mask_and`] with a single input.
#[no_mangle]
pub unsafe extern "C" fn nf_mask_not(a: *const u8, n: usize, out: *mut u8) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(a) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let a = unsafe { borrow(a, n) };
    let out = unsafe { borrow_mut(out, n) };
    mask_negate(a, out);
    OK
}

// --- Join M1 (native JOIN milestone 1): hash build + MT probe.
// Additive symbols; all frozen ABI above untouched. Caller-owned
// buffers, all contiguous. Native-only (no wasm export: probe uses
// scoped host threads). Gather/materialize stay NumPy caller-side.

/// FFI (M1): build the right hash table (H2O contract: unique keys).
///
/// Contract (caller-owned buffers, all contiguous):
/// - `right_keys[0..s]`: int32 right keys
/// - `table_keys/table_pos[0..cap]` int32, `table_occ[0..cap]` u8:
///   outputs, `occ` zeroed first, `pos` = right row position
/// - `cap`: power of two, `cap >= 2*s` (`s == 0` returns 0
///   without touching table lanes)
/// - returns 0 ok, -1 null pointer, -2 duplicate right key
///   (lanes before it written, caller must rebuild), -3 bad geometry.
#[no_mangle]
pub unsafe extern "C" fn nf_join_build(
    right_keys: *const i32,
    s: usize,
    table_keys: *mut i32,
    table_pos: *mut i32,
    table_occ: *mut u8,
    cap: usize,
) -> i32 {
    if s == 0 {
        return OK;
    }
    if !non_null(right_keys) || !non_null(table_keys) || !non_null(table_pos) || !non_null(table_occ)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let right_keys = unsafe { borrow(right_keys, s) };
    let table_keys = unsafe { borrow_mut(table_keys, cap) };
    let table_pos = unsafe { borrow_mut(table_pos, cap) };
    let table_occ = unsafe { borrow_mut(table_occ, cap) };
    crate::join::join_build(right_keys, s, table_keys, table_pos, table_occ, cap)
}

/// FFI (M1): MT hash probe of left keys against a built table.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `table_keys/table_pos/table_occ[0..cap]`: built table, read-only
/// - `left_keys[0..n]`: int32 probe keys
/// - `pos_out[0..n]` int64 (match position, 0 on miss),
///   `hit_out[0..n]` u8 0/1: outputs, overwritten in input order
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1); `n == 0` returns 0
///   without dereferencing buffers
/// - returns 0 ok, -1 null pointer, -3 bad geometry (cap not pow2).
#[no_mangle]
pub unsafe extern "C" fn nf_join_probe(
    table_keys: *const i32,
    table_pos: *const i32,
    table_occ: *const u8,
    cap: usize,
    left_keys: *const i32,
    n: usize,
    pos_out: *mut i64,
    hit_out: *mut u8,
    nthreads: usize,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(table_keys)
        || !non_null(table_pos)
        || !non_null(table_occ)
        || !non_null(left_keys)
        || !non_null(pos_out)
        || !non_null(hit_out)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let table_keys = unsafe { borrow(table_keys, cap) };
    let table_pos = unsafe { borrow(table_pos, cap) };
    let table_occ = unsafe { borrow(table_occ, cap) };
    let left_keys = unsafe { borrow(left_keys, n) };
    let pos_out = unsafe { borrow_mut(pos_out, n) };
    let hit_out = unsafe { borrow_mut(hit_out, n) };
    crate::join::join_probe(
        table_keys, table_pos, table_occ, cap, left_keys, n, pos_out, hit_out, nthreads,
    )
}

/// FFI (M2): MT positional gather of one int32 payload column.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `payload[0..s]`: int32 right payloads (resident, read-only)
/// - `pos[0..m]`: int64 positions, each in `[0, s)`
/// - `out[0..m]` int32: output, overwritten in input order
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1); `m == 0` returns 0
///   without dereferencing buffers
/// - returns 0 ok, -1 null pointer, -3 bad geometry (first OOB
///   position, lanes before it written; `s == 0` with `m > 0`).
#[no_mangle]
pub unsafe extern "C" fn nf_join_gather_i32(
    payload: *const i32,
    s: usize,
    pos: *const i64,
    m: usize,
    out: *mut i32,
    nthreads: usize,
) -> i32 {
    if m == 0 {
        return OK;
    }
    if !non_null(payload) || !non_null(pos) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let payload = unsafe { borrow(payload, s) };
    let pos = unsafe { borrow(pos, m) };
    let out = unsafe { borrow_mut(out, m) };
    crate::join::join_gather_i32(payload, s, pos, m, out, nthreads)
}

/// FFI (M3): fused LEFT — single-pass hash probe -> direct full-size
/// output (keys/v1/v2 + valid NULL sidecar), no pos/pos[hit]/
/// fancy-indexing temps, no separate materialize.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `table_keys/table_pos/table_occ[0..cap]`: built table, read-only
/// - `left_keys/left_v1[0..n]`: int32 probe keys + passthrough payload
/// - `payload[0..s]`: int32 right payloads (resident, read-only)
/// - `out_keys/out_v1/out_v2[0..n]` int32 + `valid_out[0..n]` u8:
///   outputs, overwritten in input order (miss -> v2 = 0, valid = 0)
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1); `n == 0` returns 0
///   without dereferencing buffers
/// - returns 0 ok, -1 null pointer, -3 bad geometry (cap not pow2;
///   `s == 0` with `n > 0`).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_join_fused_left_i32(
    table_keys: *const i32,
    table_pos: *const i32,
    table_occ: *const u8,
    cap: usize,
    left_keys: *const i32,
    left_v1: *const i32,
    payload: *const i32,
    s: usize,
    n: usize,
    out_keys: *mut i32,
    out_v1: *mut i32,
    out_v2: *mut i32,
    valid_out: *mut u8,
    nthreads: usize,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(table_keys)
        || !non_null(table_pos)
        || !non_null(table_occ)
        || !non_null(left_keys)
        || !non_null(left_v1)
        || !non_null(payload)
        || !non_null(out_keys)
        || !non_null(out_v1)
        || !non_null(out_v2)
        || !non_null(valid_out)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let table_keys = unsafe { borrow(table_keys, cap) };
    let table_pos = unsafe { borrow(table_pos, cap) };
    let table_occ = unsafe { borrow(table_occ, cap) };
    let left_keys = unsafe { borrow(left_keys, n) };
    let left_v1 = unsafe { borrow(left_v1, n) };
    let payload = unsafe { borrow(payload, s) };
    let out_keys = unsafe { borrow_mut(out_keys, n) };
    let out_v1 = unsafe { borrow_mut(out_v1, n) };
    let out_v2 = unsafe { borrow_mut(out_v2, n) };
    let valid_out = unsafe { borrow_mut(valid_out, n) };
    crate::join::join_fused_left_i32(
        table_keys, table_pos, table_occ, cap, left_keys, left_v1, payload, s, n, out_keys,
        out_v1, out_v2, valid_out, nthreads,
    )
}

/// FFI (M3): fused INNER — two-pass (count + re-probe fill) hash
/// probe -> direct compact output (keys/v1/v2), no pos/pos[hit]/
/// fancy-indexing temps, no separate materialize.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `table_keys/table_pos/table_occ[0..cap]`: built table, read-only
/// - `left_keys/left_v1[0..n]`: int32 probe keys + passthrough payload
/// - `payload[0..s]`: int32 right payloads (resident, read-only)
/// - `out_keys/out_v1/out_v2[0..n]` int32: outputs, first `m` lanes
///   overwritten in probe order
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1); `n == 0` returns 0
///   without dereferencing buffers
/// - returns `m` (compact hits, >= 0); -2 null pointer, -3 bad
///   geometry (cap not pow2; `s == 0` with `n > 0`).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_join_fused_inner_i32(
    table_keys: *const i32,
    table_pos: *const i32,
    table_occ: *const u8,
    cap: usize,
    left_keys: *const i32,
    left_v1: *const i32,
    payload: *const i32,
    s: usize,
    n: usize,
    out_keys: *mut i32,
    out_v1: *mut i32,
    out_v2: *mut i32,
    nthreads: usize,
) -> i64 {
    if n == 0 {
        return 0;
    }
    if !non_null(table_keys)
        || !non_null(table_pos)
        || !non_null(table_occ)
        || !non_null(left_keys)
        || !non_null(left_v1)
        || !non_null(payload)
        || !non_null(out_keys)
        || !non_null(out_v1)
        || !non_null(out_v2)
    {
        return NULL_STATE;
    }
    // SAFETY: upheld by caller contract above.
    let table_keys = unsafe { borrow(table_keys, cap) };
    let table_pos = unsafe { borrow(table_pos, cap) };
    let table_occ = unsafe { borrow(table_occ, cap) };
    let left_keys = unsafe { borrow(left_keys, n) };
    let left_v1 = unsafe { borrow(left_v1, n) };
    let payload = unsafe { borrow(payload, s) };
    let out_keys = unsafe { borrow_mut(out_keys, n) };
    let out_v1 = unsafe { borrow_mut(out_v1, n) };
    let out_v2 = unsafe { borrow_mut(out_v2, n) };
    crate::join::join_fused_inner_i32(
        table_keys, table_pos, table_occ, cap, left_keys, left_v1, payload, s, n, out_keys,
        out_v1, out_v2, nthreads,
    )
}

// --- Sort M1 (research): stable LSD radix argsort permutation.
// Additive symbols; all frozen ABI above untouched. Caller-owned
// buffers, all contiguous. M1 sort passes are single-threaded
// (same as ST `np.argsort`); no wasm export (research-only).

/// FFI (Sort M1 research): stable radix permutation of int32 keys.
///
/// Contract (caller-owned buffers, all contiguous, each `len >= n`):
/// - `keys[0..n]`: int32 sort keys (read-only, never mutated)
/// - `descending`: 0 = ascending stable (== `np.argsort(kind="stable")`
///   bit-for-bit), nonzero = descending stable (== `_stable_desc_idx`
///   bit-for-bit: groups reversed, input order inside each group)
/// - `perm_out[0..n]` int32: output permutation, overwritten
/// - `tmp0/tmp1[0..n]` u32 + `tmp_p[0..n]` i32: scratch, overwritten
/// - `n == 0` returns 0 without dereferencing buffers
/// - returns 0 ok, -1 null pointer, -3 bad geometry (`n > i32::MAX`,
///   int32 positions cannot hold it).
#[no_mangle]
pub unsafe extern "C" fn nf_sort_perm_i32(
    keys: *const i32,
    n: usize,
    descending: u8,
    perm_out: *mut i32,
    tmp0: *mut u32,
    tmp1: *mut u32,
    tmp_p: *mut i32,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(keys)
        || !non_null(perm_out)
        || !non_null(tmp0)
        || !non_null(tmp1)
        || !non_null(tmp_p)
    {
        return NULL_OR_ABORT;
    }
    if n > i32::MAX as usize {
        return crate::core::errors::MALFORMED;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let perm_out = unsafe { borrow_mut(perm_out, n) };
    let tmp0 = unsafe { borrow_mut(tmp0, n) };
    let tmp1 = unsafe { borrow_mut(tmp1, n) };
    let tmp_p = unsafe { borrow_mut(tmp_p, n) };
    crate::sort::sort_perm_i32(keys, descending != 0, perm_out, tmp0, tmp1, tmp_p);
    OK
}

/// FFI (Sort M1 research): stable radix permutation of int64 keys.
/// Same contract as [`nf_sort_perm_i32`] with int64 `keys[0..n]`
/// (positions still int32; `n > i32::MAX` returns -3).
#[no_mangle]
pub unsafe extern "C" fn nf_sort_perm_i64(
    keys: *const i64,
    n: usize,
    descending: u8,
    perm_out: *mut i32,
    tmp0: *mut u64,
    tmp1: *mut u64,
    tmp_p: *mut i32,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(keys)
        || !non_null(perm_out)
        || !non_null(tmp0)
        || !non_null(tmp1)
        || !non_null(tmp_p)
    {
        return NULL_OR_ABORT;
    }
    if n > i32::MAX as usize {
        return crate::core::errors::MALFORMED;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let perm_out = unsafe { borrow_mut(perm_out, n) };
    let tmp0 = unsafe { borrow_mut(tmp0, n) };
    let tmp1 = unsafe { borrow_mut(tmp1, n) };
    let tmp_p = unsafe { borrow_mut(tmp_p, n) };
    crate::sort::sort_perm_i64(keys, descending != 0, perm_out, tmp0, tmp1, tmp_p);
    OK
}

// --- Unique M1 (research): sorted-order unique values + inverse codes.
// Additive symbols; all frozen ABI above untouched. Caller-owned
// buffers, all contiguous. Single-threaded (same as ST `np.unique`);
// no wasm export (research-only).
//
// SEMANTICS (frozen): identical labelling to
// `np.unique(keys, return_inverse=True)` — `uniq` sorted ascending,
// `inv` sorted-position codes, `uniq[inv] == keys`. NOT first-appearance
// order; downstream depends on the sorted labelling, so parity is exact
// against `np.unique` only under this semantics.

/// FFI (Unique M1 research): sorted-order unique + inverse of int32 keys.
///
/// Contract (caller-owned buffers, all contiguous, each `len >= n`):
/// - `keys[0..n]`: int32 keys (read-only, never mutated)
/// - `uniq_out[0..n]` int32: first `ng` lanes overwritten with the sorted
///   distinct values; `inv_out[0..n]` int32: overwritten with the
///   sorted-position code per row (`uniq[inv] == keys`)
/// - `perm/tmp_p[0..n]` i32 + `tmp0/tmp1[0..n]` u32: scratch, overwritten
/// - `n == 0` returns 0 without dereferencing buffers
/// - returns `ng` (>= 0; 0 for empty input), -2 null pointer,
///   -3 bad geometry (`n > i32::MAX`, int32 codes cannot hold it).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_unique_inverse_i32(
    keys: *const i32,
    n: usize,
    uniq_out: *mut i32,
    inv_out: *mut i32,
    perm: *mut i32,
    tmp0: *mut u32,
    tmp1: *mut u32,
    tmp_p: *mut i32,
) -> i64 {
    if n == 0 {
        return 0;
    }
    if !non_null(keys)
        || !non_null(uniq_out)
        || !non_null(inv_out)
        || !non_null(perm)
        || !non_null(tmp0)
        || !non_null(tmp1)
        || !non_null(tmp_p)
    {
        return NULL_STATE;
    }
    if n > i32::MAX as usize {
        return crate::core::errors::MALFORMED as i64;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let uniq_out = unsafe { borrow_mut(uniq_out, n) };
    let inv_out = unsafe { borrow_mut(inv_out, n) };
    let perm = unsafe { borrow_mut(perm, n) };
    let tmp0 = unsafe { borrow_mut(tmp0, n) };
    let tmp1 = unsafe { borrow_mut(tmp1, n) };
    let tmp_p = unsafe { borrow_mut(tmp_p, n) };
    crate::sort::sort_perm_i32(keys, false, perm, tmp0, tmp1, tmp_p);
    crate::unique::unique_inverse_from_perm_i32(keys, perm, uniq_out, inv_out) as i64
}

/// FFI (Unique M1 research): sorted-order unique + inverse of int64 keys.
/// Same contract as [`nf_unique_inverse_i32`] with int64
/// `keys[0..n]` / `uniq_out[0..n]` (codes stay int32; `n > i32::MAX`
/// returns -3).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_unique_inverse_i64(
    keys: *const i64,
    n: usize,
    uniq_out: *mut i64,
    inv_out: *mut i32,
    perm: *mut i32,
    tmp0: *mut u64,
    tmp1: *mut u64,
    tmp_p: *mut i32,
) -> i64 {
    if n == 0 {
        return 0;
    }
    if !non_null(keys)
        || !non_null(uniq_out)
        || !non_null(inv_out)
        || !non_null(perm)
        || !non_null(tmp0)
        || !non_null(tmp1)
        || !non_null(tmp_p)
    {
        return NULL_STATE;
    }
    if n > i32::MAX as usize {
        return crate::core::errors::MALFORMED as i64;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let uniq_out = unsafe { borrow_mut(uniq_out, n) };
    let inv_out = unsafe { borrow_mut(inv_out, n) };
    let perm = unsafe { borrow_mut(perm, n) };
    let tmp0 = unsafe { borrow_mut(tmp0, n) };
    let tmp1 = unsafe { borrow_mut(tmp1, n) };
    let tmp_p = unsafe { borrow_mut(tmp_p, n) };
    crate::sort::sort_perm_i64(keys, false, perm, tmp0, tmp1, tmp_p);
    crate::unique::unique_inverse_from_perm_i64(keys, perm, uniq_out, inv_out) as i64
}

// --- RNG (CORE counter-based + R-COMPAT MT, additive). Frozen ABI above
// untouched. CORE symbols are counter-based (chunkable: lane i always
// uses counter offset + i, so chunked == unchunked bit-exact) and
// WGSL-portable (u32 mul/xor/shift/add only in the HOT lane path).
// R-COMPAT symbols are strict-sequential CPU-only (chunkable=false).
// Codes (this surface only): 0 ok, -1 null pointer, -2 bad range,
// -3 bad geometry. Lane-returning symbols return counts (>= 0) and
// reuse -2 null-state like the sorted/carry family.

/// FFI (RNG CORE): counter-based fill of int32 lanes in [lo, hi).
///
/// Contract (caller-owned buffers, all contiguous):
/// - `out[0..n]`: int32 output, overwritten
/// - `seed`/`stream`/`offset`: u64 counter domain; lane i draws with
///   counter `offset + i` (chunk: pass chunk offset, bit-exact merge)
/// - `lo`/`hi`: int32 bounds, `[lo, hi)`; width 1 -> constant `lo`
///   (counter still advances per lane)
/// - `mode`: 0 = 'bits' (Lemire `r * width >> 32` map); else -3
/// - returns 0 ok, -1 null, -2 bad range (hi <= lo), -3 bad geometry.
#[no_mangle]
pub unsafe extern "C" fn nf_rng_fill_i32(
    out: *mut i32,
    n: usize,
    seed: u64,
    stream: u64,
    offset: u64,
    lo: i32,
    hi: i32,
    mode: u32,
) -> i32 {
    if n == 0 {
        if hi <= lo {
            return BAD_RANGE;
        }
        if mode != 0 {
            return MALFORMED;
        }
        return OK;
    }
    if !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let out = unsafe { borrow_mut(out, n) };
    crate::rng::rng_fill_i32(out, seed, stream, offset, lo, hi, mode)
}

/// FFI (RNG CORE): counter-based fill of float64 lanes in [lo, hi).
///
/// Contract: 53-bit mantissa draws (`(r0 >> 5) << 26 | (r1 >> 6)`,
/// `u / 2^53` in `[0, 1)`, scaled by the span). Same counter domain
/// as [`nf_rng_fill_i32`]. lo >= hi or non-finite bounds -> -2.
/// Same symbol is exported from the wasm32 module.
#[no_mangle]
pub unsafe extern "C" fn nf_rng_fill_f64(
    out: *mut f64,
    n: usize,
    seed: u64,
    stream: u64,
    offset: u64,
    lo: f64,
    hi: f64,
) -> i32 {
    if n == 0 {
        if !(lo.is_finite() && hi.is_finite()) || !(lo < hi) {
            return BAD_RANGE;
        }
        return OK;
    }
    if !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let out = unsafe { borrow_mut(out, n) };
    crate::rng::rng_fill_f64(out, seed, stream, offset, lo, hi)
}

/// FFI (RNG CORE): half-even round to `ndigits` (0..15) with validity.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `x[0..n]` f64, `valid[0..n]` u8 (0 = invalid): inputs
/// - `out[0..n]` f64, `out_valid[0..n]` u8: outputs, overwritten
/// - NaN -> NaN, +-Inf -> +-Inf (validity kept); invalid rows ->
///   0.0 + valid 0; -0.0 signbit preserved; scaled overflow -> invalid
/// - `ndigits > 15` -> -2; returns 0 ok, -1 null.
#[no_mangle]
pub unsafe extern "C" fn nf_rng_map_round(
    x: *const f64,
    valid: *const u8,
    n: usize,
    ndigits: u32,
    out: *mut f64,
    out_valid: *mut u8,
) -> i32 {
    if n == 0 {
        if ndigits > 15 {
            return BAD_RANGE;
        }
        return OK;
    }
    if !non_null(x) || !non_null(valid) || !non_null(out) || !non_null(out_valid) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let x = unsafe { borrow(x, n) };
    let valid = unsafe { borrow(valid, n) };
    let out = unsafe { borrow_mut(out, n) };
    let out_valid = unsafe { borrow_mut(out_valid, n) };
    crate::rng::map_round(x, valid, ndigits, out, out_valid)
}

/// FFI (RNG CORE): Fisher-Yates first-k sample (draw order).
///
/// Contract (caller-owned buffers, all contiguous):
/// - `pool[0..n]` i32 scratch, `out[0..k]` i32 output (both overwritten)
/// - draws are sequential from the counter domain (`offset + draw#`),
///   so resuming needs the draw count, NOT a row offset (chunkable=false
///   for the draw stream; row-chunking N/A: single global sample)
/// - returns k (>= 0; 0 for k == 0) as i64, -2 null-state, -3 geometry
///   (n > i32::MAX), -2 range (k > n, as i64 code).
#[no_mangle]
pub unsafe extern "C" fn nf_rng_sample_no_replace(
    n: usize,
    k: usize,
    seed: u64,
    stream: u64,
    offset: u64,
    pool: *mut i32,
    out: *mut i32,
) -> i64 {
    if n == 0 && k == 0 {
        return 0;
    }
    if !non_null(pool) || !non_null(out) {
        return NULL_STATE;
    }
    // SAFETY: upheld by caller contract above.
    let pool = unsafe { borrow_mut(pool, n) };
    let out = unsafe { borrow_mut(out, k) };
    crate::rng::sample_no_replace(n, k, seed, stream, offset, pool, out)
}

/// FFI (RNG CORE): full Yates permutation of 0..n.
/// Same contract as [`nf_rng_sample_no_replace`] with k == n
/// (pool + out both len >= n); returns n.
#[no_mangle]
pub unsafe extern "C" fn nf_rng_permutation(
    n: usize,
    seed: u64,
    stream: u64,
    offset: u64,
    pool: *mut i32,
    out: *mut i32,
) -> i64 {
    if n == 0 {
        return 0;
    }
    if !non_null(pool) || !non_null(out) {
        return NULL_STATE;
    }
    // SAFETY: upheld by caller contract above.
    let pool = unsafe { borrow_mut(pool, n) };
    let out = unsafe { borrow_mut(out, n) };
    crate::rng::permutation(n, seed, stream, offset, pool, out)
}

/// FFI (R-COMPAT, CPU-only): R 4.3.x-compatible runif.
/// `out[i] = lo + (hi - lo) * unif()`, strict sequential draws
/// (Ripley init + twist/temper + fixup). chunkable=false.
/// Returns 0 ok, -1 null, -2 bad range.
#[no_mangle]
pub unsafe extern "C" fn nf_rng_compat_runif(
    out: *mut f64,
    n: usize,
    seed: i32,
    lo: f64,
    hi: f64,
) -> i32 {
    if n == 0 {
        if !(lo.is_finite() && hi.is_finite()) || !(lo < hi) {
            return BAD_RANGE;
        }
        return OK;
    }
    if !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let out = unsafe { borrow_mut(out, n) };
    crate::rng::compat_runif(out, seed, lo, hi)
}

/// FFI (R-COMPAT, CPU-only): R-compatible index draws (0-based lanes).
/// `out[i] = unif_index(n)` (rejection chunks, strict sequential,
/// same draw stream as R `sample`; 0-based for NumFast lanes).
/// Returns 0 ok, -1 null, -2 bad range (n <= 0).
#[no_mangle]
pub unsafe extern "C" fn nf_rng_compat_sample(
    n: i64,
    out: *mut i32,
    m: usize,
    seed: i32,
) -> i32 {
    if m == 0 {
        if n <= 0 {
            return BAD_RANGE;
        }
        return OK;
    }
    if !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let out = unsafe { borrow_mut(out, m) };
    crate::rng::compat_sample(n, out, seed)
}

// --- TEXT length/contains (additive). Frozen ABI above untouched.
// Length = Unicode code points (NOT bytes); contains = UTF-8 substring,
// case-sensitive, empty needle matches all rows. Validity travels
// host-side (DELTA-3 sidecar); these symbols never see validity.
// Codes (this surface only): 0 ok, -1 null pointer, -2 bad needle
// (contains: non-UTF8 needle), -3 malformed offsets / invalid UTF-8
// row (explicit, never silent). Row-independent: chunkable=true.

/// FFI: code-point length per row (`TEXT length`).
///
/// Contract (caller-owned buffers, all contiguous):
/// - `data[0..total]`: concatenated row bytes (UTF-8)
/// - `offs[0..n+1]`: int32 row boundaries, nondecreasing, inside data
/// - `out[0..n]`: int32 output, overwritten with code-point counts
/// - returns 0 ok, -1 null, -3 malformed (`n == 0` returns 0
///   without dereferencing buffers)
///
/// Same symbol is exported from the wasm32 module; on wasm32 the
/// `usize` params lower to i32 (sizes < 2^31 assumed).
#[no_mangle]
pub unsafe extern "C" fn nf_text_length(
    data: *const u8,
    total: usize,
    offs: *const i32,
    n: usize,
    out: *mut i32,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(data) || !non_null(offs) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let data = unsafe { borrow(data, total) };
    let offs = unsafe { borrow(offs, n + 1) };
    let out = unsafe { borrow_mut(out, n) };
    crate::text::text_length_scan(data, offs, out)
}

/// FFI: substring hit per row (`TEXT contains`, u8 0/1).
///
/// Contract (caller-owned buffers, all contiguous):
/// - `data[0..total]`, `offs[0..n+1]`: same wire as [`nf_text_length`]
/// - `needle[0..needle_len]`: UTF-8 substring (empty matches all rows)
/// - `out[0..n]`: u8 output, overwritten (1 = hit)
/// - returns 0 ok, -1 null, -2 non-UTF8 needle, -3 malformed
///   (`n == 0` returns 0 without dereferencing buffers; a non-UTF8
///   needle still returns -2 even when `n == 0`)
///
/// Same symbol is exported from the wasm32 module; on wasm32 the
/// `usize` params lower to i32 (sizes < 2^31 assumed).
#[no_mangle]
pub unsafe extern "C" fn nf_text_contains(
    data: *const u8,
    total: usize,
    offs: *const i32,
    n: usize,
    needle: *const u8,
    needle_len: usize,
    out: *mut u8,
) -> i32 {
    if !non_null(needle) && needle_len > 0 {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above (empty needle: no read).
    let needle = unsafe { borrow(needle, needle_len) };
    if crate::text::utf8_count_validated(needle).is_none() {
        return BAD_RANGE;
    }
    if n == 0 {
        return OK;
    }
    if !non_null(data) || !non_null(offs) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let data = unsafe { borrow(data, total) };
    let offs = unsafe { borrow(offs, n + 1) };
    let out = unsafe { borrow_mut(out, n) };
    crate::text::text_contains_scan(data, offs, needle, out)
}

/// FFI: anchored affix hit per row (`TEXT startswith/endswith`, u8 0/1).
///
/// Same wire as [`nf_text_contains`]; empty needle matches all rows.
/// Returns 0 ok, -1 null, -2 non-UTF8 needle, -3 malformed
/// (`n == 0` returns 0 without dereferencing buffers; a non-UTF8
/// needle still returns -2 even when `n == 0`).
///
/// Same symbol is exported from the wasm32 module; on wasm32 the
/// `usize` params lower to i32 (sizes < 2^31 assumed).
#[no_mangle]
pub unsafe extern "C" fn nf_text_startswith(
    data: *const u8,
    total: usize,
    offs: *const i32,
    n: usize,
    needle: *const u8,
    needle_len: usize,
    out: *mut u8,
) -> i32 {
    if !non_null(needle) && needle_len > 0 {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above (empty needle: no read).
    let needle = unsafe { borrow(needle, needle_len) };
    if crate::text::utf8_count_validated(needle).is_none() {
        return BAD_RANGE;
    }
    if n == 0 {
        return OK;
    }
    if !non_null(data) || !non_null(offs) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let data = unsafe { borrow(data, total) };
    let offs = unsafe { borrow(offs, n + 1) };
    let out = unsafe { borrow_mut(out, n) };
    crate::text::text_affix_scan(data, offs, needle, true, out)
}

/// FFI: anchored affix hit per row (`TEXT endswith`, u8 0/1).
/// Same contract as [`nf_text_startswith`] (anchored at the row tail).
#[no_mangle]
pub unsafe extern "C" fn nf_text_endswith(
    data: *const u8,
    total: usize,
    offs: *const i32,
    n: usize,
    needle: *const u8,
    needle_len: usize,
    out: *mut u8,
) -> i32 {
    if !non_null(needle) && needle_len > 0 {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above (empty needle: no read).
    let needle = unsafe { borrow(needle, needle_len) };
    if crate::text::utf8_count_validated(needle).is_none() {
        return BAD_RANGE;
    }
    if n == 0 {
        return OK;
    }
    if !non_null(data) || !non_null(offs) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let data = unsafe { borrow(data, total) };
    let offs = unsafe { borrow(offs, n + 1) };
    let out = unsafe { borrow_mut(out, n) };
    crate::text::text_affix_scan(data, offs, needle, false, out)
}

// --- UTF-8 dictionary dedup (additive). Frozen ABI above untouched.
// Caller-owned buffers, all contiguous. CPU-only (no wasm export).

/// FFI: UTF-8 dictionary dedup over Arrow-like buffers.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `data[0..data_len]`: concatenated UTF-8 bytes
/// - `offs[0..offs_len]`: int32 row boundaries (offs_len == n+1)
/// - `valid[0..n]` u8 (0 = invalid): validity bitmap; null pointer = all valid
/// - `n`: number of rows
/// - `codes_out[0..n]`: int32 output, overwritten (0 = invalid row)
/// - `uniq_data_out[0..?]`: caller-allocated buffer for sorted unique bytes
/// - `uniq_offs_out[0..ng+1]`: caller-allocated int32 offsets
/// - `ng_out`: output, number of unique valid strings
/// - returns 0 ok, -1 null pointer, -3 bad offsets geometry.
#[no_mangle]
pub unsafe extern "C" fn nf_unique_dict_utf8(
    data: *const u8,
    data_len: usize,
    offs: *const i32,
    offs_len: usize,
    valid: *const u8,
    n: usize,
    codes_out: *mut i32,
    uniq_data_out: *mut u8,
    uniq_offs_out: *mut i32,
    ng_out: *mut i32,
) -> i32 {
    if !non_null(data) || !non_null(offs) || !non_null(codes_out)
        || !non_null(uniq_data_out) || !non_null(uniq_offs_out) || !non_null(ng_out)
    {
        return NULL_OR_ABORT;
    }
    if offs_len != n + 1 {
        return -1;
    }
    if n > i32::MAX as usize {
        return -1;
    }
    if data_len > i32::MAX as usize {
        return -1;
    }
    // SAFETY: upheld by caller contract above.
    let data = unsafe { borrow(data, data_len) };
    let offs = unsafe { borrow(offs, offs_len) };
    let codes_out = unsafe { borrow_mut(codes_out, n) };
    let uniq_data_out = unsafe { borrow_mut(uniq_data_out, data_len) };
    let uniq_offs_out = unsafe { borrow_mut(uniq_offs_out, n + 1) };
    let valid_slice = if valid.is_null() {
        None
    } else {
        Some(unsafe { borrow(valid, n) })
    };
    let ng = crate::unique::unique_dict_utf8(
        data, offs, valid_slice, n,
        codes_out, uniq_data_out, uniq_offs_out,
    );
    if ng < 0 {
        return -1;
    }
    unsafe { *ng_out = ng; }
    OK
}

/// FFI: byte-exact row equality per row (`TEXT equals`, u8 0/1).
/// Same contract as [`nf_text_startswith`] (full-row `==`).
#[no_mangle]
pub unsafe extern "C" fn nf_text_equals(
    data: *const u8,
    total: usize,
    offs: *const i32,
    n: usize,
    needle: *const u8,
    needle_len: usize,
    out: *mut u8,
) -> i32 {
    if !non_null(needle) && needle_len > 0 {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above (empty needle: no read).
    let needle = unsafe { borrow(needle, needle_len) };
    if crate::text::utf8_count_validated(needle).is_none() {
        return BAD_RANGE;
    }
    if n == 0 {
        return OK;
    }
    if !non_null(data) || !non_null(offs) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let data = unsafe { borrow(data, total) };
    let offs = unsafe { borrow(offs, n + 1) };
    let out = unsafe { borrow_mut(out, n) };
    crate::text::text_equals_scan(data, offs, needle, out)
}

// --- Segmented P1 (bounds-only reduce, additive). Frozen ABI above
// untouched. Generic bounds reduce: no RoadGraph/SSSP/router vocabulary,
// no WASM export (native CPU research surface), no GPU path.
// Codes (this surface only): 0 ok, -1 null pointer, -2 bad range
// (non-monotone bounds, bounds[m] != n, lane outside [0, n],
// n == 0 with m != 0, min/max over an empty group, count/mean/unknown
// op on a reduce symbol), -3 bad geometry (n > 4194240).

/// FFI (P1): u32 `count` lane over monotone bounds.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `bounds[0..m+1]`: u32 lanes, nondecreasing, inside `[0, n]`,
///   `bounds[m] == n`; `n <= 4194240` (chunkable outside)
/// - `n == 0` requires `m == 0` (`m == 0 && n == 0` returns 0
///   without dereferencing buffers)
/// - `out[0..m]`: u32 output, overwritten with exact diffs
/// - returns 0 ok, -1 null pointer, -2 bad range, -3 bad geometry.
#[no_mangle]
pub unsafe extern "C" fn nf_segment_count(
    bounds: *const u32,
    m: usize,
    n: usize,
    out: *mut u32,
) -> i32 {
    if m == 0 && n == 0 {
        return OK;
    }
    if !non_null(bounds) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let bounds = unsafe { borrow(bounds, m + 1) };
    let out = unsafe { borrow_mut(out, m) };
    segment_count(bounds, n, out)
}

macro_rules! segment_reduce_ffi {
    ($name:ident, $ty:ty, $kernel:path, $doc:expr) => {
        #[doc = $doc]
        ///
        /// Contract (caller-owned buffers, all contiguous):
        /// - `values[0..n]`: input lanes (read-only, never mutated)
        /// - `bounds[0..m+1]`: u32 lanes, nondecreasing, inside `[0, n]`,
        ///   `bounds[m] == n`; `n <= 4194240` (chunkable outside)
        /// - `n == 0` requires `m == 0` (`m == 0 && n == 0` returns 0
        ///   without dereferencing buffers)
        /// - `op`: 0 sum, 2 min, 3 max; 1 (count lives in the u32 lane,
        ///   use [`nf_segment_count`]) and anything else (including a
        ///   `mean` request) returns -2, never garbage
        /// - `out[0..m]`: output, overwritten; min/max over an empty
        ///   group returns -2 leaving `out` untouched
        /// - returns 0 ok, -1 null pointer, -2 bad range, -3 bad geometry.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            values: *const $ty,
            n: usize,
            bounds: *const u32,
            m: usize,
            op: u32,
            out: *mut $ty,
        ) -> i32 {
            if m == 0 && n == 0 {
                return OK;
            }
            if !non_null(values) || !non_null(bounds) || !non_null(out) {
                return NULL_OR_ABORT;
            }
            // SAFETY: upheld by caller contract above.
            let values = unsafe { borrow(values, n) };
            let bounds = unsafe { borrow(bounds, m + 1) };
            let out = unsafe { borrow_mut(out, m) };
            $kernel(values, bounds, op, out)
        }
    };
}

segment_reduce_ffi!(
    nf_segment_reduce_f32,
    f32,
    crate::segmented::segment_reduce_f32,
    "FFI (P1): bounds reduce over float32 lanes (sum IEEE-propagate, min/max NaN-propagate)."
);

segment_reduce_ffi!(
    nf_segment_reduce_i32,
    i32,
    crate::segmented::segment_reduce_i32,
    "FFI (P1): bounds reduce over int32 lanes (sum saturates, never wraps)."
);

// --- Adjacency P2 (CSR slice + flat gather, additive). Frozen ABI above
// untouched. Generic CSR surface: no RoadGraph/SSSP/router vocabulary,
// no WASM export (native CPU research surface), no GPU path.
// Codes (this surface only): 0 ok, -1 null pointer, -2 bad range
// (UINT32_MAX reserved lane, non-monotone indptr, out-of-range query,
// non-empty query on an empty graph, begins > ends, ends > E),
// -3 bad geometry (indptr[V] != E, lane outside [0, E], non-empty
// indices with empty indptr, short gather output).

/// FFI (P2): CSR slice `indptr/indices/query` -> `begins/ends`.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `indptr[0..np]`: u32 lanes; `np == 0` is the empty-graph encoding
///   (requires `e == 0`), else `np == V + 1`, nondecreasing, inside
///   `[0, e]`, `indptr[V] == e`
/// - `indices[0..e]`: u32 lanes (content unread, reserved-lane checked)
/// - `query[0..k]`: u32 vertices, each `< V`
/// - `begins/ends[0..k]`: u32 outputs, overwritten with
///   `indptr[q]` / `indptr[q+1]` in query order
/// - `k == 0` enforces only the reserved-lane rule (no structural
///   checks, zero-length outputs may be null)
/// - returns 0 ok, -1 null pointer, -2 bad range, -3 bad geometry.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_adjacency_slice(
    indptr: *const u32,
    np: usize,
    indices: *const u32,
    e: usize,
    query: *const u32,
    k: usize,
    begins: *mut u32,
    ends: *mut u32,
) -> i32 {
    if k == 0 {
        if (np > 0 && !non_null(indptr)) || (e > 0 && !non_null(indices)) {
            return NULL_OR_ABORT;
        }
        // SAFETY: upheld by caller contract above (len-0 borrows are null-safe).
        let indptr = unsafe { borrow(indptr, np) };
        let indices = unsafe { borrow(indices, e) };
        let query = unsafe { borrow(query, 0) };
        let begins = unsafe { borrow_mut(begins, 0) };
        let ends = unsafe { borrow_mut(ends, 0) };
        return adjacency_slice(indptr, indices, query, begins, ends);
    }
    if !non_null(indptr)
        || !non_null(indices)
        || !non_null(query)
        || !non_null(begins)
        || !non_null(ends)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let indptr = unsafe { borrow(indptr, np) };
    let indices = unsafe { borrow(indices, e) };
    let query = unsafe { borrow(query, k) };
    let begins = unsafe { borrow_mut(begins, k) };
    let ends = unsafe { borrow_mut(ends, k) };
    adjacency_slice(indptr, indices, query, begins, ends)
}

/// FFI (P2): flat gather of `indices[begins[g]..ends[g]]` in query
/// order (storage order inside each slice, deterministic).
///
/// Contract (caller-owned buffers, all contiguous):
/// - `indices[0..e]`: u32 lanes (read-only, carried verbatim)
/// - `begins/ends[0..k]`: u32 pairs, `begins[g] <= ends[g] <= e`
///   (validated first: outputs untouched on error)
/// - `out[0..total]`: u32 output, first `sum(ends - begins)` lanes
///   overwritten (`total` must cover it)
/// - `k == 0` returns 0 without dereferencing buffers
/// - returns 0 ok, -1 null pointer, -2 bad range, -3 bad geometry.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_adjacency_gather(
    indices: *const u32,
    e: usize,
    begins: *const u32,
    ends: *const u32,
    k: usize,
    out: *mut u32,
    total: usize,
) -> i32 {
    if k == 0 {
        return OK;
    }
    if !non_null(indices) || !non_null(begins) || !non_null(ends) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let indices = unsafe { borrow(indices, e) };
    let begins = unsafe { borrow(begins, k) };
    let ends = unsafe { borrow(ends, k) };
    let out = unsafe { borrow_mut(out, total) };
    adjacency_gather(indices, begins, ends, out)
}

// --- Cost P3 (generic integer costing core, additive). Frozen ABI above
// untouched. Generic cost surface: no RoadGraph/TBRS/router/dispatcher/
// hierarchy/foam vocabulary, no WASM export (native CPU research
// surface), no GPU path. Codes (this surface only): 0 ok, -1 null
// pointer, -2 bad range (K == 0, width == 0 with n > 0), -3 bad
// geometry (short uniq buffer), -4 truck slot (profile 255, model not
// invented here). Batch lanes are directed rows (A->B / B->A separate,
// never merged) with no MIN collapse across parallel rows.

/// FFI (P3): batch integer costing `out[i] = (dist[i] * k[i] +
/// speed[i] / 2) / speed[i]` with INF guard.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `dist[0..n]`: u32 mm, traffic-independent (never scaled by bucket)
/// - `speed[0..n]`: u32 mm/s (`0 -> INF`, `INF -> INF`, no divide)
/// - `k[0..n]`: u16 permille (`K_SCALE = 1000` nominal; `0 -> -2`)
/// - `out[0..n]`: u32 ms output, overwritten (`< 1 && length > 0 -> 1`,
///   `>= UINT32_MAX -> INF`)
/// - `n == 0` returns 0 without dereferencing buffers
/// - returns 0 ok, -1 null pointer, -2 bad range (K == 0 lane; lanes
///   before it may be written).
#[no_mangle]
pub unsafe extern "C" fn nf_cost_travel_batch(
    dist: *const u32,
    speed: *const u32,
    k: *const u16,
    n: usize,
    out: *mut u32,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(dist) || !non_null(speed) || !non_null(k) || !non_null(out) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let dist = unsafe { borrow(dist, n) };
    let speed = unsafe { borrow(speed, n) };
    let k = unsafe { borrow(k, n) };
    let out = unsafe { borrow_mut(out, n) };
    cost_travel_batch(dist, speed, k, out)
}

/// FFI (P3): stateless row interning (compact multidimensional costs).
///
/// Contract (caller-owned buffers, all contiguous):
/// - `vecs[0..n*width]`: u32 row-major cost vectors (e.g. per-bucket
///   travel times; one directed edge per row, parallel rows separate)
/// - `ids[0..n]`: u32 outputs, overwritten with first-appearance
///   `cost_id`s (identical vectors share one id, different vectors get
///   different ids -- never MIN-collapsed, never merged across rows)
/// - `uniq[0..total]`: u32 output, first `ng * width` lanes overwritten
///   with the compact table in first-appearance order (`total` must
///   cover `n * width`)
/// - `n == 0` returns 0 without dereferencing buffers
/// - returns `ng` (compact rows, >= 0); -2 null pointer / zero width
///   with `n > 0`; -3 bad geometry (short `uniq` buffer).
#[no_mangle]
pub unsafe extern "C" fn nf_cost_intern(
    vecs: *const u32,
    n: usize,
    width: usize,
    ids: *mut u32,
    uniq: *mut u32,
    total: usize,
) -> i64 {
    if n == 0 {
        return 0;
    }
    if !non_null(vecs) || !non_null(ids) || !non_null(uniq) {
        return NULL_STATE;
    }
    if width == 0 {
        return BAD_RANGE as i64;
    }
    let need = n.saturating_mul(width);
    if total < need {
        return crate::core::errors::MALFORMED as i64;
    }
    // SAFETY: upheld by caller contract above.
    let vecs = unsafe { borrow(vecs, need) };
    let ids = unsafe { borrow_mut(ids, n) };
    let uniq = unsafe { borrow_mut(uniq, total) };
    cost_intern_rows(vecs, n, width, ids, uniq)
}

// --- GroupedHashMT native port (additive; frozen ABI above untouched).
// Native port of the proven Numba GroupedHashMT lane (partitioned
// pair open-addressing hash + fingerprint routing + MT + reusable
// caller-owned buffers). int64 lanes; narrower integer dtypes widen
// exactly caller-side, uint64 past i64::MAX never reaches here.
// Old Numba lane stays intact as fallback/reference. Return codes
// (this ABI only): 0 ok, -1 null pointer, -3 bad geometry
// (`p` not a power of two, `p == 0`, `p > 256`).

/// FFI: MT pair fingerprint, rows -> partition ids.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `k/v[0..n]`: int64 keys/values (read-only, never mutated)
/// - `fp[0..n]`: int32... int64 output, overwritten (`(h & pmask)`)
/// - `pmask`: `P - 1` (`P` power of two, `16 <= P <= 256`)
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1); `n == 0` returns 0
///   without dereferencing buffers
/// - returns 0 ok, -1 null pointer.
#[no_mangle]
pub unsafe extern "C" fn nf_ghash_fp_i64(
    k: *const i64,
    v: *const i64,
    n: usize,
    fp: *mut i64,
    pmask: i64,
    nthreads: usize,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(k) || !non_null(v) || !non_null(fp) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let k = unsafe { borrow(k, n) };
    let v = unsafe { borrow(v, n) };
    let fp = unsafe { borrow_mut(fp, n) };
    gh_fp(k, v, n, fp, pmask as u64, nthreads)
}

/// FFI: MT per-thread histogram, partition ids -> counts.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `fp[0..n]`: int64 partition ids in `[0, p)` (read-only)
/// - `tc[0..t*p]` int64: output, thread `tt` owns row `tt`
///   (WARM-zeroed by the caller, accumulated into)
/// - `t`: fan-out rows (matches the scatter pass), `p`: partitions
///   (power of two, `1 <= p <= 256`)
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1); `n == 0` returns 0
///   without dereferencing buffers
/// - returns 0 ok, -1 null pointer, -3 bad geometry.
#[no_mangle]
pub unsafe extern "C" fn nf_ghash_count_i64(
    fp: *const i64,
    n: usize,
    tc: *mut i64,
    t: usize,
    p: usize,
    nthreads: usize,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(fp) || !non_null(tc) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let fp = unsafe { borrow(fp, n) };
    let tc = unsafe { borrow_mut(tc, t * p) };
    gh_count(fp, n, tc, t, p, nthreads)
}

/// FFI: MT reservation scatter, rows -> partitioned (pk, pv) lanes.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `k/v/fp[0..n]`: int64 keys/values + partition ids in `[0, p)`
///   (read-only, never mutated)
/// - `tpos[0..t*p]` int64: caller prefix (thread `tt` owns
///   `tpos[tt * p + q]` bases — same tiling as the count pass)
/// - `t`/`p`: fan-out rows / partition count (power of two)
/// - `pk/pv[0..n]`: outputs, overwritten (per-partition SET is
///   thread-count invariant; order inside a partition is not output)
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1); `n == 0` returns 0
///   without dereferencing buffers
/// - returns 0 ok, -1 null pointer, -3 bad geometry.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_ghash_scatter_i64(
    k: *const i64,
    v: *const i64,
    fp: *const i64,
    n: usize,
    tpos: *const i64,
    t: usize,
    p: usize,
    pk: *mut i64,
    pv: *mut i64,
    nthreads: usize,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(k) || !non_null(v) || !non_null(fp) || !non_null(tpos) || !non_null(pk) || !non_null(pv) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let k = unsafe { borrow(k, n) };
    let v = unsafe { borrow(v, n) };
    let fp = unsafe { borrow(fp, n) };
    let tpos = unsafe { borrow(tpos, t * p) };
    let pk = unsafe { borrow_mut(pk, n) };
    let pv = unsafe { borrow_mut(pv, n) };
    gh_scatter(k, v, fp, n, tpos, t, p, pk, pv, nthreads)
}

/// FFI: MT pair-hash insert, partitioned rows -> slot tables.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `pk/pv[0..n]`: int64 scattered pairs (`n == starts[p]`)
/// - `starts/toff[0..p]`: int64 row/table-range bases
/// - `tm[0..p]`: int64 slot masks (`slots - 1`, power-of-two minus one)
/// - `tk/tv[0..m]` int64 + `used[0..m]` u8 (`m == toff[p]`):
///   outputs, `used` zeroed first (WARM), overwritten
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1)
/// - returns 0 ok, -1 null pointer, -3 bad geometry.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_ghash_pins_i64(
    pk: *const i64,
    pv: *const i64,
    starts: *const i64,
    toff: *const i64,
    tm: *const i64,
    tk: *mut i64,
    tv: *mut i64,
    used: *mut u8,
    p: usize,
    nthreads: usize,
) -> i32 {
    if !non_null(pk)
        || !non_null(pv)
        || !non_null(starts)
        || !non_null(toff)
        || !non_null(tm)
        || !non_null(tk)
        || !non_null(tv)
        || !non_null(used)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let starts = unsafe { borrow(starts, p + 1) };
    let toff = unsafe { borrow(toff, p + 1) };
    let tm = unsafe { borrow(tm, p) };
    let n = starts[p] as usize;
    let m = toff[p] as usize;
    let pk = unsafe { borrow(pk, n) };
    let pv = unsafe { borrow(pv, n) };
    let tk = unsafe { borrow_mut(tk, m) };
    let tv = unsafe { borrow_mut(tv, m) };
    let used = unsafe { borrow_mut(used, m) };
    gh_pins(pk, pv, starts, toff, tm, tk, tv, used, p, nthreads)
}

/// FFI: MT occupancy count, slot tables -> per-partition counts.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `used[0..m]` u8 + `toff[0..p]` int64 (`m == toff[p]`)
/// - `cnt[0..p]` int64: output, overwritten
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1)
/// - returns 0 ok, -1 null pointer, -3 bad geometry.
#[no_mangle]
pub unsafe extern "C" fn nf_ghash_occ_count(
    used: *const u8,
    toff: *const i64,
    p: usize,
    cnt: *mut i64,
    nthreads: usize,
) -> i32 {
    if !non_null(used) || !non_null(toff) || !non_null(cnt) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let toff = unsafe { borrow(toff, p + 1) };
    let m = toff[p] as usize;
    let used = unsafe { borrow(used, m) };
    let cnt = unsafe { borrow_mut(cnt, p) };
    gh_occ_count(used, toff, p, cnt, nthreads)
}

/// FFI: MT occupancy fill, slot tables -> occupied keys.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `tk[0..m]` int64 + `used[0..m]` u8 + `toff[0..p]` int64
/// - `ostart[0..p]` int64: prefix offsets (`ostart[p] == nocc`)
/// - `ok[0..nocc]` int64: output, overwritten (ranges disjoint
///   across threads by prefix construction)
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1)
/// - returns 0 ok, -1 null pointer, -3 bad geometry.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_ghash_occ_fill(
    tk: *const i64,
    used: *const u8,
    toff: *const i64,
    ostart: *const i64,
    ok: *mut i64,
    p: usize,
    nthreads: usize,
) -> i32 {
    if !non_null(tk) || !non_null(used) || !non_null(toff) || !non_null(ostart) || !non_null(ok) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let toff = unsafe { borrow(toff, p + 1) };
    let ostart = unsafe { borrow(ostart, p + 1) };
    let m = toff[p] as usize;
    let nocc = ostart[p] as usize;
    let tk = unsafe { borrow(tk, m) };
    let used = unsafe { borrow(used, m) };
    let ok = unsafe { borrow_mut(ok, nocc) };
    gh_occ_fill(tk, used, toff, ostart, ok, p, nthreads)
}

// --- Rowwise MIN4 + argmin + gather (additive). Frozen ABI above
// untouched. Generic rowwise surface: no query vocabulary, no WASM
// export (native CPU research surface), no GPU path.
// Codes (this surface only): 0 ok, -1 null pointer.

/// FFI: generic rowwise MIN over 4 int32 lanes keyed by 4 float32 lanes.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `t0..t3[0..n]`: int32 value lanes (read-only, carried verbatim)
/// - `d0..d3[0..n]`: float32 key lanes (read-only)
/// - `t_best[0..n]` int32, `d_best[0..n]` float32, `m_best[0..n]` u8:
///   outputs, overwritten with `m = argmin` (ties keep the smallest
///   index), `t_best = t[m]`, `d_best = d[m]` (bit-exact gather)
/// - `n == 0` returns 0 without dereferencing buffers
/// - returns 0 ok, -1 null pointer.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_rowwise_min4_argmin_gather(
    t0: *const i32,
    t1: *const i32,
    t2: *const i32,
    t3: *const i32,
    d0: *const f32,
    d1: *const f32,
    d2: *const f32,
    d3: *const f32,
    n: usize,
    t_best: *mut i32,
    d_best: *mut f32,
    m_best: *mut u8,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(t0)
        || !non_null(t1)
        || !non_null(t2)
        || !non_null(t3)
        || !non_null(d0)
        || !non_null(d1)
        || !non_null(d2)
        || !non_null(d3)
        || !non_null(t_best)
        || !non_null(d_best)
        || !non_null(m_best)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let t0 = unsafe { borrow(t0, n) };
    let t1 = unsafe { borrow(t1, n) };
    let t2 = unsafe { borrow(t2, n) };
    let t3 = unsafe { borrow(t3, n) };
    let d0 = unsafe { borrow(d0, n) };
    let d1 = unsafe { borrow(d1, n) };
    let d2 = unsafe { borrow(d2, n) };
    let d3 = unsafe { borrow(d3, n) };
    let t_best = unsafe { borrow_mut(t_best, n) };
    let d_best = unsafe { borrow_mut(d_best, n) };
    let m_best = unsafe { borrow_mut(m_best, n) };
    rowwise_min4_argmin_gather(
        t0, t1, t2, t3, d0, d1, d2, d3, n, t_best, d_best, m_best,
    );
    OK
}

// --- Rowwise MIN4 time-argmin + gather (FIX-1, additive). Frozen ABI
// above untouched, existing nf_rowwise_min4_argmin_gather untouched.
// Generic rowwise surface: no query/transport/Dispatcher/RoadGraph
// vocabulary, no N-specific or benchmark branches, no WASM export
// (native CPU research surface), no GPU path.
// Codes (this surface only): 0 ok, -1 null pointer.

/// FFI: generic rowwise MIN over 4 int32 time lanes + optional float32
/// payload gather.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `t0..t3[0..n]`: int32 time lanes (read-only)
/// - `d0..d3`: float32 payload lanes (read-only) -- OPTIONAL as a group:
///   either all four non-null (gather path) or all four null (time-only
///   path); mixed null/non-null is a contract violation (-1)
/// - `t_best[0..n]` int32, `m_best[0..n]` u8: outputs, always written
/// - `d_best`: float32 output -- non-null on the gather path (written),
///   null on the time-only path (untouched, never dereferenced)
/// - semantics both paths: `m = argmin_k(t[k])` strict `<` from lane 0
///   (ties keep the smallest index), `t_best = t[m]` exact; gather path
///   adds `d_best = d[m]` bit-exact (D never selects the mode)
/// - `n == 0` returns 0 without dereferencing buffers
/// - returns 0 ok, -1 null pointer / contract violation.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_rowwise_min4_time_argmin_gather(
    t0: *const i32,
    t1: *const i32,
    t2: *const i32,
    t3: *const i32,
    d0: *const f32,
    d1: *const f32,
    d2: *const f32,
    d3: *const f32,
    n: usize,
    t_best: *mut i32,
    d_best: *mut f32,
    m_best: *mut u8,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(t0)
        || !non_null(t1)
        || !non_null(t2)
        || !non_null(t3)
        || !non_null(t_best)
        || !non_null(m_best)
    {
        return NULL_OR_ABORT;
    }
    let d_all = non_null(d0) && non_null(d1) && non_null(d2) && non_null(d3);
    let d_none = !non_null(d0) && !non_null(d1) && !non_null(d2) && !non_null(d3);
    if d_all {
        if !non_null(d_best) {
            return NULL_OR_ABORT;
        }
        // SAFETY: upheld by caller contract above.
        let t0 = unsafe { borrow(t0, n) };
        let t1 = unsafe { borrow(t1, n) };
        let t2 = unsafe { borrow(t2, n) };
        let t3 = unsafe { borrow(t3, n) };
        let d0 = unsafe { borrow(d0, n) };
        let d1 = unsafe { borrow(d1, n) };
        let d2 = unsafe { borrow(d2, n) };
        let d3 = unsafe { borrow(d3, n) };
        let t_best = unsafe { borrow_mut(t_best, n) };
        let d_best = unsafe { borrow_mut(d_best, n) };
        let m_best = unsafe { borrow_mut(m_best, n) };
        rowwise_min4_time_argmin_gather(
            t0, t1, t2, t3, d0, d1, d2, d3, n, t_best, d_best, m_best,
        );
        return OK;
    }
    if d_none && !non_null(d_best) {
        // SAFETY: upheld by caller contract above.
        let t0 = unsafe { borrow(t0, n) };
        let t1 = unsafe { borrow(t1, n) };
        let t2 = unsafe { borrow(t2, n) };
        let t3 = unsafe { borrow(t3, n) };
        let t_best = unsafe { borrow_mut(t_best, n) };
        let m_best = unsafe { borrow_mut(m_best, n) };
        rowwise_min4_time_argmin(t0, t1, t2, t3, n, t_best, m_best);
        return OK;
    }
    NULL_OR_ABORT
}

// --- Rowwise K-way time-argmin + gather (canonical, additive). Frozen ABI
// above untouched, existing K4 symbols kept as thin compat wrappers.
// Generic rowwise surface: no N-specific or
// benchmark branches, no WASM export (native CPU research surface),
// no GPU path.
// Codes (this surface only): 0 ok, -1 null pointer, -2 bad range
// (k == 0 or k > 256, m_best is u8).

/// FFI (canonical): generic K-way rowwise MIN over int32 selector lanes
/// + float32 payload gather.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `t_ptrs[0..k]`: pointers to int32 selector lanes, each `len >= n`
///   (read-only)
/// - `d_ptrs[0..k]`: pointers to float32 payload lanes, each `len >= n`
///   (read-only, pure payload, never compared)
/// - `k`: lane count, `1 <= k <= 256` (else -2, outputs untouched)
/// - `t_best[0..n]` int32, `d_best[0..n]` float32, `m_best[0..n]` u8:
///   outputs, overwritten with `m = argmin_k(t[k])` strict `<` from
///   lane 0 (ties keep the smallest index), `t_best = t[m]` exact,
///   `d_best = d[m]` bit-exact
/// - `n == 0` returns 0 without dereferencing buffers
/// - returns 0 ok, -1 null pointer, -2 bad range.
#[no_mangle]
pub unsafe extern "C" fn nf_rowwise_kway_time_argmin_gather(
    t_ptrs: *const *const i32,
    d_ptrs: *const *const f32,
    k: usize,
    n: usize,
    t_best: *mut i32,
    d_best: *mut f32,
    m_best: *mut u8,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(t_ptrs) || !non_null(d_ptrs) || !non_null(t_best) || !non_null(d_best) || !non_null(m_best) {
        return NULL_OR_ABORT;
    }
    if k == 0 || k > 256 {
        return BAD_RANGE;
    }
    // SAFETY: upheld by caller contract above.
    let t_ptr_arr = unsafe { borrow(t_ptrs, k) };
    let d_ptr_arr = unsafe { borrow(d_ptrs, k) };
    let mut t_lanes: [&[i32]; 256] = [&[]; 256];
    let mut d_lanes: [&[f32]; 256] = [&[]; 256];
    for lane in 0..k {
        if t_ptr_arr[lane].is_null() || d_ptr_arr[lane].is_null() {
            return NULL_OR_ABORT;
        }
        // SAFETY: each lane pointer covers `n` lanes per contract.
        t_lanes[lane] = unsafe { borrow(t_ptr_arr[lane], n) };
        d_lanes[lane] = unsafe { borrow(d_ptr_arr[lane], n) };
    }
    let t_best = unsafe { borrow_mut(t_best, n) };
    let d_best = unsafe { borrow_mut(d_best, n) };
    let m_best = unsafe { borrow_mut(m_best, n) };
    rowwise_kway_time_argmin_gather(&t_lanes[..k], &d_lanes[..k], n, t_best, d_best, m_best);
    OK
}

// --- Bounded select P4 (generic paired top-K, additive). Frozen ABI above
// untouched. Generic surface: inputs are `a` (primary) and `b`
// (secondary) int32 lanes + CSR `offs`; no domain/query vocabulary,
// no K-specific or N-specific branches, no WASM export (native CPU
// research surface), no GPU path.
// Codes (this surface only): 0 ok, -1 null pointer, -2 bad range
// (k == 0), -3 bad geometry (non-monotone offs, offs[g] != n,
// offs[0] != 0).

/// FFI (P4): per-group bounded top-K over paired int32 keys.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `a/b[0..n]`: int32 lanes (read-only, never mutated); infeasible
///   rows are exactly `a == INT32_MAX` (no mask input, they sink by order)
/// - `offs[0..g]`: int32 CSR bounds, nondecreasing, `offs[0] == 0`,
///   `offs[g] == n` (`g == 0` requires `n == 0`)
/// - `k`: runtime bound, `>= 1` (caller allocates `g*k` lanes; `K > 32`
///   supported, same code path — no hardcoded K)
/// - `out_a/out_b/out_idx[0..g*k]`: outputs, overwritten per group in
///   `(a, b, idx)` ascending order; short groups pad the tail with
///   `(INT32_MAX, INT32_MAX, -1)`; all-`INF` group => all lanes `INF`
/// - `n == 0 && g == 0` returns 0 without dereferencing buffers
/// - returns 0 ok, -1 null pointer, -2 bad range, -3 bad geometry.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_bounded_select_2i32(
    a: *const i32,
    b: *const i32,
    n: usize,
    offs: *const i32,
    g: usize,
    k: usize,
    out_a: *mut i32,
    out_b: *mut i32,
    out_idx: *mut i32,
) -> i32 {
    if n == 0 && g == 0 {
        return OK;
    }
    if !non_null(a) || !non_null(b) || !non_null(offs) || !non_null(out_a) || !non_null(out_b) || !non_null(out_idx) {
        return NULL_OR_ABORT;
    }
    if k == 0 {
        return BAD_RANGE;
    }
    // SAFETY: upheld by caller contract above.
    let a = unsafe { borrow(a, n) };
    let b = unsafe { borrow(b, n) };
    let offs = unsafe { borrow(offs, g + 1) };
    let total = g.saturating_mul(k);
    let out_a = unsafe { borrow_mut(out_a, total) };
    let out_b = unsafe { borrow_mut(out_b, total) };
    let out_idx = unsafe { borrow_mut(out_idx, total) };
    crate::bounded_select::bounded_select_2i32(a, b, offs, g, k, out_a, out_b, out_idx)
}

// --- Join int64 lane (additive, LDBC ids exceed int32). Frozen int32
// `nf_join_build` / `nf_join_probe` untouched. Caller-owned buffers,
// all contiguous. Native-only (no wasm export: probe uses scoped host
// threads). Gather/materialize stay NumPy caller-side.

/// FFI: build the right hash table over int64 keys (unique contract).
///
/// Contract (caller-owned buffers, all contiguous):
/// - `right_keys[0..s]`: int64 right keys (full range incl MIN/MAX)
/// - `table_keys[0..cap]` int64, `table_pos[0..cap]` int32,
///   `table_occ[0..cap]` u8: outputs, `occ` zeroed first,
///   `pos` = right row position
/// - `cap`: power of two, `cap >= 2*s` (`s == 0` returns 0
///   without touching table lanes)
/// - returns 0 ok, -1 null pointer, -2 duplicate right key
///   (lanes before it written, caller must rebuild), -3 bad geometry.
#[no_mangle]
pub unsafe extern "C" fn nf_join_build_i64(
    right_keys: *const i64,
    s: usize,
    table_keys: *mut i64,
    table_pos: *mut i32,
    table_occ: *mut u8,
    cap: usize,
) -> i32 {
    if s == 0 {
        return OK;
    }
    if !non_null(right_keys) || !non_null(table_keys) || !non_null(table_pos) || !non_null(table_occ)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let right_keys = unsafe { borrow(right_keys, s) };
    let table_keys = unsafe { borrow_mut(table_keys, cap) };
    let table_pos = unsafe { borrow_mut(table_pos, cap) };
    let table_occ = unsafe { borrow_mut(table_occ, cap) };
    crate::join_i64::join_build_i64(right_keys, s, table_keys, table_pos, table_occ, cap)
}

/// FFI: MT hash probe of int64 left keys against a built table.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `table_keys[0..cap]` int64, `table_pos[0..cap]` int32,
///   `table_occ[0..cap]` u8: built table, read-only
/// - `left_keys[0..n]`: int64 probe keys (full range incl MIN/MAX)
/// - `pos_out[0..n]` int64 (match position, 0 on miss),
///   `hit_out[0..n]` u8 0/1: outputs, overwritten in input order
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1); `n == 0` returns 0
///   without dereferencing buffers
/// - returns 0 ok, -1 null pointer, -3 bad geometry (cap not pow2).
#[no_mangle]
pub unsafe extern "C" fn nf_join_probe_i64(
    table_keys: *const i64,
    table_pos: *const i32,
    table_occ: *const u8,
    cap: usize,
    left_keys: *const i64,
    n: usize,
    pos_out: *mut i64,
    hit_out: *mut u8,
    nthreads: usize,
) -> i32 {
    if n == 0 {
        return OK;
    }
    if !non_null(table_keys)
        || !non_null(table_pos)
        || !non_null(table_occ)
        || !non_null(left_keys)
        || !non_null(pos_out)
        || !non_null(hit_out)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let table_keys = unsafe { borrow(table_keys, cap) };
    let table_pos = unsafe { borrow(table_pos, cap) };
    let table_occ = unsafe { borrow(table_occ, cap) };
    let left_keys = unsafe { borrow(left_keys, n) };
    let pos_out = unsafe { borrow_mut(pos_out, n) };
    let hit_out = unsafe { borrow_mut(hit_out, n) };
    crate::join_i64::join_probe_i64(
        table_keys, table_pos, table_occ, cap, left_keys, n, pos_out, hit_out, nthreads,
    )
}

// --- SSSP (canonical generic single-source shortest path, additive).
// Frozen ABI above untouched. Generic CSR surface: graph vertices /
// u32 weights / sources only, no domain vocabulary, no WASM export
// (native CPU research surface), no GPU path.
// Codes (this surface only): 0 ok, -1 null pointer, -2 bad range
// (UINT32_MAX reserved lane, out-of-range index/source, non-monotone
// indptr, empty graph), -3 bad geometry (lane outside [0, E],
// indptr[V] != E, short output).

/// FFI: canonical `sssp_csr` — CSR + weights + source -> dist.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `indptr[0..np]`: u32 lanes, `np == V + 1` (`V >= 1`), nondecreasing,
///   inside `[0, e]`, `indptr[V] == e`
/// - `indices[0..e]` u32 in `[0, V)`, `weights[0..e]` u32
///   (`UINT32_MAX` edge skipped, never relaxed)
/// - `source`: u32 in `[0, V)` (`UINT32_MAX` rejected)
/// - `dist[0..v]` u32 output, overwritten (`UINT32_MAX` = unreachable)
/// - returns 0 ok, -1 null pointer, -2 bad range, -3 bad geometry.
#[no_mangle]
pub unsafe extern "C" fn nf_sssp_csr(
    indptr: *const u32,
    np: usize,
    indices: *const u32,
    weights: *const u32,
    e: usize,
    source: u32,
    dist: *mut u32,
) -> i32 {
    if !non_null(indptr) || !non_null(indices) || !non_null(weights) || !non_null(dist) {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let indptr = unsafe { borrow(indptr, np) };
    let indices = unsafe { borrow(indices, e) };
    let weights = unsafe { borrow(weights, e) };
    let v = np.saturating_sub(1);
    let dist = unsafe { borrow_mut(dist, v) };
    crate::sssp::sssp_csr(indptr, indices, weights, source, dist, None)
}

/// FFI: `sssp_csr` with predecessors — same contract plus
/// `pred[0..v]` i32 output, overwritten (`-1` = root/unreached,
/// else parent vertex id on the relaxed shortest-path tree).
#[no_mangle]
pub unsafe extern "C" fn nf_sssp_csr_pred(
    indptr: *const u32,
    np: usize,
    indices: *const u32,
    weights: *const u32,
    e: usize,
    source: u32,
    dist: *mut u32,
    pred: *mut i32,
) -> i32 {
    if !non_null(indptr)
        || !non_null(indices)
        || !non_null(weights)
        || !non_null(dist)
        || !non_null(pred)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let indptr = unsafe { borrow(indptr, np) };
    let indices = unsafe { borrow(indices, e) };
    let weights = unsafe { borrow(weights, e) };
    let v = np.saturating_sub(1);
    let dist = unsafe { borrow_mut(dist, v) };
    let pred = unsafe { borrow_mut(pred, v) };
    // i32 lanes viewed as u32 (bit-identical, `-1` == INF sentinel).
    let pred_u: &mut [u32] =
        unsafe { std::slice::from_raw_parts_mut(pred.as_mut_ptr() as *mut u32, v) };
    crate::sssp::sssp_csr(indptr, indices, weights, source, dist, Some(pred_u))
}

/// FFI: `sssp_batch` — same graph + `sources[K]` -> `out[K*V]`
/// row-major (`out[r*V..(r+1)*V]` holds dist from `sources[r]`).
///
/// Contract (caller-owned buffers, all contiguous):
/// - graph lanes same as [`nf_sssp_csr`]; `sources[0..k]` u32 in `[0, V)`
/// - `out[0..k*v]` u32 output, overwritten in source order
///   (thread-count invariant)
/// - `nthreads`: clamped to `[1, 64]` (0 -> 1); `k == 0` returns 0
///   without dereferencing `sources`/`out`
/// - returns 0 ok, -1 null pointer, -2 bad range, -3 bad geometry.
#[no_mangle]
pub unsafe extern "C" fn nf_sssp_batch(
    indptr: *const u32,
    np: usize,
    indices: *const u32,
    weights: *const u32,
    e: usize,
    sources: *const u32,
    k: usize,
    out: *mut u32,
    nthreads: usize,
) -> i32 {
    if k == 0 {
        return OK;
    }
    if !non_null(indptr)
        || !non_null(indices)
        || !non_null(weights)
        || !non_null(sources)
        || !non_null(out)
    {
        return NULL_OR_ABORT;
    }
    // SAFETY: upheld by caller contract above.
    let indptr = unsafe { borrow(indptr, np) };
    let indices = unsafe { borrow(indices, e) };
    let weights = unsafe { borrow(weights, e) };
    let sources = unsafe { borrow(sources, k) };
    let v = np.saturating_sub(1);
    let out = unsafe { borrow_mut(out, k.saturating_mul(v)) };
    crate::sssp::sssp_batch(indptr, indices, weights, sources, k, out, nthreads)
}

// --- PairInsert (generic batch-assignment pair insert, additive). Frozen
// ABI above untouched. Generic typeless 64-bit lanes (caller views them
// as u64 or i64, bit-identical): no domain vocabulary, no WASM export
// (native CPU research surface, same gate as `sssp`), no GPU path,
// single-threaded (ST reference owns the order: input order, last-write
// -wins on duplicate keys). Reuses `core::hash` probe atoms (same family
// as `join_build`); `core`-only (no alloc, no collections).
// Codes (this surface only): `ng >= 0` occupied slots; -2 null-state
// (like the sorted/carry lane-returning symbols); -3 bad geometry (cap
// not pow2, cap == 0, short table).

/// FFI: generic open-addressing insert of `(key, value)` i64 pairs.
///
/// Contract (caller-owned buffers, all contiguous):
/// - `keys/vals[0..n]`: int64 lanes (read-only, full range incl MIN/MAX,
///   typeless bits: u64 callers pass through bit-identical);
/// - `tkeys/tvals/used[0..cap]`: outputs, `cap` power of two, `cap >= 1`
///   (`used` u8 0/1, zeroed first/WARM, then overwritten);
/// - duplicate key = last-write-wins (value overwritten, occupancy
///   unchanged), never an error;
/// - `n == 0` returns 0 after zeroing `used` (still dereferences table
///   lanes for the WARM fill);
/// - returns `ng` (occupied slots, >= 0); -2 null-state; -3 bad geometry.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nf_pair_insert_i64(
    keys: *const i64,
    vals: *const i64,
    n: usize,
    tkeys: *mut i64,
    tvals: *mut i64,
    used: *mut u8,
    cap: usize,
) -> i64 {
    if !non_null(keys) || !non_null(vals) || !non_null(tkeys) || !non_null(tvals) || !non_null(used)
    {
        return NULL_STATE;
    }
    if !cap.is_power_of_two() || cap == 0 {
        return crate::pair_insert::PAIR_GEOM;
    }
    // SAFETY: upheld by caller contract above.
    let keys = unsafe { borrow(keys, n) };
    let vals = unsafe { borrow(vals, n) };
    let tkeys = unsafe { borrow_mut(tkeys, cap) };
    let tvals = unsafe { borrow_mut(tvals, cap) };
    let used = unsafe { borrow_mut(used, cap) };
    crate::pair_insert::pair_insert_i64(keys, vals, n, tkeys, tvals, used, cap)
}
