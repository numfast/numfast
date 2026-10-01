// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Return-code table. Values frozen by the C ABI — do not renumber.
//!
//! | code | meaning per symbol |
//! |------|--------------------|
//! | `0` | ok (or `ng >= 0` count where the symbol returns lanes) |
//! | `-1` | null pointer (`nf_group_*`, `nf_pack_*`, `nf_pattern_encode`); unsorted abort (`nf_sorted_run_*`) |
//! | `-2` | out-of-range key (`nf_group_multi_*`); negative input / int32 overflow (`nf_pack_*`); int32 overflow (`nf_pattern_encode`); null pointer (`nf_sorted_run_*`, `nf_carry_build_*`) |
//! | `-3` | malformed offsets (`nf_pattern_encode`) |

/// Success.
pub const OK: i32 = 0;
/// Null pointer on the 4 buffer-style symbols; unsorted-input abort on
/// the sorted-run symbols.
pub const NULL_OR_ABORT: i32 = -1;
/// Out-of-range key / overflow / null-state, per symbol (see table).
pub const BAD_RANGE: i32 = -2;
/// Malformed offsets (`nf_pattern_encode` only).
pub const MALFORMED: i32 = -3;
/// Null state pointer (`nf_sorted_run_*`, `nf_carry_build_*`).
pub const NULL_STATE: i64 = -2;
/// Unsorted input, speculative abort (`nf_sorted_run_*`).
pub const UNSORTED_ABORT: i64 = -1;

// --- Research-only codes (new `nf_group_variant_*` symbols only).
// Existing codes above are frozen and untouched.

/// Research-only: integer overflow, `checked_add` failed
/// (`nf_group_variant_i64/i32`). Explicit error, never wraps.
pub const OVERFLOW: i32 = -4;
/// Research-only: unsorted input on a `V_SORTED` variant
/// (`nf_group_variant_*`). Speculative abort, same rule as
/// `sorted_runs`: outputs past the abort point are caller-owned garbage.
pub const UNSORTED: i32 = -5;
/// Research-only: unknown `variant` id (`nf_group_variant_*`).
/// Outputs left untouched.
pub const BAD_VARIANT: i32 = -3;
