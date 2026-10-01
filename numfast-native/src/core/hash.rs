// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `hash_insert` mechanic atoms (RESERVED).
//!
//! Linear-probe step primitives for the future hash-groupby surface.
//! No consumers yet — STOP: that surface is not started, nothing here
//! is exported over the C ABI. Kept small so the wiring point exists
//! without dead generic machinery.

/// First probe slot: multiplicative mix, power-of-two mask.
#[allow(dead_code)]
#[inline(always)]
pub fn probe_step(hash: u64, mask: usize) -> usize {
    ((hash.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize) & mask
}

/// Next slot: linear probing with mask wrap.
#[allow(dead_code)]
#[inline(always)]
pub fn next_probe(slot: usize, mask: usize) -> usize {
    (slot + 1) & mask
}
