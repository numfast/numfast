// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Shared core: data-level internals + tiny inline atoms.
//!
//! No hot loops here (except the two 1-param generic traversals in
//! `compact`); only the pieces every kernel repeats: pointer/length
//! handling, error codes, output zeroing, validity, small numerics,
//! and the reserved hash-probe atoms (no consumers yet — STOP: hash
//! surface not started).

pub mod buffers;
pub mod checks;
pub mod compact;
pub mod errors;
pub mod hash;
pub mod numeric;
pub mod validity;
