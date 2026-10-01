// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Hash groupby wiring point (RESERVED).
//!
//! Re-exports the [`crate::core::hash`] probe atoms for the future
//! hash-groupby surface. No kernels, no C ABI symbols — STOP: that
//! surface is not started.

pub use crate::core::hash::{next_probe, probe_step};
