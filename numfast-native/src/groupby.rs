// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Group-by mechanics: dense scatter, fused SoA scatter, sorted runs.
//! Hash groupby is a reserved facade (surface not started).

pub mod dense;
pub mod fused;
pub mod hash;
pub mod hash_grouped;
pub mod mixed;
pub mod pack_fused;
pub mod scaled;
pub mod sorted;
pub mod variants;
