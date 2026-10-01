// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Series mechanics, decomposed: `scan` (ordered key traversal atoms)
//! + `reduce` (lane accumulation atoms) + `select` (boolean selection and
//! mask combine for the `filter`/`mask` IR nodes).
//!
//! Deliberately NOT one parameterized mega-kernel: float bit-exactness
//! (accumulation order) and the abort/overflow contracts must stay
//! visible in each flat, specialized hot loop. These atoms inline to
//! zero-cost code at every use site.

pub mod cumsum;
pub mod map;
pub mod reduce;
pub mod scan;
pub mod select;
pub mod shift;
