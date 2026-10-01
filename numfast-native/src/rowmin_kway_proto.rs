// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Deprecated thin shim over the canonical K-way kernel.
//! Promoted to `rowmin_kway`; this module is kept so existing
//! `rowmin_kway_proto::rowwise_kway_time_argmin_gather` paths keep
//! resolving to the identical canonical implementation.

pub use crate::rowmin_kway::rowwise_kway_time_argmin_gather;
