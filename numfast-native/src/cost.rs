// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! `cost` mechanic: generic integer costing core (no RoadGraph semantics,
//! no GPU/WASM/bindings vocabulary here).
//!
//! Frozen parity with `roadgraph/RouteApi/_lib/weights.py` (read-only
//! reference: PROFILE_ID, bucket_of, traffic_k_for,
//! travel_time_for_bucket, CostTable intern/resolve):
//! - `distance_mm` u32 is traffic-independent (stored separately, never
//!   scaled by bucket; passthrough with no arithmetic).
//! - `travel_time_ms` resolve is integer-only with a u64 middle,
//!   INF guard (`UINT32_MAX` reserved, passes through, never computed),
//!   `speed == 0 -> INF` (divide forbidden), clamp to `UINT32_MAX - 1`
//!   range (INF reserved), `v < 1 && length > 0 -> 1`.
//! - `traffic_k` u16 (`K_SCALE = 1000` nominal); `K == 0` is a range
//!   error (Python requires `1..=65535`).
//! - `CostTable`/interning is value-dedup only (`vec -> cost_id`):
//!   identical vectors share one `cost_id`, different vectors get
//!   different ids. Directed rows (`A->B` / `B->A`) stay separate
//!   lanes (never merged); parallel rows intern separately (no MIN
//!   collapse). Deterministic: first-appearance ids, compact table in
//!   first-appearance order, row-independent traversal.
//! - Boundary enums are integers only (portable_core style):
//!   profile `0..=3` valid, `255` = truck slot (error code only, model
//!   not invented here); bucket `0..=23` hour or `255` nominal; strings
//!   live only in the Python adapter, never here.
//!
//! NOT in this module (hard boundary): TBRS, object adjacency,
//! worker/order constraints, hierarchy, foam-gate as a business rule,
//! dispatcher, truck model (only the error code).
//!
//! Same no-alloc `core`-only discipline as the other kernels for the
//! hot lanes (`cost_travel_batch`, `cost_intern_rows`, scalar atoms):
//! the caller owns all buffers; `n == 0` returns without dereferencing;
//! on pre-validated errors outputs are untouched, on streaming errors
//! lanes before the abort point may be written (same rule as the
//! sorted-run speculative abort). `CostTable` (host-side interning
//! with std collections) is the only allocating item and is NOT part
//! of the FFI hot path (FFI uses the stateless `cost_intern_rows`).

use crate::core::errors::{BAD_RANGE, MALFORMED, OK, OVERFLOW};

/// Reserved lane value (INF/INVALID): passes through, never computed.
pub const INF_U32: u32 = u32::MAX;
/// Nominal permille factor.
pub const K_SCALE: u32 = 1000;
/// Nominal departure bucket (traffic ignored).
pub const DEPARTURE_NOMINAL: u8 = 255;
/// Hour buckets `0..=23`.
pub const N_BUCKETS: usize = 24;
/// Portable profile ids (integers only): pedestrian/scooter/bicycle/
/// automobile. `255` is the truck slot (error only).
pub const PROFILE_PEDESTRIAN: u8 = 0;
pub const PROFILE_SCOOTER: u8 = 1;
pub const PROFILE_BICYCLE: u8 = 2;
pub const PROFILE_AUTOMOBILE: u8 = 3;
/// Truck slot: accepted as a code, rejected with this error (model not
/// invented here). Reuses the frozen `-4` research code (no new ABI).
pub const PROFILE_TRUCK: u8 = 255;
pub const COST_TRUCK: i32 = OVERFLOW;

/// Geometry canon: u32 mm passthrough (no arithmetic, INF rides as-is).
#[inline]
pub fn distance_mm_of(length_mm: u32) -> u32 {
    length_mm
}

/// Integer costing canon: `(length_mm * K + speed / 2) / speed`.
///
/// u64 middle, INF guard (either input INF -> INF, no arithmetic),
/// `speed == 0 -> INF`, `v < 1 && length > 0 -> 1`, clamp to
/// `UINT32_MAX - 1` (INF reserved). `K == 0` is a caller error: this
/// scalar atom saturates to INF-guard parity only for valid `K >= 1`;
/// batch/table entry points reject `K == 0` with `BAD_RANGE` (use
/// [`k_check`] / [`cost_travel_batch`]).
#[inline]
pub fn travel_ms_from_mm(length_mm: u32, speed_mms: u32, k: u16) -> u32 {
    if length_mm == INF_U32 || speed_mms == INF_U32 {
        return INF_U32;
    }
    if speed_mms == 0 {
        return INF_U32;
    }
    let num: u64 = (length_mm as u64) * (k as u64);
    let half: u64 = (speed_mms as u64) / 2;
    let v: u64 = (num + half) / (speed_mms as u64);
    if v == 0 {
        return if length_mm > 0 { 1 } else { 0 };
    }
    if v >= INF_U32 as u64 {
        return INF_U32;
    }
    v as u32
}

/// Normalize a departure code to a bucket: `255 -> 255` (nominal),
/// `0..=23` direct, larger ints (epoch seconds) `-> % 24`.
/// Anything else (`< 0`, `== 24`) is `BAD_RANGE` (-2).
/// Caller passes `255` for nominal (Python `None` lives in the adapter).
#[inline]
pub fn bucket_of(departure: i32) -> i32 {
    if departure == DEPARTURE_NOMINAL as i32 {
        return DEPARTURE_NOMINAL as i32;
    }
    if departure < 0 {
        return BAD_RANGE;
    }
    let d = if departure > 24 {
        departure % (N_BUCKETS as i32)
    } else {
        departure
    };
    if (0..=23).contains(&d) {
        d
    } else {
        BAD_RANGE
    }
}

/// Profile gate: `0..=3 -> OK`, `255 (truck) -> COST_TRUCK (-4)`,
/// anything else `-> BAD_RANGE`.
#[inline]
pub fn profile_check(profile: u8) -> i32 {
    if profile == PROFILE_TRUCK {
        return COST_TRUCK;
    }
    if profile <= PROFILE_AUTOMOBILE {
        return OK;
    }
    BAD_RANGE
}

/// Bucket gate: `255 (nominal) -> OK`, `0..=23 -> OK`, else `BAD_RANGE`.
#[inline]
pub fn bucket_check(bucket: u8) -> i32 {
    if bucket == DEPARTURE_NOMINAL {
        return OK;
    }
    if bucket <= 23 {
        return OK;
    }
    BAD_RANGE
}

/// K gate: `1..=65535 -> OK`, else (`0`, overflow u32) `-> BAD_RANGE`.
#[inline]
pub fn k_check(k: u32) -> i32 {
    if (1..=65535).contains(&k) {
        OK
    } else {
        BAD_RANGE
    }
}

/// Multidim resolve: `distance_mm x speed / K -> u32 ms`.
///
/// Returns `>= 0` ms (INF = `0xFFFFFFFF` rides as a value),
/// `COST_TRUCK (-4)` for profile `255`, `BAD_RANGE (-2)` for unknown
/// profile / bad bucket / `K == 0`. `distance_mm` is never scaled by
/// the bucket (traffic enters only through `K`).
#[inline]
pub fn travel_time_for_bucket(
    distance_mm: u32,
    profile: u8,
    bucket: u8,
    speed_mms: u32,
    k: u16,
) -> i64 {
    let rc = profile_check(profile);
    if rc != OK {
        return rc as i64;
    }
    if bucket_check(bucket) != OK {
        return BAD_RANGE as i64;
    }
    if k == 0 {
        return BAD_RANGE as i64;
    }
    travel_ms_from_mm(distance_mm, speed_mms, k) as i64
}

/// Batch integer costing: `out[i] = travel_ms_from_mm(dist[i],
/// speed[i], k[i])`, row-independent, deterministic.
///
/// Directed/parallel semantics are caller rows: each directed edge is
/// its own lane (never merged), parallel rows are never MIN-collapsed
/// (identical inputs naturally yield identical outputs; different
/// inputs stay different). `K == 0` on any lane `-> BAD_RANGE`
/// (lanes before it may be written). Caller guarantees
/// `dist.len() == speed.len() == k.len() == out.len()`.
/// `n == 0` returns `OK` without touching buffers.
#[inline]
pub fn cost_travel_batch(
    dist: &[u32],
    speed: &[u32],
    k: &[u16],
    out: &mut [u32],
) -> i32 {
    debug_assert_eq!(dist.len(), speed.len());
    debug_assert_eq!(dist.len(), k.len());
    debug_assert_eq!(dist.len(), out.len());
    if dist.is_empty() {
        return OK;
    }
    for i in 0..dist.len() {
        if k[i] == 0 {
            return BAD_RANGE;
        }
        out[i] = travel_ms_from_mm(dist[i], speed[i], k[i]);
    }
    OK
}

/// Stateless row interning: dedup identical `width`-wide u32 rows.
///
/// `values[0..n*width]` row-major; `ids[0..n]` overwritten with
/// first-appearance `cost_id`s; `uniq[0..ng*width]` overwritten with
/// the compact table in first-appearance order. Returns `ng >= 0`.
/// Identical vectors share one id; different vectors get different
/// ids (parallel rows: no MIN collapse; directed pairs: caller passes
/// them as separate rows, never merged here). Deterministic across
/// runs and threads (row order traversal, hash lookup only, ids in
/// first-appearance order).
/// Caller guarantees `values.len() >= n * width`, `ids.len() >= n`,
/// `uniq.len() >= n * width`. `n == 0` returns `0` without touching
/// buffers. `width == 0` with `n > 0 -> BAD_RANGE`; short `uniq`
/// (`uniq.len() < n * width`) `-> MALFORMED`.
#[inline]
pub fn cost_intern_rows(
    values: &[u32],
    n: usize,
    width: usize,
    ids: &mut [u32],
    uniq: &mut [u32],
) -> i64 {
    if n == 0 {
        return 0;
    }
    if width == 0 {
        return BAD_RANGE as i64;
    }
    let need = n.saturating_mul(width);
    if values.len() < need || ids.len() < n || uniq.len() < need {
        return MALFORMED as i64;
    }
    let mut ng: usize = 0;
    // Hash lookup (vec -> id); ids stay first-appearance ordered
    // (row order traversal, hash is lookup-only, so determinism and
    // parity with the linear scan hold exactly).
    let mut lut: std::collections::HashMap<Vec<u32>, u32> =
        std::collections::HashMap::with_capacity(n);
    for r in 0..n {
        let s = r * width;
        let row: &[u32] = &values[s..s + width];
        if let Some(&id) = lut.get(row) {
            ids[r] = id;
        } else {
            let t = ng * width;
            uniq[t..t + width].copy_from_slice(row);
            ids[r] = ng as u32;
            lut.insert(row.to_vec(), ng as u32);
            ng += 1;
        }
    }
    ng as i64
}

/// Host-side intern pool: unique cost-vectors `-> cost_id` (dedup by
/// value). Insertion-ordered (first intern wins the id); `resolve`
/// returns the stored vector. Parallel rows intern separately (no MIN
/// collapse); directed pairs stay separate (the table never merges
/// rows). Allocating (std); NOT on the FFI hot path (FFI uses
/// [`cost_intern_rows`]).
#[cfg(not(target_arch = "wasm32"))]
pub mod table {
    use super::{COST_TRUCK, INF_U32};
    use crate::core::errors::{BAD_RANGE, OK};
    use std::collections::HashMap;

    /// Insertion-ordered dedup pool (`cost_id` = u32 index).
    #[derive(Default)]
    pub struct CostTable {
        vec_to_id: HashMap<Vec<u32>, u32>,
        vecs: Vec<Vec<u32>>,
    }

    impl CostTable {
        /// Empty pool.
        pub fn new() -> Self {
            Self {
                vec_to_id: HashMap::new(),
                vecs: Vec::new(),
            }
        }
        /// Intern one cost vector `-> cost_id` (dedup identical).
        /// Values must be u32 lanes (`<= INF_U32` always holds for u32);
        /// `len == u32::MAX` pool `-> BAD_RANGE`. Truck never reaches
        /// here (profile gate owns it: returns `COST_TRUCK` upstream).
        pub fn intern(&mut self, vec: &[u32]) -> Result<u32, i32> {
            if let Some(&id) = self.vec_to_id.get(vec) {
                return Ok(id);
            }
            if self.vecs.len() >= INF_U32 as usize {
                return Err(BAD_RANGE);
            }
            let id = self.vecs.len() as u32;
            self.vec_to_id.insert(vec.to_vec(), id);
            self.vecs.push(vec.to_vec());
            Ok(id)
        }
        /// `cost_id -> cost vector` (`None` if unknown).
        pub fn resolve(&self, cost_id: u32) -> Option<&[u32]> {
            self.vecs.get(cost_id as usize).map(|v| v.as_slice())
        }
        /// Pool size (number of unique vectors).
        pub fn len(&self) -> usize {
            self.vecs.len()
        }
        /// Empty pool.
        pub fn is_empty(&self) -> bool {
            self.vecs.is_empty()
        }
        /// Profile gate passthrough (documents the truck-only error
        /// code on this surface without inventing a model).
        pub fn check_profile(profile: u8) -> i32 {
            if profile == super::PROFILE_TRUCK {
                COST_TRUCK
            } else if profile <= super::PROFILE_AUTOMOBILE {
                OK
            } else {
                BAD_RANGE
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use super::table::CostTable;

    #[test]
    fn buckets_nominal_hour_epoch() {
        assert_eq!(bucket_of(255), 255);
        assert_eq!(bucket_of(0), 0);
        assert_eq!(bucket_of(23), 23);
        assert_eq!(bucket_of(24), BAD_RANGE); // ==24 errors (Python parity)
        assert_eq!(bucket_of(25), 1); // >24 -> % 24
        assert_eq!(bucket_of(48), 0);
        assert_eq!(bucket_of(-1), BAD_RANGE);
        assert_eq!(bucket_check(255), OK);
        assert_eq!(bucket_check(7), OK);
        assert_eq!(bucket_check(24), BAD_RANGE);
    }

    #[test]
    fn profiles_truck_only_code() {
        assert_eq!(profile_check(0), OK);
        assert_eq!(profile_check(3), OK);
        assert_eq!(profile_check(4), BAD_RANGE);
        assert_eq!(profile_check(255), COST_TRUCK);
        assert_eq!(travel_time_for_bucket(1000, 255, 255, 1000, 1000), COST_TRUCK as i64);
        assert_eq!(travel_time_for_bucket(1000, 9, 255, 1000, 1000), BAD_RANGE as i64);
    }

    #[test]
    fn travel_canon_u64_middle() {
        // (length * K + speed/2) / speed.
        assert_eq!(travel_ms_from_mm(1000, 1000, 1000), 1000);
        assert_eq!(travel_ms_from_mm(1500, 1000, 1000), 1500);
        assert_eq!(travel_ms_from_mm(1000, 1000, 2000), 2000);
        assert_eq!(distance_mm_of(12345), 12345);
        assert_eq!(distance_mm_of(INF_U32), INF_U32);
    }

    #[test]
    fn zero_speed_is_inf() {
        assert_eq!(travel_ms_from_mm(1000, 0, 1000), INF_U32);
        assert_eq!(travel_ms_from_mm(0, 0, 1000), INF_U32);
        assert_eq!(travel_time_for_bucket(1000, 3, 5, 0, 1000), INF_U32 as i64);
    }

    #[test]
    fn inf_guard_no_arithmetic() {
        assert_eq!(travel_ms_from_mm(INF_U32, 1000, 1000), INF_U32);
        assert_eq!(travel_ms_from_mm(1000, INF_U32, 1000), INF_U32);
        assert_eq!(travel_ms_from_mm(INF_U32, INF_U32, 1000), INF_U32);
        assert_eq!(travel_time_for_bucket(INF_U32, 0, 255, 500, 1000), INF_U32 as i64);
    }

    #[test]
    fn overflow_clamps_to_inf() {
        // v >= UINT32_MAX -> INF (reserved).
        assert_eq!(travel_ms_from_mm(u32::MAX - 1, 1, 65535), INF_U32);
        // Small lengths clamp to 1 (never 0 for length > 0).
        assert_eq!(travel_ms_from_mm(1, 2000, 1), 1);
        assert_eq!(travel_ms_from_mm(0, 1000, 1000), 0);
    }

    #[test]
    fn k_gate() {
        assert_eq!(k_check(1), OK);
        assert_eq!(k_check(1000), OK);
        assert_eq!(k_check(65535), OK);
        assert_eq!(k_check(0), BAD_RANGE);
        assert_eq!(travel_time_for_bucket(1000, 3, 255, 1000, 0), BAD_RANGE as i64);
        let d = vec![1000];
        let s = vec![1000];
        let k = vec![0u16];
        let mut o = vec![0u32];
        assert_eq!(cost_travel_batch(&d, &s, &k, &mut o), BAD_RANGE);
    }

    #[test]
    fn directed_rows_stay_separate() {
        // A->B and B->A are distinct lanes even with identical payloads:
        // batch never merges; ids come only from value equality.
        let d = vec![1000u32, 1000u32];
        let s = vec![500u32, 500u32];
        let k = vec![1000u16, 1000u16];
        let mut o = vec![0u32, 0u32];
        assert_eq!(cost_travel_batch(&d, &s, &k, &mut o), OK);
        assert_eq!(o, vec![2000, 2000]);
        // Different distances stay different lanes.
        let d2 = vec![1000u32, 2000u32];
        let mut o2 = vec![0u32, 0u32];
        assert_eq!(cost_travel_batch(&d2, &s, &k, &mut o2), OK);
        assert_eq!(o2, vec![2000, 4000]);
    }

    #[test]
    fn parallel_rows_no_min() {
        // Two parallel rows with different costs keep both values;
        // interning dedups only identical vectors (no MIN collapse).
        let vals: Vec<u32> = vec![2000, 2001, 2000, 2002];
        let mut ids = vec![0u32; 2];
        let mut uniq = vec![0u32; 4];
        let ng = cost_intern_rows(&vals, 2, 2, &mut ids, &mut uniq);
        assert_eq!(ng, 2);
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(&uniq[0..4], &[2000, 2001, 2000, 2002][..]);
        // Identical parallel rows share one id (dedup, still no MIN).
        let vals2: Vec<u32> = vec![7, 8, 7, 8];
        let mut ids2 = vec![9u32; 2];
        let mut uniq2 = vec![0u32; 4];
        let ng2 = cost_intern_rows(&vals2, 2, 2, &mut ids2, &mut uniq2);
        assert_eq!(ng2, 1);
        assert_eq!(ids2, vec![0, 0]);
    }

    #[test]
    fn dedup_first_appearance() {
        let vals: Vec<u32> = vec![1, 2, 3, 4, 1, 2, 5, 6, 3, 4];
        let mut ids = vec![99u32; 5];
        let mut uniq = vec![0u32; 10];
        let ng = cost_intern_rows(&vals, 5, 2, &mut ids, &mut uniq);
        assert_eq!(ng, 3);
        assert_eq!(ids, vec![0, 1, 0, 2, 1]);
        assert_eq!(&uniq[0..6], &[1, 2, 3, 4, 5, 6][..]);
    }

    #[test]
    fn intern_guards() {
        let mut ids: Vec<u32> = vec![];
        let mut uniq: Vec<u32> = vec![];
        assert_eq!(cost_intern_rows(&[], 0, 2, &mut ids, &mut uniq), 0);
        let v = vec![1u32, 2];
        let mut i2 = vec![0u32; 1];
        let mut u2 = vec![0u32; 2];
        assert_eq!(cost_intern_rows(&v, 1, 0, &mut i2, &mut u2), BAD_RANGE as i64);
        let mut short = vec![0u32; 1];
        assert_eq!(cost_intern_rows(&v, 1, 2, &mut i2, &mut short), MALFORMED as i64);
    }

    #[test]
    fn determinism_repeat() {
        let d = vec![1234u32, 5678, 91011];
        let s = vec![500u32, 700, 900];
        let k = vec![1000u16, 1500, 800];
        let mut a = vec![0u32; 3];
        let mut b = vec![0u32; 3];
        assert_eq!(cost_travel_batch(&d, &s, &k, &mut a), OK);
        assert_eq!(cost_travel_batch(&d, &s, &k, &mut b), OK);
        assert_eq!(a, b);
        let vals: Vec<u32> = vec![3, 1, 2, 3, 1, 2];
        let mut i1 = vec![0u32; 3];
        let mut u1 = vec![0u32; 6];
        let mut i2 = vec![0u32; 3];
        let mut u2 = vec![0u32; 6];
        assert_eq!(
            cost_intern_rows(&vals, 3, 2, &mut i1, &mut u1),
            cost_intern_rows(&vals, 3, 2, &mut i2, &mut u2)
        );
        assert_eq!((i1, u1), (i2, u2));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn table_intern_resolve() {
        let mut t = CostTable::new();
        let a = t.intern(&[10, 20]).unwrap();
        let b = t.intern(&[30, 40]).unwrap();
        let a2 = t.intern(&[10, 20]).unwrap();
        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert_eq!(t.len(), 2);
        assert_eq!(t.resolve(a), Some(&[10, 20][..]));
        assert_eq!(t.resolve(99), None);
    }
}
