// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//! Sorted-order unique values + inverse codes (Unique M1 research, additive).
//!
//! Replaces `np.unique(keys, return_inverse=True)` for int32/int64 keys
//! with a native O(n) path: proven M1 stable LSD radix ascending
//! permutation (`sort::sort_perm_*`, bit-for-bit `np.argsort(kind="stable")`)
//! + one linear run-encode over the permutation.
//!
//! SEMANTICS (frozen for this research, parity-checked exactly):
//! sorted-order codes, identical labelling to `np.unique`: `uniq` holds
//! the sorted ascending distinct values; `inv[r]` is the sorted position
//! of `keys[r]`; `uniq[inv] == keys` elementwise. NOT first-appearance
//! order: downstream (`groupindex.unique_fallback_index` contract) depends
//! on sorted-position codes feeding order-sensitive aggregation, so any
//! relabelling would silently change results — hence exact parity.
//!
//! `inv` lanes are int32: `ng <= n <= i32::MAX` is enforced at the FFI
//! boundary, so every code fits. `np.unique` returns an int64 (intp)
//! inverse; callers compare after cast (values bit-exact, codes
//! value-exact). Same buffers in/out discipline as the other kernels:
//! caller-owned scratch, no alloc, no collections, `std` only at the
//! FFI wrap. Single-threaded (matches ST `np.unique`).

/// Run-encode over an ascending permutation: `uniq_out[0..ng]` gets the
/// sorted distinct values, `inv_out[row]` the sorted-position code.
/// Caller guarantees `perm` sorts `keys` ascending (`perm` is a
/// permutation of `0..n`), all lanes `len >= n`. Returns `ng`.
#[inline]
pub fn unique_inverse_from_perm_i32(
    keys: &[i32],
    perm: &[i32],
    uniq_out: &mut [i32],
    inv_out: &mut [i32],
) -> usize {
    let n = keys.len();
    debug_assert!(perm.len() >= n);
    debug_assert!(uniq_out.len() >= n);
    debug_assert!(inv_out.len() >= n);
    if n == 0 {
        return 0;
    }
    let mut prev = keys[perm[0] as usize];
    uniq_out[0] = prev;
    inv_out[perm[0] as usize] = 0;
    let mut ng: usize = 1;
    for r in 1..n {
        let row = perm[r] as usize;
        let k = keys[row];
        if k != prev {
            prev = k;
            uniq_out[ng] = k;
            ng += 1;
        }
        inv_out[row] = (ng - 1) as i32;
    }
    ng
}

/// int64 variant of [`unique_inverse_from_perm_i32`]; codes stay int32
/// (`ng <= n <= i32::MAX`), values are int64.
#[inline]
pub fn unique_inverse_from_perm_i64(
    keys: &[i64],
    perm: &[i32],
    uniq_out: &mut [i64],
    inv_out: &mut [i32],
) -> usize {
    let n = keys.len();
    debug_assert!(perm.len() >= n);
    debug_assert!(uniq_out.len() >= n);
    debug_assert!(inv_out.len() >= n);
    if n == 0 {
        return 0;
    }
    let mut prev = keys[perm[0] as usize];
    uniq_out[0] = prev;
    inv_out[perm[0] as usize] = 0;
    let mut ng: usize = 1;
    for r in 1..n {
        let row = perm[r] as usize;
        let k = keys[row];
        if k != prev {
            prev = k;
            uniq_out[ng] = k;
            ng += 1;
        }
        inv_out[row] = (ng - 1) as i32;
    }
    ng
}

// ---------------------------------------------------------------------------
// UTF-8 dictionary dedup: Arrow buffer consumption, no Python str objects.
// ---------------------------------------------------------------------------

use std::collections::HashMap;

/// FNV-1a 32-bit hash — fast, deterministic, no dependencies.
#[inline]
fn fnv1a(data: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// UTF-8 dictionary dedup over Arrow-like buffers.
///
/// `data`: concatenated UTF-8 bytes, `offs`: int32[D+1] offsets,
/// `valid`: bool[N] validity (null pointer = all valid).
///
/// Returns `(ng, uniq_offsets, uniq_data, codes)`:
/// - `ng`: number of unique valid strings
/// - `uniq_offsets`: int32[ng+1] offsets into uniq_data (sorted unique)
/// - `uniq_data`: concatenated sorted unique UTF-8 bytes
/// - `codes`: int32[N] per-row codes (0 for invalid rows)
///
/// Deterministic sorted-unique order (lexicographic by byte).
pub fn unique_dict_utf8(
    data: &[u8],
    offs: &[i32],
    valid: Option<&[u8]>,
    n: usize,
    codes_out: &mut [i32],
    uniq_data_out: &mut [u8],
    uniq_offs_out: &mut [i32],
) -> i32 {
    // Production guards: explicit bounds, no panics (panic=abort stays in
    // Cargo, but valid/boundary inputs must never trap). -1 = malformed.
    if offs.len() != n + 1 {
        return -1;
    }
    if codes_out.len() < n {
        return -1;
    }
    if let Some(v) = valid {
        if v.len() < n {
            return -1;
        }
    }
    if n == 0 {
        return 0;
    }
    if n > i32::MAX as usize {
        return -1;
    }
    if uniq_offs_out.len() < n + 1 {
        return -1;
    }

    // 1. Single pass: extract byte slices, FNV hash -> collision-safe map.
    //    HashMap<u32, Vec<(usize, usize, i32)>>: hash -> list of
    //    (data_offset, byte_len, code) for strings with that hash.
    //    On hash match, scan the small collision list for byte equality.
    //    For typical TEXT columns (low cardinality), lists are 0-1 elements.
    let mut map: HashMap<u32, Vec<(usize, usize, i32)>> = HashMap::new();
    let mut uniq_data_offsets: Vec<(usize, usize)> = Vec::new(); // (start, len) per unique

    for i in 0..n {
        let is_valid = match valid {
            Some(v) => v[i] != 0,
            None => true,
        };
        if !is_valid {
            codes_out[i] = 0;
            continue;
        }
        let a_raw = offs[i];
        let b_raw = offs[i + 1];
        if a_raw < 0 || b_raw < 0 {
            return -1;
        }
        let a = a_raw as usize;
        let b = b_raw as usize;
        if a > b || b > data.len() {
            return -1;
        }
        let slice = &data[a..b];
        let h = fnv1a(slice);

        match map.get_mut(&h) {
            Some(bucket) => {
                // Scan collision list for byte-exact match.
                let mut found = false;
                for &(sa, sl, code) in bucket.iter() {
                    if sl == slice.len() && data[sa..sa + sl] == *slice {
                        codes_out[i] = code;
                        found = true;
                        break;
                    }
                }
                if !found {
                    if uniq_data_offsets.len() >= i32::MAX as usize {
                        return -1;
                    }
                    let code = uniq_data_offsets.len() as i32;
                    uniq_data_offsets.push((a, b - a));
                    bucket.push((a, b - a, code));
                    codes_out[i] = code;
                }
            }
            None => {
                if uniq_data_offsets.len() >= i32::MAX as usize {
                    return -1;
                }
                let code = uniq_data_offsets.len() as i32;
                uniq_data_offsets.push((a, b - a));
                map.insert(h, vec![(a, b - a, code)]);
                codes_out[i] = code;
            }
        }
    }

    let ng = uniq_data_offsets.len();
    if ng == 0 {
        return 0;
    }
    if ng > i32::MAX as usize {
        return -1;
    }

    // 2. Sort unique slices lexicographically (byte order == UTF-8 lex order).
    let mut sorted_idx: Vec<usize> = (0..ng).collect();
    sorted_idx.sort_by(|&a, &b| {
        let (sa, sl) = uniq_data_offsets[a];
        let (sb, sbl) = uniq_data_offsets[b];
        data[sa..sa + sl].cmp(&data[sb..sb + sbl])
    });

    // 3. Build sorted uniq_data + uniq_offs, and code remap.
    let mut remap = vec![0i32; ng];
    let mut body_offs = vec![0i32; ng + 1];
    let mut pos = 0usize;
    for (sorted_rank, &orig_idx) in sorted_idx.iter().enumerate() {
        remap[orig_idx] = sorted_rank as i32;
        let (a, len) = uniq_data_offsets[orig_idx];
        if match pos.checked_add(len) {
            Some(end) => end > uniq_data_out.len(),
            None => true,
        } {
            return -1;
        }
        uniq_data_out[pos..pos + len].copy_from_slice(&data[a..a + len]);
        pos += len;
        if pos > i32::MAX as usize {
            return -1;
        }
        body_offs[sorted_rank + 1] = pos as i32;
    }

    // 4. Copy offsets to output.
    if uniq_offs_out.len() < ng + 1 {
        return -1;
    }
    uniq_offs_out[..ng + 1].copy_from_slice(&body_offs);

    // 5. Remap codes to sorted order; invalid rows stay 0 (null -> 0).
    match valid {
        Some(v) => {
            for (idx, code) in codes_out[..n].iter_mut().enumerate() {
                if v[idx] == 0 {
                    *code = 0;
                } else {
                    let c = *code;
                    if c < 0 || (c as usize) >= ng {
                        return -1;
                    }
                    *code = remap[c as usize];
                }
            }
        }
        None => {
            for code in codes_out[..n].iter_mut() {
                let c = *code;
                if c < 0 || (c as usize) >= ng {
                    return -1;
                }
                *code = remap[c as usize];
            }
        }
    }

    ng as i32
}
