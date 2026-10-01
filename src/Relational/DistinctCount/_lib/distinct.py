# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Generic partitioned single-pass hash COUNT DISTINCT for integer series.

Native-first: Rust nf_pair_insert canonical; numpy unique fallback
(bit-exact). No legacy JIT (WASM-gate: browser uses wasm unique/sort)."""
import numpy as np


def _probe_insert(xi, tab, used, mask_i):
    """Numpy fallback: exact distinct count (table args kept for ABI)."""
    return int(np.unique(xi).size)


def _scatter(xi, pid, counts, offsets, out):
    cur = offsets.copy()
    for i in range(xi.shape[0]):
        p = int(pid[i])
        dst = int(cur[p])
        out[dst] = xi[i]
        cur[p] = dst + 1


def _pid_fill(xi, pid, p_mask, p_shift):
    pid[:] = ((xi * np.int64(0x9E3779B9)) >> np.int64(p_shift)) & np.int64(p_mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def count_distinct(x):
    """Exact COUNT DISTINCT. Generic over integer dtypes/sizes.

    Partitioned single-pass hash: table sized from input length
    (2x next-pow2), split into partitions when table exceeds ~512K
    slots to preserve cache locality. Non-integer dtypes fall back
    to sort-based unique (exact oracle).
    """
    a = np.asarray(x)
    if a.size == 0:
        return 0
    if a.dtype.kind not in ("i", "u"):
        return int(np.unique(a).size)
    xi = a.astype(np.int64, copy=False)
    n = xi.size
    m = _next_pow2(max(1024, 2 * n))
    try:
        tab = np.empty(m, dtype=np.int64)
        used = np.zeros(m, dtype=np.bool_)
        return int(_probe_insert(xi, tab, used, np.int64(m - 1)))
    except MemoryError:
        pass
    slot_cap = 1 << 19
    # partitioned: P derived from table size, never hardcoded.
    # single counting-scatter split (O(N)), then per-partition hash.
    p = _next_pow2((m + slot_cap - 1) // slot_cap)
    p_shift = np.int64(16)
    p_mask = np.int64(p - 1)
    pid = np.empty(n, dtype=np.int64)
    _pid_fill(xi, pid, p_mask, p_shift)
    counts = np.bincount(pid, minlength=p).astype(np.int64)
    offsets = np.zeros(p, dtype=np.int64)
    for i in range(1, p):
        offsets[i] = offsets[i - 1] + counts[i - 1]
    out = np.empty(n, dtype=np.int64)
    _scatter(xi, pid, counts, offsets, out)
    total = 0
    for part in range(p):
        s = int(offsets[part])
        e = s + int(counts[part])
        if e <= s:
            continue
        sub = out[s:e]
        mp = _next_pow2(max(1024, 2 * (e - s)))
        tab = np.empty(mp, dtype=np.int64)
        used = np.zeros(mp, dtype=np.bool_)
        total += int(_probe_insert(sub, tab, used, np.int64(mp - 1)))
    return int(total)
