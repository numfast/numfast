# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Generic grouped COUNT DISTINCT via partitioned pair open-addressing hash.

The composite key is two independent arrays (keys, values): no factorization
of either domain, no packed sort key, no cardinality assumptions. Rows are
routed by pair fingerprint into P partitions (counting scatter, no sort);
each partition owns a small linear-probing table over (key, value) pairs
with full-key compare on every probe (exact by construction, independent of
fingerprint quality — collisions only affect balance, never correctness).
Occupied keys are collected per partition, then counted by an adaptive
unique: measured key range -> dense bincount, else sort fallback.

Falsifier evidence (ClickBench 10M Q9 RegionID x DISTINCT UserID, seed 42,
chk-exact vs DuckDB full map, own workdir cb_q9res):
- monolithic ST pair hash 538ms (insert ~394 + scan ~53)
- partitioned-sort 2800ms -> falsified (prior batch, not repeated)
- partitioned OA, index-scatter ST 773-992ms -> falsified (random gather)
- partitioned OA, copy-scatter ST 699ms -> falsified vs 538ms
- copy-scatter MT 320ms @16t -> survived (insert 455 -> 75ms)
- + parallel front (fp/count/scatter) 160ms @16t -> survived
- + slot factor 2.0 -> 1.5 (M halved) 124ms -> survived
- + ordered collect + adaptive dense unique 101-102ms exact
- Rust ST kernel 454ms vs legacy-JIT ST 511ms (0.89x, bit-identical tables) ->
  NO-GO as closer: codegen is not the lever (DRAM latency is); MT vehicle
  already exists here. Residual closed without a native toolchain in the
  Windows build path.

Structural constants (NOT data cards): P from N only (~256K rows per
partition keeps per-partition tables LLC-resident); slot factor 1.5;
DENSE_CAP 2**24 bins max (128MB worst-case transient, else sort fallback);
threads from os.cpu_count capped by P. No result cache, no fixed M,
no hardcoded cardinalities, dtypes preserved (no int32 narrowing).

Any doubt (retired lane, non-integer dtypes, allocation failure) ->
_HashMiss; the caller keeps the proven sorted_dedup lane.
This module never raises past _HashMiss.

NATIVE-ALL: the MT JIT vehicle below is retired (target zero). Canonical
lane is the caller-owned sorted_dedup path (C-speed sort + vector scan,
bit-exact). grouped_distinct_hash always raises _HashMiss; the surviving
helpers (partition sizing, buffer pool, adaptive unique) stay for reuse.
"""
import os
import threading

import numpy as np


class _HashMiss(Exception):
    """Fast-lane miss: caller must use the proven sorted_dedup lane."""


GroupedHashMiss = _HashMiss


_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1FF973)

# Worst-case dense-histogram span (bins). Above it the adaptive unique
# takes the sort fallback. Structural cap, not a data cardinality.
_DENSE_CAP = 1 << 24

# Nominal rows per partition. P derives from N only; per-partition tables
# stay LLC-resident while P stays >> thread counts for balance.
_ROWS_PER_PART = 1 << 18

# Table slots per row (load ~0.67 nominal). Halving M vs 2.0 cut insert
# latency (falsifier: 73 -> 51ms) with probe chains still short.
_SLOTF = 1.5

# Generic transient-buffer pool: reuse large first-touched transients
# across calls. Keyed by (role, dtype) with largest-buffer + slice
# semantics so any N / dtype reuses warm pages. Roles are distinct
# live ranges (FP/PK/PV/TK/TV/USED/...) so no aliasing. Buffers that
# the kernels fully overwrite are reused without zeroing; buffers the
# kernels accumulate into (TC, USED) are zeroed on reuse. No size or
# dtype constants: purely mechanical reuse, algorithm untouched.
_POOL = {}
_POOL_LOCK = threading.Lock()


def _pool_key(role, dtype):
    return (role, np.dtype(dtype).str)


def _pool_get(role, n, dtype, zero):
    dt = np.dtype(dtype)
    key = (role, dt.str)
    with _POOL_LOCK:
        buf = _POOL.get(key)
        if buf is not None and buf.size >= n:
            out = buf[:n]
            hit = True
        else:
            out = None
            hit = False
    if hit:
        if zero:
            out.fill(0)
        return out
    try:
        fresh = np.zeros(n, dtype=dt) if zero else np.empty(n, dtype=dt)
    except (MemoryError, ValueError, OverflowError) as e:
        raise _HashMiss(f"pool alloc {role}: {e}")
    with _POOL_LOCK:
        old = _POOL.get(key)
        if old is None or old.size < n:
            _POOL[key] = fresh
            return fresh
        out = old[:n]
        if zero:
            out.fill(0)
        return out

# NATIVE-ALL: MT JIT kernels deleted (target zero). The orchestration they
# implemented (fingerprint -> count -> scatter -> per-partition insert ->
# ordered collect -> adaptive unique) is preserved in sorted_dedup form by
# the caller; _adaptive_unique below stays as the shared tail.


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _partitions_for(n):
    return int(min(256, max(16, _next_pow2(
        max(1, (n + _ROWS_PER_PART - 1) // _ROWS_PER_PART)))))


def _threads_for(p):
    try:
        c = int(os.cpu_count() or 0)
    except (NotImplementedError, TypeError):
        c = 0
    if c <= 0:
        c = 16
    # Falsifier (ClickBench 10M, 20C/40T, seed 42, exact): threads
    # 12 -> 110ms / 16 -> 101ms / 20 -> 107ms / 32 -> degraded.
    # Cap 16: memory-bound kernels oversubscribe past it; fewer than
    # 16 only when the box itself is smaller. Structural, not a card.
    return int(min(max(c, 1), p, 16))


def grouped_distinct_hash(kc, vc):
    """(ukeys int64 sorted, nunique int64, hash_ms, scan_ms).

    NATIVE-ALL: MT lane retired (target zero). Always raises _HashMiss:
    the caller runs the proven sorted_dedup lane verbatim (C-speed sort +
    vector scan, bit-exact).
    """
    raise _HashMiss("mt lane retired NATIVE-ALL: sorted_dedup canonical")


def _adaptive_unique(OK):
    """Sorted (ukeys int64, counts int64). Range-measured dense bincount,
    sort fallback for wide spans or int64-unsafe keys."""
    n = int(OK.size)
    if n == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    try:
        lo = int(OK.min())
        hi = int(OK.max())
    except (ValueError, TypeError) as e:
        raise _HashMiss(f"range: {e}")
    i64lo = -(1 << 63)
    i64hi = (1 << 63) - 1
    if lo >= i64lo and hi <= i64hi and (hi - lo) <= _DENSE_CAP:
        try:
            bc = np.bincount((OK.astype(np.int64) - lo).astype(np.int64),
                             minlength=(hi - lo) + 1)
        except (MemoryError, ValueError, OverflowError) as e:
            raise _HashMiss(f"dense alloc: {e}")
        nz = np.flatnonzero(bc)
        ukeys = np.ascontiguousarray((nz + lo).astype(np.int64))
        nunique = np.ascontiguousarray(bc[nz].astype(np.int64))
        return ukeys, nunique
    uk, cn = np.unique(OK, return_counts=True)
    return (np.ascontiguousarray(uk.astype(np.int64, copy=False)),
            np.ascontiguousarray(cn.astype(np.int64, copy=False)))
