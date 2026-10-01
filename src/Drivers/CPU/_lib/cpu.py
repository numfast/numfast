# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU driver: numpy-oracle, exact on small N (spec 06).

Hot paths (H2O 10M gate) are fully vectorized — no Python per-row loops:
- encode_pattern: Arrow bulk (starts_with/slice/ascii-predicates/cast);
  no-pyarrow fallback is pure-NumPy vectorized (codepoint matrix + Horner,
  same contract, no per-row Python loop at any N);
  strict digit-body validation; int32 overflow raises, never wraps; nullable.
- groupby/groupby_multi: Planner-chosen group_index strategy (sorted-run via
  vectorized probe, memory-gated dense direct, unique fallback), one grouping
  traversal, fused sum+count state.
- group_count_distinct: CPU-op (no IR): valid-mask compact + _sort_perm over
  (keys, values) + one vector scan (group bounds + distinct transitions);
  {key: nunique} dict, keys sorted, counts int64, #strategy="sorted_dedup".
- pack_keys: pack (bit), radix (mixed-radix), hash (tuple-hash reference).
Zero-copy: series/pack reuse buffers (asarray/astype(copy=False)); no
tolist/list()/object-array copies on hot paths.
Dtype mapping derived from Core.canonical_dtype (single source); errors use
Core.format_error contract (injected via kernel.alias, no private imports).
"""

import numpy as np

import os
import time
from concurrent.futures import ThreadPoolExecutor

from _lib.carry import ColumnCarry as _ColumnCarry
from _lib.bitpack import BitPack as _BitPack
from _lib.bitpack import check_width as _check_width
from _lib.bitpack import combine as _bit_combine
from _lib.bitpack import to_bool_array as _to_bool_array
from _lib.groupindex import bycode_estimate_bytes as _bycode_est
from _lib.groupindex import composite_tuple_index as _composite_tuple_index
from _lib.groupindex import dense_estimate_bytes as _dense_est
from _lib.groupindex import fused_dense_aggregate as _fused_dense_aggregate
from _lib.groupindex import group_index as _generic_group_index
from _lib.groupindex import range_probe as _range_probe
from _lib.groupindex import routed_aggregate as _routed_aggregate
from _lib.groupindex import sorted_fused_aggregate as _sorted_fused_aggregate
from _lib.native_cpu import available as _native_available
from _lib.native_cpu import carry_build as _native_carry
from _lib.native_cpu import cumsum_scatter as _native_cumsum
from _lib.native_cpu import fused_available as _fused_available
from _lib.native_cpu import fused_sum_count as _native_fused1
from _lib.native_cpu import i32_gate_ok as _i32_gate_ok
from _lib.native_cpu import map_scatter as _native_map
from _lib.native_cpu import mask_and as _native_mask_and
from _lib.native_cpu import mask_not as _native_mask_not
from _lib.native_cpu import mask_or as _native_mask_or
from _lib.native_cpu import mixed_sum_count as _native_mixed
from _lib.native_cpu import mixed_sum_only_count as _native_sumonly
from _lib.native_cpu import multi_sum_count as _native_fusedN
from _lib.native_cpu import owner_available as _owner_available
from _lib.native_cpu import owner_shard_into as _owner_shard
from _lib.native_cpu import pack_i32_direct as _native_pack
from _lib.native_cpu import pack_sum_count_f64 as _fused_f64
from _lib.native_cpu import pack_sum_count_i32 as _fused_i32
from _lib.native_cpu import resolve_on_worse as _resolve_on_worse
from _lib.native_cpu import rng_available as _rng_native_available
from _lib.native_cpu import rng_compat_runif_native as _rng_compat_runif_n
from _lib.native_cpu import rng_compat_sample_native as _rng_compat_sample_n
from _lib.native_cpu import rng_fill_f64_native as _rng_fill_f64_n
from _lib.native_cpu import rng_fill_i32_native as _rng_fill_i32_n
from _lib.native_cpu import rng_map_round_native as _rng_map_round_n
from _lib.native_cpu import rng_permutation_native as _rng_permutation_n
from _lib.native_cpu import rng_sample_native as _rng_sample_n
from _lib.native_cpu import select_available as _select_available
from _lib.native_cpu import select_scatter as _native_select
from _lib.native_cpu import sort_perm_step as _native_sort_step
from _lib.native_cpu import text_available as _text_available
from _lib.native_cpu import build_text_buffers as _native_build_text
from _lib.native_cpu import text_contains_buffers as _native_text_hit
from _lib.native_cpu import text_endswith_buffers as _native_text_tail
from _lib.native_cpu import text_equals_buffers as _native_text_eq
from _lib.native_cpu import text_length_buffers as _native_text_len
from _lib.native_cpu import text_startswith_buffers as _native_text_head
from _lib.native_cpu import unique_inverse as _native_unique_inv
from _lib.rng import compat_runif as _rng_compat_runif_fb
from _lib.rng import compat_sample as _rng_compat_sample_fb
from _lib.rng import fill_f64 as _rng_fill_f64_fb
from _lib.rng import fill_i32 as _rng_fill_i32_fb
from _lib.rng import map_round as _rng_map_round_fb
from _lib.rng import permutation as _rng_permutation_fb
from _lib.rng import sample_no_replace as _rng_sample_fb

# Host-level MT (ST Rust primitives stay single+reentrant; threads only here).
# NUMFAST_THREADS=1 default: current ST semantics verbatim. Ladder 1/2/4/8/12/16.
_MT_MAX = 64
_POOLS = {}

# P2 (decoupled sum-only) dispatch: total dense state (counts + one sums
# lane per value column, 8B each) above fast-cache scale. Below it the
# counts RMW is L2-resident (~free) while P2 still pays its fixed
# streaming costs (SoA assembly + keys-only count pass) — measured:
# M=101 multi-lane regresses 2x, M=1M wins 14-21% (100M/16T exact,
# tests/heavy/bench_p12_100M.py). 1MB = L2 scale, generic over queries.
_P2_STATE_MIN = 1 << 20

# M7b native-lane gates. Measured on this host (see
# develop/audit_foundation/M7b_cpu_call_sites.md); every threshold is a
# crossover of the SAME work, not a behaviour fork -- below it the proven
# NumPy reference stays the owner, above it the fused DLL lane wins. Both
# sides are bit-identical, so a gate changes speed only, never a value.
#   cumsum float32/float64  n >= 8192. The int32 chain it replaces is 5
#       passes + 3 int64 temporaries, so it repays the ~9 us wrapper tax at
#       any n (measured 1.31x at n=1, ungated). The float chain is a single
#       np.cumsum, which does NOT repay it below 8192 (measured 0.60-0.87x,
#       absolute penalty <= 9 us) and does above (1.21-1.32x at n=1e4,
#       2.30-2.65x at n=1e5).
#   unique_inverse  n >= 1e5. The 4-pass LSD radix has a fixed ~24 us floor
#       that np.unique beats while cardinality is small; at n >= 1e5 the
#       native lane is neutral-to-faster at EVERY cardinality measured
#       (1.08-2.54x, one 0.94x cell at n=2e5 card=2). The n=1e4..3e4 band is a
#       win for cardinality >= 64 (1.37-1.89x) and a loss for cardinality <= 8
#       (0.62-0.82x), and the driver has no cheap cardinality signal to split
#       it -- see the RISKS section of the audit.
#   map div         n >= 1e5 (end-to-end wrapper crossover; 1.9-2.2x at
#       n=3e5..1e6). Neutral for int32 at exactly n=1e5 (1.01x), 1.36-1.76x
#       for float32/float64 there.
_CUMSUM_FLOAT_MIN = 8192
_UNIQUE_NATIVE_MIN = 100_000
_MAP_DIV_NATIVE_MIN = 100_000


def _native_cumsum_lane(a):
    """M7b: fused inclusive scan (nf_cumsum_i32/f32/f64); None -> NumPy owns it.

    Same contract as the NumPy lanes of `_cumsum_ref`: int32 wraps mod 2**32
    and float32/float64 accumulate sequentially in their own dtype, so both
    lanes are bit-identical (scan, not reduce -- no pairwise reassociation).
    Never raises: backend absent / unsupported dtype -> NumPy reference.
    int32 is ungated (its NumPy chain pays for the wrapper); float is gated at
    `_CUMSUM_FLOAT_MIN` because its NumPy chain is one pass.
    """
    if a.dtype != np.dtype(np.int32) and a.size < _CUMSUM_FLOAT_MIN:
        return None
    try:
        return _native_cumsum(a)
    except Exception:  # noqa: BLE001 -- fallback owns it
        return None


def _native_unique_lane(keys):
    """M7b: fused radix unique+inverse (nf_unique_inverse_i32/i64) or None.

    Same contract as `np.unique(keys, return_inverse=True)`: uniq sorted
    ascending, `uniq[inv] == keys`, inv returned in the caller's int32 shape.
    Only entered at `keys.size >= _UNIQUE_NATIVE_MIN`; below that the 4-pass
    LSD radix loses to np.unique at small cardinality (measured 0.62-0.82x).
    None -> NumPy owns it (symbol absent or call failed).
    """
    if keys.size < _UNIQUE_NATIVE_MIN:
        return None
    try:
        uk, inv, backend = _native_unique_inv(keys)
        if backend != "native":
            return None
        return uk, inv.astype(np.int32)
    except Exception:  # noqa: BLE001 -- fallback owns it
        return None


def _native_map_div_lane(a, fn, b):
    """M7b: fused elementwise map (nf_map_i32/f32/f64) for div/pow; None -> NumPy.

    Only `div` reaches this lane: `pow` with an array exponent is rejected by
    the driver before this branch, and the scalar `pow` lane measured
    0.12-1.06x. `add`/`sub`/`mul`/`mod` measured 0.88-1.18x (noise) and stay
    on NumPy too. Collapses the 4-pass astype(f64)->div->rint->astype round
    trip into one pass; EXACT parity with the NumPy lane, including the
    rint-then-cast rule for integer division. None -> NumPy owns it.
    """
    try:
        return _native_map(a, fn, b)
    except Exception:  # noqa: BLE001 -- fallback owns it
        return None


def _p2_worth_it(ncols, m):
    """P2 dispatch on generic observables (lane count + dense domain)."""
    return ncols > 1 and (ncols + 1) * int(m) * 8 > _P2_STATE_MIN


# Q5-P1 (accepted): decouple MT private states from worker threads.
# Total private dense state (S states x (ncols+1) lanes x M x 8B) above
# L3 scale makes merge + cross-core traffic dominate: measured 100M/16T
# exact, Q5 shape (3 lanes, M=1M): S=16 -> 2307ms, S=12 -> 2132ms,
# S=10 -> 2095ms, S=8 -> 2042ms (-11.5%), S=6 -> 2084ms, S=4 wash,
# S<=2 worse (streaming cores starve), S=1 ST 1.8x worse (P2 control).
# Cap total private state at _MT_STATE_BUDGET (generic over lane counts
# + M, never queries): small-M shapes keep S=T (bit-identical slices),
# large-M shapes run fewer, larger slices on the same pool.
_MT_STATE_BUDGET = 256 << 20


def _mt_states(ncols, m, t, n):
    """Private-state count for _native_mt (see _MT_STATE_BUDGET)."""
    per = (int(ncols) + 1) * int(m) * 8
    s = int(t) if per <= 0 else max(1, min(int(t), _MT_STATE_BUDGET // per))
    return max(1, min(s, int(n)))


# Q5-P3 (accepted): owner-group shards for (2xi32 + 1xf) over large dense
# state. Measured 100M/16T exact, Q5 shape (M=1M): owner8 -> 1639ms
# (-30.7% vs 2367.7 baseline, beats 1901 target), owner4 -> 2137ms,
# owner2 -> 3427ms (replication + mispredicts dominate). 8 owners on a
# 16-lane pool: each scans all rows, writes disjoint group ranges of
# shared state (no privates, no merge, no atomics, no partition pass).
_OWNER_SHARDS = 8


def _owner_worth_it(cols, m):
    """Owner-group dispatch on generic observables (lane signature + M-class).

    Exactly 2 int32 + 1 float lanes over large dense state (same L2-scale
    split as _p2_worth_it): owner threads own disjoint group ranges.
    Small-M shapes keep S-states (replication pointless); other lane
    mixes keep their proven paths. Never query names, never cardinalities.
    """
    if len(cols) != 3 or int(m) <= 0:
        return False
    if (np.asarray(cols[0]).dtype != np.dtype(np.int32)
            or np.asarray(cols[1]).dtype != np.dtype(np.int32)
            or np.asarray(cols[2]).dtype.kind != "f"):
        return False
    return (len(cols) + 1) * int(m) * 8 > _P2_STATE_MIN


def _owner_mt(keys, cols, m, t, _bd):
    """Owner-group MT (Q5-P3): disjoint group ranges of shared dense state.

    S_owner threads (see _OWNER_SHARDS) each scan all rows (zero-copy
    views) and accumulate only owned groups: no private states, no merge,
    no atomics, no partition/materialization pass. Shared outputs are
    np.empty (each owner zeroes its own lanes exactly once in-kernel).
    i32 lanes hold GLOBAL totals: the shared bounds probe gates them
    (no rescan); overflow/range/backend failure -> None (S-states path
    owns the query, never wraps). Returns same shape as _native_mt with
    merge_ms=0.0. _bd reuses one concurrent bounds pass (same guard).
    """
    try:
        if not _owner_available():
            return None
        k = np.ascontiguousarray(keys, dtype=np.int32)
        a = np.ascontiguousarray(cols[0], dtype=np.int32)
        b = np.ascontiguousarray(cols[1], dtype=np.int32)
        c = np.ascontiguousarray(cols[2], dtype=np.float64)
        n = int(k.size)
        m = int(m)
        if n == 0 or m <= 0:
            return None
        _b0 = _bd[0] if _bd is not None and len(_bd) > 0 else None
        _b1 = _bd[1] if _bd is not None and len(_bd) > 1 else None
        if not _i32_gate_ok(a, n, _b=_b0) or not _i32_gate_ok(b, n, _b=_b1):
            return None
        so = max(1, min(int(t), _OWNER_SHARDS, n))
        s1 = np.empty(m, dtype=np.int32)
        s2 = np.empty(m, dtype=np.int32)
        s3 = np.empty(m, dtype=np.float64)
        cc = np.empty(m, dtype=np.int64)
        gb = np.linspace(0, m, so + 1).astype(np.int64)
        ex = _pool(t)
        t0 = time.perf_counter()
        futs = [ex.submit(_owner_shard, k, a, b, c,
                          int(gb[w]), int(gb[w + 1]), m,
                          s1, s2, s3, cc) for w in range(so)]
        oks = [f.result() for f in futs]
        agg_ms = (time.perf_counter() - t0) * 1000
        if not all(oks):
            return None
    except Exception:  # noqa: BLE001 -- S-states path owns it
        return None
    try:
        ukeys, counts, out, cms = None, None, [], 0.0
        for s_m in (s1, s2, s3):
            uk, c, s, cm = _native_compact(cc, s_m)
            if ukeys is None:
                ukeys, counts = uk, c
            cms += cm
            out.append(s)
        return ukeys, counts, out, agg_ms, cms, 0.0
    except Exception:  # noqa: BLE001
        return None


def _mt_threads():
    try:
        t = int(os.environ.get("NUMFAST_THREADS", "1"))
    except (TypeError, ValueError):
        return 1
    return max(1, min(_MT_MAX, t))


def _pool(t):
    p = _POOLS.get(t)
    if p is None:
        p = ThreadPoolExecutor(max_workers=t)
        _POOLS[t] = p
    return p


_SENT = object()


def _mm_one(a):
    a = np.asarray(a)
    return int(a.min()), int(a.max())


def _probe_all(keys, vcols, t):
    """One concurrent bounds pass for the native path (MT win; ST identical).

    min/max per array are independent numpy reductions (GIL released on mass
    data); results bit-identical to the sequential scans. Returns
    (kbounds, vbounds): kbounds=(kmin,kmax) or None (probe failed -> proven
    chain, same as the old _dense_gate exception path); vbounds[i]=(mn,mx),
    None (non-int/empty col, skipped like before) or the whole list is None
    (a value probe failed -> int-exact guard fails, same as before)."""
    kas = np.asarray(keys)
    vas = [np.asarray(v) for v in vcols]
    need = [kas] + [v for v in vas
                    if v.dtype.kind in "iu" and v.size > 0]
    try:
        if len(need) > 1 and int(t) > 1:
            got = list(_pool(int(t)).map(_mm_one, need))
        else:
            got = [_mm_one(a) for a in need]
    except Exception:  # noqa: BLE001 -- same fallback as proven scans
        return None, None
    kb = got[0]
    vb, it = [], iter(got[1:])
    for v in vas:
        vb.append(next(it) if v.dtype.kind in "iu" and v.size > 0 else None)
    return kb, vb


def _exact_from(vcols, vb):
    """int-exact predicate on precomputed bounds (same math as _int_exact_ok)."""
    if vb is None:
        return False
    for v, b in zip(vcols, vb):
        v = np.asarray(v)
        if v.dtype.kind in "iu":
            if v.size == 0 or b is None:
                continue
            mn, mx = b
            if int(v.size) * max(abs(mn), abs(mx)) >= 2 ** 53:
                return False
    return True


def _intlane_ok(vcols, _vb=_SENT):
    """int-lane gate: every int col servable exactly without f64.

    int32 passing the i32 gate -> int32-direct lane; any other int with
    n*maxabs < 2**63-1 -> int64-direct lane (checked add: OverflowError
    falls back, never wraps). Bounds reuse the shared concurrent probe
    (_vb, same numbers); missing entries probe C-speed here (rare).
    Beyond int64 range -> False (proven chain owns it, as before).
    """
    for i, v in enumerate(vcols):
        v = np.asarray(v)
        if v.dtype.kind not in "iu" or v.size == 0:
            continue
        b = None
        if _vb is not _SENT and _vb is not None and i < len(_vb):
            b = _vb[i]
        if b is None:
            try:
                mn, mx = int(v.min()), int(v.max())
            except (ValueError, OverflowError):
                return False
        else:
            mn, mx = b
        if int(v.size) * max(abs(mn), abs(mx)) >= 2 ** 63 - 1:
            return False
    return True


def _dense_gate(keys, plan_groupby, naggs=1, _kb=_SENT):
    """Planner-owned dense_by_code gate (probe here, decide in Planner).

    Returns (dec, m_code, probe_ms) or (None, None, ms) when the gate does
    not apply (no Planner / Planner rejects dense). Never raises: any probe
    failure means 'fall through to the proven chain'.
    _kb=(kmin,kmax) reuses one concurrent probe pass (same numbers).
    """
    t0 = time.perf_counter()
    try:
        n = int(np.asarray(keys).size)
        if _kb is _SENT:
            kmin, kmax, r = _range_probe(keys)
        elif _kb is None:
            raise ValueError("shared probe failed")
        else:
            kmin, kmax = _kb
            r = kmax - kmin + 1
        est = _dense_est(r, n)
        m_code = int(kmax) + 1 if kmin >= 0 else None
        ki64 = np.asarray(keys).dtype == np.int64
        est_code = (_bycode_est(m_code, n, ki64)
                    if m_code is not None else None)
        try:
            dec = plan_groupby(n, False, span=r, est_dense_bytes=est,
                               budget_bytes=None, kmin=kmin,
                               est_code_bytes=est_code, naggs=naggs)
        except TypeError:
            try:
                dec = plan_groupby(n, False, span=r, est_dense_bytes=est,
                                   budget_bytes=None)
            except TypeError:
                dec = plan_groupby(n, False)
    except Exception:  # noqa: BLE001 -- gate failure = proven chain
        return None, None, (time.perf_counter() - t0) * 1000
    ms = (time.perf_counter() - t0) * 1000
    if not isinstance(dec, dict) or dec.get("strategy") != "dense_by_code":
        return None, None, ms
    if m_code is None:
        return None, None, ms
    return dec, int(m_code), ms


def _int_exact_ok(vcols, _vb=_SENT):
    """int cols via native f64 path stay exact iff totals < 2**53.

    All partial sums are then representable: each sequential f64 addition is
    exact. Beyond that the proven int64 bincount path owns the query.
    One abs/max scan per int col (C-speed, outside agg timing).
    _vb reuses one concurrent bounds pass (same predicate)."""
    if _vb is not _SENT:
        return _exact_from(vcols, _vb)
    for v in vcols:
        v = np.asarray(v)
        if v.dtype.kind in "iu":
            if v.size == 0:
                continue
            try:
                # No widening copy: min/max on the narrow dtype (C-speed),
                # abs() on Python ints (no INT_MIN wrap).
                mn, mx = int(v.min()), int(v.max())
            except (ValueError, OverflowError):
                return False
            if int(v.size) * max(abs(mn), abs(mx)) >= 2 ** 53:
                return False
    return True


def _native_compact(counts_m, sums_m):
    """ColumnCarry compact (native aggregate -> ColumnCarry -> explicit dict).

    counts_m/sums_m are dense M states; returns (ukeys, counts, sums) compact
    and carry_ms. _native_carry falls back to numpy when the DLL is absent,
    so semantics never depend on backend presence."""
    t0 = time.perf_counter()
    uk, c, s = _native_carry(np.ascontiguousarray(counts_m),
                             np.ascontiguousarray(sums_m))
    return uk, c, s, (time.perf_counter() - t0) * 1000


def _native_st(keys, vcols, m, _vb=_SENT):
    """Single-thread native dense aggregate + per-column carry.

    keys: int32 resident codes 0<=k<m. Returns
    (ukeys, counts, sums_list, agg_ms, carry_ms) or None (fallback owns it).
    _vb reuses one concurrent bounds pass (same guard)."""
    try:
        keys = np.ascontiguousarray(keys, dtype=np.int32)
        cols = [np.ascontiguousarray(v) for v in vcols]
        _bd = None if _vb is _SENT else _vb
        if any(c.dtype.kind in "iu" for c in cols):
            # Int fast lane: zero-copy int kernels when gated; any lane
            # failure (gate/overflow/backend) -> legacy f64 path verbatim.
            t0 = time.perf_counter()
            _mx = None
            if _p2_worth_it(len(cols), m):
                # P2: decoupled counts + single sum-only pass over all
                # lanes (measured win on large-state multi-lane at 16T).
                # Small-state / single-col keeps the proven paths below.
                _mx = _native_sumonly(keys, cols, m, _bounds=_bd)
            if _mx is None:
                _mx = _native_mixed(keys, cols, m, _bounds=_bd)
            if _mx is not None:
                sums_list, counts_m = _mx
                agg_ms = (time.perf_counter() - t0) * 1000
                try:
                    ukeys, counts, out, cms = None, None, [], 0.0
                    for s_m in sums_list:
                        uk, c, s, cm = _native_compact(counts_m, s_m)
                        if ukeys is None:
                            ukeys, counts = uk, c
                        cms += cm
                        out.append(s)
                    return ukeys, counts, out, agg_ms, cms
                except Exception:  # noqa: BLE001
                    return None
        if not _int_exact_ok(cols, _vb):
            return None
        t0 = time.perf_counter()
        if len(cols) == 1:
            sums_dense, counts_m = _native_fused1(keys, cols[0], m)
            sums_list = [sums_dense]
        else:
            sums_list, counts_m = _native_fusedN(keys, cols, m)
        agg_ms = (time.perf_counter() - t0) * 1000
    except Exception:  # noqa: BLE001 -- fallback must never raise
        return None
    try:
        ukeys, counts, out, cms = None, None, [], 0.0
        for s_m in sums_list:
            uk, c, s, cm = _native_compact(counts_m, s_m)
            if ukeys is None:
                ukeys, counts = uk, c
            cms += cm
            out.append(s)
        return ukeys, counts, out, agg_ms, cms
    except Exception:  # noqa: BLE001
        return None


def _native_mt(keys, vcols, m, t, _vb=_SENT):
    """Host MT: contiguous row partition, worker-private native states.

    Each worker runs the ST native primitive on its slice (Rust never sees
    threads); merge is a deterministic sequential sum w=0..s-1, then one
    ColumnCarry. Private-state count S comes from _mt_states (decoupled
    from the thread pool: fewer, larger slices for large dense state).
    Returns same shape as _native_st + (merge_ms).
    _vb reuses one concurrent bounds pass (same guard)."""
    try:
        keys = np.ascontiguousarray(keys)
        cols = [np.ascontiguousarray(v) for v in vcols]
        _bd = None if _vb is _SENT else _vb
        _mix = any(c.dtype.kind in "iu" for c in cols)
        # P2: int-bearing multi-lane aggregates with large dense state
        # take the decoupled sum-only path first (small-state / single-col
        # keeps the proven one-pass / fused SoA paths — see _p2_worth_it).
        _mixMulti = _mix and _p2_worth_it(len(cols), m)
        if not _mix and not _int_exact_ok(cols, _vb):
            return None
        if _mix and not _intlane_ok(cols, _vb):
            return None
        n = int(keys.size)
        t = max(1, min(int(t), n))
        if t == 1:
            r = _native_st(keys, cols, m)
            return (*r, 0.0) if r else None
        # Q5-P3: owner-group path for (2xi32 + 1xf) over large dense state
        # (disjoint group ranges of shared state, merge_ms=0). Any doubt
        # (backend/gate/shard failure) -> S-states below run verbatim.
        if _mix and _owner_worth_it(cols, m):
            _ow = _owner_mt(keys, cols, m, t, _bd)
            if _ow is not None:
                return _ow
        s = _mt_states(len(cols), m, t, n)
        bounds = np.linspace(0, n, s + 1).astype(np.int64)
        k32 = np.ascontiguousarray(keys, dtype=np.int32)
        ex = _pool(t)
        t0 = time.perf_counter()
        futs = []
        for w in range(s):
            a, b = int(bounds[w]), int(bounds[w + 1])
            kk = k32[a:b]
            vv = [c[a:b] for c in cols]

            def _work(kk=kk, vv=vv):
                if _mixMulti:
                    r = _native_sumonly(kk, vv, m, _bounds=_bd)
                    if r is not None:
                        return r
                if _mix:
                    r = _native_mixed(kk, vv, m, _bounds=_bd)
                    if r is not None:
                        return r
                if len(vv) == 1:
                    sums_dense, cc = _native_fused1(kk, vv[0], m)
                    return [sums_dense], cc
                return _native_fusedN(kk, vv, m)

            futs.append(ex.submit(_work))
        parts = [f.result() for f in futs]
        agg_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        ncols = len(parts[0][0])
        # Merge lanes stay exact: int32 worker states widen to int64 on
        # merge (per-slice i32 gate covers the kernel only; global totals
        # can exceed int32 -- numpy would wrap silently). f64 lanes merge
        # in f64 (same order -> bit-identical to the sequential merge).
        _mdt = [np.int64 if s.dtype.kind in "iu" else s.dtype
                for s in parts[0][0]]
        sums_m = [np.zeros(m, dtype=dt) for dt in _mdt]
        counts_m = np.zeros(m, dtype=np.int64)
        # Row-sharded deterministic merge (numpy releases GIL): M-lanes
        # split into ns shards, each shard accumulates states w=0..s-1 in
        # order -> bit-identical to the sequential merge. Shard count grows
        # with M (64K lanes/shard): small M stays at ~column-parallel
        # overhead, large M scales with core count. Pool is free here
        # (workers already joined).
        _ns = max(1, min(int(t), (int(m) // 65536) + 1))
        _mb = np.linspace(0, int(m), _ns + 1).astype(np.int64)

        def _mshard(dst, outs, s0, s1):
            sl = slice(int(s0), int(s1))
            if sl.start >= sl.stop:
                return
            x = dst[sl]
            for o in outs:
                x += o[sl]

        _mf = []
        for j in range(ncols):
            _outs = [ss[j] for ss, _ in parts]
            for w in range(_ns):
                _mf.append(ex.submit(_mshard, sums_m[j], _outs,
                                     _mb[w], _mb[w + 1]))
        _cout = [cc for _, cc in parts]
        for w in range(_ns):
            _mf.append(ex.submit(_mshard, counts_m, _cout,
                                 _mb[w], _mb[w + 1]))
        for _f in _mf:
            _f.result()
        merge_ms = (time.perf_counter() - t1) * 1000
    except Exception:  # noqa: BLE001
        return None
    try:
        ukeys, counts, out, cms = None, None, [], 0.0
        for s_m in sums_m:
            uk, c, s, cm = _native_compact(counts_m, s_m)
            if ukeys is None:
                ukeys, counts = uk, c
            cms += cm
            out.append(s)
        return ukeys, counts, out, agg_ms, cms, merge_ms
    except Exception:  # noqa: BLE001
        return None


def _native_mt_direct(keys, vcols, m, t, _bd):
    """Fail-fast MT over caller-cached bounds (Q4-B path, additive).

    Zero scans inside: int-lane gates run on _bd (cached (mn,mx) per int
    col, None for float/empty); a missing entry -> None (the verbatim
    chain owns it, never a rescan here). int32 lanes run FIRST: the
    range-checked i32 kernel validates every key before any unchecked
    lane (f64 single / int64 V_DENSE) runs, so a wrong cache degrades to
    None (BAD_RANGE/OVERFLOW -> fallback), never to a wrong answer. Lane
    reorder is exact (per-lane sums independent, shared counts identical;
    slice/merge order verbatim -> bit-identical f64). Requires >=1 int32
    lane (the validator); otherwise None. Slices/merges like _native_mt
    (S from _mt_states, iu widened to int64 on merge). Returns the same
    shape as _native_mt, or None (caller runs the verbatim chain).
    Measured 100M/16T exact, Q4 shape (M=101, 2xi32+1xf64): 184.7->93.5ms
    (-49%, beats 118 Polars target); Q1/Q2/Q3/Q5 paths untouched.
    """
    try:
        if _bd is None:
            return None
        if np.asarray(keys).dtype != np.dtype(np.int32):
            return None
        cols = [np.ascontiguousarray(v) for v in vcols]
        if any(np.asarray(c).dtype.kind not in "iuf" for c in cols):
            return None
        if any(np.asarray(c).dtype.kind in "iu"
               and np.asarray(c).dtype != np.dtype(np.int32) for c in cols):
            # Only int32 int lanes: the checked i32 kernel is the validator
            # (int64 V_DENSE is unchecked, so int64 lanes could wrap on an
            # understated cache instead of failing -> verbatim owns them).
            return None
        n = int(np.asarray(keys).size)
        m = int(m)
        if n == 0 or m <= 0:
            return None
        if len(_bd) != len(cols):
            return None
        for c, b in zip(cols, _bd):
            c = np.asarray(c)
            if c.dtype.kind in "iu":
                if b is None:
                    return None
                mn, mx = b
                if int(c.size) * max(abs(int(mn)), abs(int(mx))) >= 2 ** 63 - 1:
                    return None
        _ord = ([i for i, c in enumerate(cols)
                 if np.asarray(c).dtype == np.dtype(np.int32)]
                + [i for i, c in enumerate(cols)
                   if np.asarray(c).dtype != np.dtype(np.int32)])
        # Validator rule: >=1 int32 lane must take the CHECKED i32 path
        # under the cached bounds. It runs first and range-validates every
        # key before any unchecked lane (f64 single) runs; a wrong cache
        # (OOB key / real overflow) fails inside the checked kernel ->
        # None -> verbatim. Without a validator no unchecked lane may run
        # (int64 V_DENSE / f64 single never see unvalidated keys).
        _valid = [i for i in _ord
                  if np.asarray(cols[i]).dtype == np.dtype(np.int32)
                  and _i32_gate_ok(cols[i], n, _b=_bd[i])]
        if not _valid:
            return None
        _ord = _valid + [i for i in _ord if i not in _valid]
        _ocol = [cols[i] for i in _ord]
        _obd = [_bd[i] for i in _ord]
        t = max(1, min(int(t), n))
        if t == 1:
            _r = _native_mixed(np.ascontiguousarray(keys, dtype=np.int32),
                               _ocol, m, _bounds=_obd)
            if _r is None:
                return None
            _sums_o, _cc = _r
            _back = [None] * len(cols)
            for rank, i in enumerate(_ord):
                _back[i] = _sums_o[rank]
            try:
                ukeys, counts, out, cms = None, None, [], 0.0
                for s_m in _back:
                    uk, c, s, cm = _native_compact(_cc, s_m)
                    if ukeys is None:
                        ukeys, counts = uk, c
                    cms += cm
                    out.append(s)
                return ukeys, counts, out, 0.0, cms, 0.0
            except Exception:  # noqa: BLE001
                return None
        s = _mt_states(len(cols), m, t, n)
        bounds = np.linspace(0, n, s + 1).astype(np.int64)
        k32 = np.ascontiguousarray(keys, dtype=np.int32)
        ex = _pool(t)
        t0 = time.perf_counter()
        futs = []
        for w in range(s):
            a, b = int(bounds[w]), int(bounds[w + 1])
            kk = k32[a:b]
            vv = [c[a:b] for c in _ocol]

            def _work(kk=kk, vv=vv):
                return _native_mixed(kk, vv, m, _bounds=_obd)

            futs.append(ex.submit(_work))
        parts = [f.result() for f in futs]
        agg_ms = (time.perf_counter() - t0) * 1000
        if any(p is None for p in parts):
            return None
        t1 = time.perf_counter()
        ncols = len(parts[0][0])
        _mdt = [np.int64 if s.dtype.kind in "iu" else s.dtype
                for s in parts[0][0]]
        sums_m = [np.zeros(m, dtype=dt) for dt in _mdt]
        counts_m = np.zeros(m, dtype=np.int64)
        _ns = max(1, min(int(t), (int(m) // 65536) + 1))
        _mb = np.linspace(0, int(m), _ns + 1).astype(np.int64)

        def _mshard(dst, outs, s0, s1):
            sl = slice(int(s0), int(s1))
            if sl.start >= sl.stop:
                return
            x = dst[sl]
            for o in outs:
                x += o[sl]

        _mf = []
        for j in range(ncols):
            _outs = [ss[j] for ss, _ in parts]
            for w in range(_ns):
                _mf.append(ex.submit(_mshard, sums_m[j], _outs,
                                     _mb[w], _mb[w + 1]))
        _cout = [cc for _, cc in parts]
        for w in range(_ns):
            _mf.append(ex.submit(_mshard, counts_m, _cout,
                                 _mb[w], _mb[w + 1]))
        for _f in _mf:
            _f.result()
        merge_ms = (time.perf_counter() - t1) * 1000
    except Exception:  # noqa: BLE001
        return None
    try:
        _back_m = [None] * len(cols)
        for rank, i in enumerate(_ord):
            _back_m[i] = sums_m[rank]
        ukeys, counts, out, cms = None, None, [], 0.0
        for s_m in _back_m:
            uk, c, s, cm = _native_compact(counts_m, s_m)
            if ukeys is None:
                ukeys, counts = uk, c
            cms += cm
            out.append(s)
        return ukeys, counts, out, agg_ms, cms, merge_ms
    except Exception:  # noqa: BLE001
        return None


def _fused_pack_st(k1, k2, m2, v, m):
    """Single-thread fused pack+aggregate + carry (P4, generic one lane).

    k1/k2 int32 codes, m2 Python int, v int32 (i32-gated) or float
    (f64 lane); other dtypes -> None (proven path owns them, never a
    silent lossy route). Returns (ukeys, counts, sums, agg_ms,
    carry_ms) or None (fallback owns it).
    """
    try:
        k1 = np.ascontiguousarray(k1, dtype=np.int32)
        k2 = np.ascontiguousarray(k2, dtype=np.int32)
        v = np.ascontiguousarray(v)
        t0 = time.perf_counter()
        if v.dtype == np.dtype(np.int32):
            if not _i32_gate_ok(v, int(k1.size)):
                return None
            sums_m, counts_m = _fused_i32(k1, k2, m2, v, m)
        elif v.dtype.kind == "f":
            sums_m, counts_m = _fused_f64(k1, k2, m2, v, m)
        else:
            return None
        agg_ms = (time.perf_counter() - t0) * 1000
    except Exception:  # noqa: BLE001 -- fallback must never raise
        return None
    try:
        uk, c, s, cm = _native_compact(counts_m, sums_m)
        return uk, c, s, agg_ms, cm
    except Exception:  # noqa: BLE001
        return None


def _fused_pack_mt(k1, k2, m2, v, m, t):
    """Host MT fused pack+aggregate (P4): row-sharded fused workers.

    Same shape as _native_mt for one value column: worker-private i32
    states merged widened to int64 (global totals can exceed int32 --
    numpy would wrap silently), then one ColumnCarry. Returns
    (ukeys, counts, sums, agg_ms, carry_ms, merge_ms) or None.
    """
    try:
        k1 = np.ascontiguousarray(k1, dtype=np.int32)
        k2 = np.ascontiguousarray(k2, dtype=np.int32)
        v = np.ascontiguousarray(v)
        lane32 = v.dtype == np.dtype(np.int32)
        if lane32:
            if not _i32_gate_ok(v, int(k1.size)):
                return None
        elif v.dtype.kind != "f":
            return None
        n = int(k1.size)
        t = max(1, min(int(t), n)) if n else 1
        if t == 1:
            r = _fused_pack_st(k1, k2, m2, v, m)
            return (*r, 0.0) if r else None
        bounds = np.linspace(0, n, t + 1).astype(np.int64)
        ex = _pool(t)
        t0 = time.perf_counter()
        futs = []
        for w in range(t):
            a, b = int(bounds[w]), int(bounds[w + 1])

            def _work(a=a, b=b):
                if lane32:
                    return _fused_i32(k1[a:b], k2[a:b], m2, v[a:b], m)
                return _fused_f64(k1[a:b], k2[a:b], m2, v[a:b], m)

            futs.append(ex.submit(_work))
        parts = [f.result() for f in futs]
        agg_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        sums_m = np.zeros(m, dtype=np.int64)
        counts_m = np.zeros(m, dtype=np.int64)
        _ns = max(1, min(int(t), (int(m) // 65536) + 1))
        _mb = np.linspace(0, int(m), _ns + 1).astype(np.int64)

        def _mshard(dst, outs, s0, s1):
            sl = slice(int(s0), int(s1))
            if sl.start >= sl.stop:
                return
            x = dst[sl]
            for o in outs:
                x += o[sl]

        _mf = []
        _outs = [ss for ss, _ in parts]
        for w in range(_ns):
            _mf.append(ex.submit(_mshard, sums_m, _outs, _mb[w], _mb[w + 1]))
        _cout = [cc for _, cc in parts]
        for w in range(_ns):
            _mf.append(ex.submit(_mshard, counts_m, _cout, _mb[w], _mb[w + 1]))
        for _f in _mf:
            _f.result()
        merge_ms = (time.perf_counter() - t1) * 1000
    except Exception:  # noqa: BLE001
        return None
    try:
        uk, c, s, cm = _native_compact(counts_m, sums_m)
        return uk, c, s, agg_ms, cm, merge_ms
    except Exception:  # noqa: BLE001
        return None


def _fused_pack_consume(p, gnode, k1, k2, m2, kmin1, kmax1, kmin2, kmax2,
                        bufs, plan_groupby):
    """P4 fused pack+groupby (generic producer-consumer fusion, no queries).

    Guards (all generic observables; any failure -> None, proven pack +
    groupby run verbatim): consumer is one groupby sum/count/mean over
    this pack output; no validity sidecars on pack inputs/output, values
    or keys; values int32 (i32-gated) or float (f64 lane); Planner dense
    gate accepts the composite domain; fused backend present. Returns
    (ukeys, counts, sums, gi_dec, gi_ms, agg_ms, carry_ms, merge_ms,
    threads) or None.
    """
    try:
        if plan_groupby is None or gnode is None:
            return None
        if gnode.get("kernel_id", gnode.get("op")) != "groupby":
            return None
        if gnode["params"].get("op") not in ("sum", "count", "mean"):
            return None
        gins = gnode.get("inputs") or []
        if len(gins) != 2 or gins[1] != p["out"]:
            return None
        vname = gins[0]
        if vname not in bufs:
            return None
        for _n in (p["inputs"][0], p["inputs"][1], p["out"], vname, gins[1]):
            if _n + "#validity" in bufs:
                return None
        v = np.ascontiguousarray(bufs[vname])
        n = int(np.ascontiguousarray(k1).size)
        if int(v.size) != n:
            return None
        if int(kmin1) < 0 or int(kmin2) < 0:
            return None
        if not _fused_available():
            return None
        if v.dtype == np.dtype(np.int32):
            if not _i32_gate_ok(v, n):
                return None
        elif v.dtype.kind != "f":
            return None
        pkmin = int(kmin1) * int(m2) + int(kmin2)
        pkmax = int(kmax1) * int(m2) + int(kmax2)
        dec, mm, pms = _dense_gate(k1, plan_groupby, naggs=1,
                                   _kb=(pkmin, pkmax))
        if dec is None:
            return None
        t = _mt_threads()
        if t > 1:
            r = _fused_pack_mt(k1, k2, m2, v, int(mm), t)
            if r is None:
                return None
            uk, cc, ss, ams, cms, mms = r
            return uk, cc, ss, dec, pms, ams, cms, mms, t
        r = _fused_pack_st(k1, k2, m2, v, int(mm))
        if r is None:
            return None
        uk, cc, ss, ams, cms = r
        return uk, cc, ss, dec, pms, ams, cms, 0.0, 1
    except Exception:  # noqa: BLE001 -- proven path owns it
        return None

try:
    import pyarrow as pa
    import pyarrow.compute as pc

    _HAS_PA = True
except ImportError:  # correctness-only numpy fallback below
    pa, pc, _HAS_PA = None, None, False


def _fallback_err(what, fix="", doc=""):
    msg = str(what)
    if fix:
        msg += f" Fix: {fix}."
    if doc:
        msg += f" See {doc}"
    return ValueError(msg)


def _as_dtype(name, canonical_dtype):
    try:
        return np.dtype(canonical_dtype(name)["logical"])
    except (ValueError, KeyError):
        pass
    # Accumulator-only names (e.g. int64 from Core.accum_dtype) are not raw
    # series dtypes; still derived from Core, converted directly, never stored.
    if name in ("int64",):
        return np.dtype(name)
    raise _fallback_err(
        f"CPU driver: unsupported dtype '{name}': use int32/float32/float64",
        fix="pass dtype='int32'/'float32'/'float64'",
        doc="specs/06-drivers-gpu-cpu.md",
    ) from None


_PACK_I32_MAX = 2 ** 31 - 1


def _pack_i32_direct(k1, k2, m2, k1max, k2max, err):
    """One-alloc int32 mixed-radix composite c0*M2+c1 (production Q2 path).

    k1/k2 int32 codes, m2/k1max/k2max Python ints (probed once by caller).
    Bound k1max*m2+k2max <= 2**31-1 enforced on Python ints first: overflow
    raises an explicit error, never wraps. Single int32 alloc, no casts.
    Output bit-identical to the int64 radix composite by construction.
    """
    bound = int(k1max) * int(m2) + int(k2max)
    if bound > _PACK_I32_MAX:
        raise err(
            "CPU driver: pack_keys int32-direct composite overflows int32 "
            f"({k1max}*{m2}+{k2max}={bound}>2**31-1)",
            fix="Planner routes unsafe shapes to int64 radix; pass "
                "mode='pack' for stable bitpack labels",
            doc="specs/delta-2-composite-keys.md",
        )
    # Native pack primitive (same contract, one API); numpy below is the
    # fallback when the backend is absent. Bit-identical by construction.
    try:
        if _native_available():
            return _native_pack(np.ascontiguousarray(k1, dtype=np.int32),
                                np.ascontiguousarray(k2, dtype=np.int32),
                                int(m2))
    except Exception:  # noqa: BLE001 -- fallback owns it
        pass
    out = np.empty(k1.size, dtype=np.int32)
    np.multiply(k1, np.int32(m2), out=out)
    np.add(out, k2, out=out)
    return out


def _pack_radix_i64(ins, rad, err):
    """Legacy int64 mixed-radix composite (verbatim fallback, any range).

    Bound checked on Python ints before any int64 arithmetic: overflow
    raises an explicit error, never wraps; wide composites (e.g. Q36
    4-col) use the generic tuple path via pack_keys(mode='hash').
    """
    comp = ins[0].astype(np.int64, copy=False)
    cur = int(np.abs(comp).max(initial=0))
    for ci, mi in zip(ins[1:], rad[1:]):
        ci64 = ci.astype(np.int64, copy=False)
        cur = cur * int(mi) + int(np.abs(ci64).max(initial=0))
        if cur > 2**63 - 1:
            raise err(
                "CPU driver: pack_keys radix composite overflows int64",
                fix="use pack_keys(mode='hash') tuple path for wide composites",
                doc="specs/delta-2-composite-keys.md",
            )
        comp = comp * np.int64(mi) + ci64
    return comp


def _native_filter(a, eff):
    """Native select path (same contract as _filter_ref); None -> fallback.

    Never raises: backend absent or dtype uncovered -> numpy reference owns it.
    """
    try:
        if _select_available():
            return _native_select(a, eff)
    except Exception:  # noqa: BLE001 -- fallback owns it
        pass
    return None


def _native_mask(a, b, op):
    """Native mask combine (same contract as _mask_ref); None -> fallback."""
    try:
        if _select_available():
            if op == "and":
                return _native_mask_and(a, b)
            if op == "or":
                return _native_mask_or(a, b)
            if op == "not":
                return _native_mask_not(a)
    except Exception:  # noqa: BLE001 -- fallback owns it
        pass
    return None


def _filter_ref(values, mask, mask_valid=None):
    """Python reference: boolean selection, order preserved (WHERE semantics).

    Invalid mask rows (3VL) are excluded: effective = mask & mask_valid.
    Empty / all-true / all-false valid. Fancy-index copy (correctness first;
    zero-copy impossible for compacting selection -- output is a new compact
    buffer by construction). Returns (selected_values, effective_mask).
    Packed BitPack masks unpack bits -> bool once here (proven-necessary:
    the take needs a materialized row selector; never int32).
    """
    m = _to_bool_array(mask)
    if not isinstance(values, _BitPack):
        values = np.asarray(values)
    if m.size != values.size:
        raise ValueError(
            f"filter values/mask size mismatch {values.size} != {m.size}")
    eff = m if mask_valid is None else m & np.asarray(mask_valid, dtype=bool)
    return values[eff], eff


def _mask_ref(a, b, op):
    """Python reference: boolean combine (AND/OR/NOT -> BoolMask)."""
    a = np.asarray(a)
    if a.dtype != np.dtype(bool):
        raise TypeError(f"mask input must be bool, got {a.dtype}")
    if op == "not":
        if b is not None:
            raise ValueError("mask 'not' takes a single input")
        return ~a
    b = np.asarray(b)
    if b.dtype != np.dtype(bool):
        raise TypeError(f"mask input must be bool, got {b.dtype}")
    if a.size != b.size:
        raise ValueError(f"mask size mismatch {a.size} != {b.size}")
    if op == "and":
        return a & b
    if op == "or":
        return a | b
    raise ValueError(f"unknown mask op '{op}'")


def _gather_ref(values, indices, err):
    """Python reference: positional take, output order = indices order.

    Fast path: contiguous range [s, s+1, ..., s+k) -> zero-copy slice view.
    """
    values = np.asarray(values)
    idx = np.asarray(indices)
    if idx.dtype.kind not in "iu":
        raise err(
            f"CPU driver: gather needs integer indices, got {idx.dtype}",
            fix="pass int32/int64 positions",
            doc="specs/02-semantic-ir.md",
        )
    n = values.size
    if idx.size and (bool((idx < 0).any()) or bool((idx >= n).any())):
        raise err(
            "CPU driver: gather index out of range",
            fix="pass 0 <= i < n",
            doc="specs/02-semantic-ir.md",
        )
    # Fast path: contiguous range -> zero-copy slice (O(1), no alloc).
    if idx.size > 0:
        s = int(idx[0])
        k = idx.size
        if s >= 0 and s + k <= n:
            # Contiguous if: first + size - 1 == last AND step from first to
            # second is +1 (sort perm is always sorted so first two suffice).
            if k <= 1 or (int(idx[-1]) == s + k - 1
                          and int(idx[1]) - s == 1):
                return values[s:s + k]
    return values[idx]


def _sort_key_array(col, err, what):
    """Sortable key column: ints/floats/bool (any width, C-speed); TEXT and
    exotic kinds are rejected with an explicit error (encode TEXT to its
    dictionary numeric rank first -- sort never sees strings)."""
    a = col.to_array() if isinstance(col, _BitPack) else np.asarray(col)
    if a.dtype.kind in "iufb":
        return np.ascontiguousarray(a.reshape(-1))
    raise err(
        f"CPU driver: sort needs numeric keys, got {a.dtype} for {what}",
        fix="encode TEXT via dictionary_encode and sort its int32 rank codes",
        doc="specs/02-semantic-ir.md",
    )


def _stable_desc_idx(k):
    """Stable descending permutation of one key column (all C-speed).

    Ascending stable argsort, then group rows by equal key (vectorized change
    flags) and reverse the GROUP order while keeping input order inside each
    group. No negation (INT32_MIN-safe), no Python row/group loops.
    """
    asc = np.argsort(k, kind="stable")
    if asc.size == 0:
        return asc
    sk = k[asc]
    ch = np.empty(sk.shape, dtype=bool)
    ch[0] = True
    if sk.size > 1:
        ch[1:] = sk[1:] != sk[:-1]
    grp = np.cumsum(ch)
    rev = grp.max(initial=0) - grp
    return asc[np.argsort(rev, kind="stable")]


def _sort_perm(cols, descending, valid_mask, err):
    """Stable permutation for one (or lexicographic multi-) key sort.

    Successive stable passes from the LAST key to the first (lexicographic by
    stability). Invalid rows (valid_mask False) go last in input order, always.
    Returns int32 positions (int64 only if n exceeds int32 -- impossible on
    chunked backends, guarded by construction).
    """
    n = cols[0].size
    for c in cols:
        if c.size != n:
            raise err(
                f"CPU driver: sort keys size mismatch {c.size} != {n}",
                fix="pass equal-length key columns",
                doc="specs/02-semantic-ir.md",
            )
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if m.size != n:
            raise err(
                f"CPU driver: sort validity size {m.size} != keys {n}",
                fix="pass validity matching key length",
                doc="specs/delta-3-null-contract.md",
            )
        vpos = np.flatnonzero(m)
        ipos = np.flatnonzero(~m)
    else:
        vpos = np.arange(n)
        ipos = np.zeros(0, dtype=np.int64)
    idx = np.arange(vpos.size)
    for c, desc in zip(reversed(cols), reversed(descending)):
        k = c[vpos][idx] if vpos.size else c[:0]
        try:
            _s_step, _s_be = _native_sort_step(k, bool(desc))
        except Exception:  # noqa: BLE001 -- candidate never owns correctness
            _s_step = None
        if _s_step is None:
            _s_step = (_stable_desc_idx(k) if desc
                       else np.argsort(k, kind="stable"))
            idx = idx[_s_step]
        elif _s_be != "ident":
            idx = idx[_s_step]
        # "ident": same-dir monotonic -> idx unchanged (exact stable perm,
        # M2 guard design), skip the idx copy.
    # Direct int32 output — avoids int64 alloc + astype copy (~4 ms/1M).
    if n <= 2 ** 31 - 1:
        perm = np.empty(n, dtype=np.int32)
        perm[:vpos.size] = vpos[idx]
        perm[vpos.size:] = ipos
        return perm
    perm = np.empty(n, dtype=np.int64)
    perm[:vpos.size] = vpos[idx]
    perm[vpos.size:] = ipos
    return perm


def _slice_take(values, limit, offset, err):
    """Row-range take values[offset:offset+limit] (limit None -> to end).

    Offset beyond n yields an empty column (SQL semantics). Basic slice =
    view (zero-copy); BitPack stays packed via its own getitem.
    """
    n = values.n if isinstance(values, _BitPack) else np.asarray(values).size
    if offset >= n:
        return values[:0]
    end = n if limit is None else min(n, offset + limit)
    return values[offset:end]


def _shift_ref(values, validity, periods, err):
    """Positional shift N->N right by periods (vectorized, no per-row loop).

    values: numeric ndarray (int32/float32/float64; other numeric passes
    through with same semantics). validity: None (all valid) or bool mask.
    NaN is a value and rides with its row; only the sidecar decides NA.
    Returns (out_values, out_validity_or_None): out_validity None iff
    input had none AND periods == 0 (identity keeps the no-sidecar shape);
    otherwise a bool mask (head periods False). Data fill for new head
    rows is 0 (DELTA-3: invalid rows hold fill 0/0.0). Empty -> empty.
    """
    if isinstance(values, _BitPack):
        raise err(
            "CPU driver: shift needs numeric series, got packed bool/enum",
            fix="shift v1 covers int32/float32/float64 only",
            doc="specs/02-semantic-ir.md",
        )
    a = np.ascontiguousarray(np.asarray(values))
    if a.ndim != 1:
        raise err(
            f"CPU driver: shift needs rank-1, got shape {a.shape}",
            fix="pass a flat column",
            doc="specs/02-semantic-ir.md",
        )
    n = int(a.size)
    if n == 0:
        v = None if validity is None else np.zeros(0, dtype=bool)
        return np.ascontiguousarray(a.copy()), v
    if periods == 0:
        v = None if validity is None else np.ascontiguousarray(
            np.asarray(validity, dtype=bool).copy())
        return np.ascontiguousarray(a.copy()), v
    out = np.zeros(n, dtype=a.dtype)
    if validity is None:
        ov = np.zeros(n, dtype=bool)
        if periods < n:
            out[periods:] = a[:n - periods]
            ov[periods:] = True
        return np.ascontiguousarray(out), np.ascontiguousarray(ov)
    m = np.ascontiguousarray(np.asarray(validity, dtype=bool))
    if m.size != n:
        raise err(
            f"CPU driver: shift validity size {m.size} != values {n}",
            fix="pass validity matching values length",
            doc="specs/delta-3-null-contract.md",
        )
    ov = np.zeros(n, dtype=bool)
    if periods < n:
        out[periods:] = a[:n - periods]
        ov[periods:] = m[:n - periods]
    return np.ascontiguousarray(out), np.ascontiguousarray(ov)


def _cumsum_ref(values, validity, err):
    """Inclusive prefix sum N->N (vectorized, no per-row loop).

    values: int32/float32/float64 ndarray (rank-1). Other dtypes
    (packed bool/enum, text codes, int64 raw) are explicit errors.
    validity: None (all valid) or bool mask (DELTA-3 sidecar).
    NaN is a value: rides in data with IEEE forward propagation;
    never a validity signal. Invalid rows contribute 0 to the
    running sum; output validity is a per-row copy of the input
    (downstream valid rows resume, like shift carry). Data at
    invalid rows holds the prefix value (0-fill contribution);
    Series.to_numpy restores NaN at invalid float rows.
    int32 wraps mod 2**32 via an int64 accumulator (two's
    complement, never saturate/trap -- GPU scan lane policy).
    Floats accumulate in their own dtype (f32 stays f32).
    Empty -> empty (same dtype). Returns (out, out_valid_or_None):
    out_validity None iff input had none (all-valid, no sidecar).
    """
    if isinstance(values, _BitPack):
        raise err(
            "CPU driver: cumsum needs numeric series, got packed bool/enum",
            fix="cumsum v1 covers int32/float32/float64 only",
            doc="specs/02-semantic-ir.md",
        )
    a = np.ascontiguousarray(np.asarray(values))
    if a.ndim != 1:
        raise err(
            f"CPU driver: cumsum needs rank-1, got shape {a.shape}",
            fix="pass a flat column",
            doc="specs/02-semantic-ir.md",
        )
    if a.dtype != np.dtype(np.int32) and a.dtype != np.dtype(np.float32) \
            and a.dtype != np.dtype(np.float64):
        raise err(
            f"CPU driver: cumsum needs int32/float32/float64, got {a.dtype}",
            fix="cumsum v1 covers int32/float32/float64 only "
            "(bool/text/int64 have no cumsum semantics)",
            doc="specs/02-semantic-ir.md",
        )
    n = int(a.size)
    if n == 0:
        v = None if validity is None else np.zeros(0, dtype=bool)
        return np.ascontiguousarray(a.copy()), v
    if validity is None:
        # M7b: fused native scan replaces the 5-pass int32 chain below
        # (astype int64 -> cumsum -> %2**32 -> where -> astype int32) with one
        # pass and no int64 temporaries. Bit-identical (scan, not reduce, in
        # both lanes). Unconditional here -- the helper itself gates float
        # lanes at `_CUMSUM_FLOAT_MIN`; int32 is ungated (1.31-44.1x measured).
        out = _native_cumsum_lane(a)
        if out is not None:
            return np.ascontiguousarray(out), None
        if a.dtype == np.dtype(np.int32):
            acc = np.cumsum(a.astype(np.int64, copy=False), dtype=np.int64)
            w = acc % np.int64(2 ** 32)
            out = np.where(w >= np.int64(2 ** 31),
                           w - np.int64(2 ** 32), w).astype(np.int32)
            return np.ascontiguousarray(out), None
        return np.ascontiguousarray(
            np.cumsum(a, dtype=a.dtype)), None
    m = np.ascontiguousarray(np.asarray(validity, dtype=bool))
    if m.size != n:
        raise err(
            f"CPU driver: cumsum validity size {m.size} != values {n}",
            fix="pass validity matching values length",
            doc="specs/delta-3-null-contract.md",
        )
    if a.dtype == np.dtype(np.int32):
        filled = np.where(m, a, np.int32(0))
        # M7b: same fused lane on the 0-filled copy (identical result).
        out = _native_cumsum_lane(filled)
        if out is not None:
            return np.ascontiguousarray(out), np.ascontiguousarray(m.copy())
        acc = np.cumsum(filled.astype(np.int64, copy=False), dtype=np.int64)
        w = acc % np.int64(2 ** 32)
        out = np.where(w >= np.int64(2 ** 31),
                       w - np.int64(2 ** 32), w).astype(np.int32)
        return np.ascontiguousarray(out), np.ascontiguousarray(m.copy())
    zero = a.dtype.type(0)
    filled = np.where(m, a, zero)
    out = _native_cumsum_lane(filled)
    if out is not None:
        return np.ascontiguousarray(out), np.ascontiguousarray(m.copy())
    out = np.cumsum(filled.astype(a.dtype, copy=False), dtype=a.dtype)
    return np.ascontiguousarray(out), np.ascontiguousarray(m.copy())


def _valid_mask(bufs, *names):
    """AND of validity sidecars (DELTA-3); None if no NA present."""
    masks = [bufs[n + "#validity"] for n in names if n + "#validity" in bufs]
    if not masks:
        return None
    m = masks[0]
    for extra in masks[1:]:
        m = m & extra
    return m


def _packed_width(name, canonical_dtype):
    """Packed bit-width for bool/enum dtypes; None for numeric (verbatim path)."""
    try:
        info = canonical_dtype(name)
    except (ValueError, KeyError):
        return None
    return info.get("width") if isinstance(info, dict) else None


def _series_packed(values, width, err, what):
    """Series materialize for packed dtypes: validated once, stored as BitPack.

    bool accepts bool arrays or 0/1 ints (other values -> explicit error, no
    silent 2->True coercion); enum:W accepts int codes in [0, 2**W).
    BitPack input of matching width is kept by reference (zero-copy).
    """
    try:
        w = _check_width(width)
    except (ValueError, TypeError):
        raise err(
            f"CPU driver: {what} has unsupported packed width {width!r}",
            fix="pass dtype='bool'/'enum:2'/'enum:4'/'enum:8'",
            doc="specs/delta-5-bitmask-enum.md",
        ) from None
    if isinstance(values, _BitPack):
        if values.width != w:
            raise err(
                f"CPU driver: {what} packed width {values.width} != dtype width {w}",
                fix="repack to the series dtype width first",
                doc="specs/delta-5-bitmask-enum.md",
            )
        return values
    if _HAS_PA and isinstance(values, (pa.ChunkedArray, pa.Array)):
        values = values.to_numpy(zero_copy_only=False)
    arr = np.asarray(values)
    if arr.size == 0:
        base = np.zeros(0, dtype=bool if w == 1 else np.uint8)
        return _BitPack.from_bool(base) if w == 1 else _BitPack.from_codes(base, w)
    if arr.ndim != 1:
        raise err(
            f"CPU driver: {what} must be rank-1, got shape {arr.shape}",
            fix="pass a flat column",
            doc="specs/02-semantic-ir.md",
        )
    if w == 1:
        if arr.dtype == np.dtype(bool):
            return _BitPack.from_bool(np.ascontiguousarray(arr.ravel()))
        if arr.dtype.kind in "iu":
            v = np.ascontiguousarray(arr.ravel())
            if v.min(initial=0) < 0 or v.max(initial=0) > 1:
                raise err(
                    f"CPU driver: {what} bool series needs 0/1 values",
                    fix="pass bools or 0/1 ints (2 does not mean True)",
                    doc="specs/delta-5-bitmask-enum.md",
                )
            return _BitPack.from_bool(v.astype(bool, copy=False))
        raise err(
            f"CPU driver: {what} bool series needs bool/0/1 values, "
            f"got {arr.dtype}",
            fix="pass bools or 0/1 ints",
            doc="specs/delta-5-bitmask-enum.md",
        )
    if arr.dtype.kind not in "iu":
        raise err(
            f"CPU driver: {what} enum:{w} series needs int codes, got {arr.dtype}",
            fix="pass int codes in [0, 2**w)",
            doc="specs/delta-5-bitmask-enum.md",
        )
    try:
        return _BitPack.from_codes(np.ascontiguousarray(arr.ravel()), w)
    except ValueError as e:
        raise err(
            f"CPU driver: {what} {e}",
            fix=f"pass int codes in [0, {(1 << w) - 1}]",
            doc="specs/delta-5-bitmask-enum.md",
        ) from None


def _as_column(values, dtype, err, what, check_int32_range=None):
    """Bulk column materialize: ndarray/arrow by view (no copy when dtype
    matches), lists copied once. Never tolist()/object-array round-trips.

    Invariant #1: int64->int32 narrowing is range-checked first (OverflowError
    with what+range+value+fix, never a silent wrap). The checker arrives via
    kernel.alias (Core, injected by the entry); None keeps the legacy coerce
    path for standalone use only. int64 dtype stays int64 (no check, no
    narrow) for int64-capable ops (group/filter/sort/compare/agg-payload);
    execution codes stay int32 (sort positions, gather indices, pack_keys
    codes, inv codes, group key codes).
    """
    if _HAS_PA and isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if _HAS_PA and isinstance(values, pa.Array):
        values = values.to_numpy(zero_copy_only=False)
    if np.dtype(dtype) == np.dtype(np.int64):
        arr = np.asarray(values, dtype=np.dtype(np.int64))
    elif check_int32_range is not None and np.dtype(dtype) == np.dtype(np.int32):
        # Single-pass narrowing guard (no discovery overhead): int32 input is a
        # view; other ndarrays are probed in place (C-speed min/max); lists are
        # filled once as int64 (same single fill as the legacy direct cast),
        # probed, then cast. OverflowError before any wrap, never a silent wrap.
        if isinstance(values, np.ndarray) and values.dtype == np.dtype(np.int32):
            arr = values
        elif isinstance(values, np.ndarray):
            check_int32_range(values, what)
            arr = values.astype(np.int32, copy=False)
        else:
            wide = np.asarray(values, dtype=np.int64)
            check_int32_range(wide, what)
            arr = wide.astype(np.int32, copy=False)
    else:
        arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise err(
            f"CPU driver: {what} must be rank-1, got shape {arr.shape}",
            fix="pass a flat column",
            doc="specs/02-semantic-ir.md",
        )
    return arr


def _as_pa_string(values, err):
    """Bulk string view for encode_pattern (Arrow zero-copy when input is
    already an Arrow string column; one C-speed bulk copy from numpy <U)."""
    if not _HAS_PA:
        raise err(
            "CPU driver: encode_pattern needs pyarrow for bulk strings",
            fix="install pyarrow or pass small lists (fallback path)",
            doc="specs/delta-4-pattern-strings.md",
        )
    if isinstance(values, pa.ChunkedArray):
        s = values.combine_chunks()
    elif isinstance(values, pa.Array):
        s = values
    elif isinstance(values, np.ndarray) and values.dtype.kind == "U":
        s = pa.array(values)
    elif isinstance(values, np.ndarray) and values.dtype.kind == "O":
        try:
            s = pa.array(values, type=pa.string())
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError) as e:
            raise err(
                f"CPU driver: encode_pattern needs a string column, got object mix ({e})",
                fix="pass prefix+int strings or None (nullable), never raw numbers",
                doc="specs/delta-4-pattern-strings.md",
            ) from None
    elif isinstance(values, (list, tuple)):
        try:
            s = pa.array(values, type=pa.string())
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError) as e:
            raise err(
                f"CPU driver: encode_pattern needs a string column ({e})",
                fix="pass prefix+int strings or None (nullable), never raw numbers",
                doc="specs/delta-4-pattern-strings.md",
            ) from None
    else:
        raise err(
            f"CPU driver: encode_pattern needs a string column, got {type(values).__name__}",
            fix="pass a numpy <U / arrow string / list[str|None] column",
            doc="specs/delta-4-pattern-strings.md",
        )
    if not (pa.types.is_string(s.type) or pa.types.is_large_string(s.type)):
        raise err(
            f"CPU driver: encode_pattern needs a string column, got {s.type}",
            fix="encode categoricals as prefix+int strings first",
            doc="specs/delta-4-pattern-strings.md",
        )
    return s


def _encode_pattern_vec(values, prefix, err):
    """Generic prefix+int -> (codes int32, validity bool, width|None). Bulk.

    Contract: exact prefix match; numeric body strictly optional '-' +
    digits (whitespace/underscore forms -> invalid, no silent coercion);
    int32 overflow -> explicit error (no silent wrap);
    non-matching/None -> invalid (None-equivalent), never an error.
    Width sidecar = max digit-width of valid rows; lossless restore
    f"{prefix}{code:0{width}d}" is exact for fixed-width patterns.
    """
    if _HAS_PA:
        s = _as_pa_string(values, err)
        n = len(s)
        codes = np.zeros(n, dtype=np.int32)
        valid = np.zeros(n, dtype=bool)
        width = None
        if n == 0:
            return codes, valid, width
        starts = pc.fill_null(pc.starts_with(s, pattern=prefix), False)
        nbytes = len(prefix.encode("utf-8"))
        suffix = pc.utf8_slice_codeunits(s, start=nbytes)
        # Sign-aware digit core: optional single leading '-' stripped, then
        # ALL-ASCII-decimal + non-empty (regex-free: ~13x faster than RE2).
        neg = pc.fill_null(pc.starts_with(suffix, pattern="-"), False)
        core = pc.if_else(neg, pc.utf8_slice_codeunits(suffix, start=1), suffix)
        ok_digits = pc.fill_null(pc.ascii_is_decimal(core), False)
        ok_len = pc.fill_null(pc.greater(pc.utf8_length(core), 0), False)
        ok = pc.and_(pc.and_(starts, ok_digits), ok_len)
        ok_np = ok.to_numpy(zero_copy_only=False)
        idx = np.nonzero(ok_np)[0]
        if idx.size:
            # Width recorded for lossless restore of fixed-width patterns
            # (H2O id%03d/id%010d); variable-width bodies stay valid, the
            # sidecar keeps max width and decode pads explicitly (documented).
            core_len = pc.utf8_length(core).to_numpy(zero_copy_only=False)
            width = int(core_len[idx].max())
        if idx.size:
            safe = pc.if_else(ok, suffix, pa.scalar("0"))
            try:
                ints = pc.cast(safe, pa.int32(), safe=True)
            except pa.ArrowInvalid as e:
                raise err(
                    f"CPU driver: encode_pattern int32 overflow for prefix '{prefix}': {e}",
                    fix="use a wider code dtype path or shorter pattern ints",
                    doc="specs/delta-4-pattern-strings.md",
                ) from None
            codes_all = ints.to_numpy(zero_copy_only=False)
            codes[idx] = codes_all[idx]
            valid[idx] = True
        return codes, valid, width
    # No-pyarrow path: pure-NumPy vectorized bulk encode (no per-row Python
    # loop at any N). Contract mirrors the Arrow branch exactly: exact prefix
    # match; numeric body = optional single '-' + non-empty ASCII digits
    # (non-ASCII decimals rejected, same as ascii_is_decimal); int32 overflow
    # raises; non-matching/None/non-str -> invalid. Work:
    #   1. normalize to 1-D object (C-speed, one pass, no list() blowup);
    #   2. None mask (elementwise C loop);
    #   3. one astype('<U') (C loop) + zero-copy uint32 codepoint view;
    #   4. prefix/body/digit validation as integer matrix masks;
    #   5. values via vectorized Horner in int64 (exact: non-negative partials
    #      bounded by the final magnitude, <=18 digits always fit; >=19 digits
    #      always overflow int32 -> explicit error, same as safe Arrow cast).
    # Width sidecar = max digit-width of valid rows (codepoints, same unit as
    # Arrow utf8_length). <U input skips the str-identity check (provably all
    # str); object/list input checks it via one elementwise compare.
    if not isinstance(values, (np.ndarray, list, tuple)):
        try:
            values = list(values)
        except TypeError:
            raise err(
                f"CPU driver: encode_pattern needs a string column, got {type(values).__name__}",
                fix="pass a numpy <U / list[str|None] column",
                doc="specs/delta-4-pattern-strings.md",
            ) from None
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1:
        raise err(
            f"CPU driver: encode_pattern needs a rank-1 column, got shape {arr.shape}",
            fix="pass a flat string column",
            doc="specs/02-semantic-ir.md",
        )
    n = arr.size
    codes = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=bool)
    width = None
    if n == 0:
        return codes, valid, width
    try:
        nonnull = np.asarray(arr != None, dtype=bool)  # noqa: E711 -- elementwise C loop
    except TypeError as e:
        raise err(
            f"CPU driver: encode_pattern needs a string column (exotic scalars: {e})",
            fix="pass prefix+int strings or None (nullable), never raw numbers",
            doc="specs/delta-4-pattern-strings.md",
        ) from None
    if not bool(nonnull.any()):
        return codes, valid, width
    s = arr[nonnull]
    try:
        U = s.astype("U")
    except (TypeError, ValueError) as e:
        raise err(
            f"CPU driver: encode_pattern needs a string column ({e})",
            fix="pass prefix+int strings or None (nullable), never raw numbers",
            doc="specs/delta-4-pattern-strings.md",
        ) from None
    if isinstance(values, np.ndarray) and values.dtype.kind == "U":
        is_str = np.ones(s.size, dtype=bool)
    else:
        try:
            is_str = np.asarray(s == U, dtype=bool)
        except TypeError as e:
            raise err(
                f"CPU driver: encode_pattern needs a string column (exotic scalars: {e})",
                fix="pass prefix+int strings or None (nullable), never raw numbers",
                doc="specs/delta-4-pattern-strings.md",
            ) from None
    if not bool(is_str.any()):
        return codes, valid, width
    Uc = U[is_str]
    Lc = np.char.str_len(Uc)
    W = int(U.itemsize // 4)
    C = Uc.view(np.uint32).reshape(Uc.size, W)
    P = len(prefix)
    if P > W:
        return codes, valid, width  # no row can even hold the prefix
    pfx = np.array([ord(c) for c in prefix], dtype=np.uint32)
    prefix_ok = (Lc >= P) & np.all(C[:, :P] == pfx, axis=1)
    body_len = np.where(prefix_ok, Lc - P, 0)
    has_body = body_len >= 1
    sign = has_body & (C[:, P] == np.uint32(45))
    core_start = P + sign.astype(np.int64)
    core_len = np.where(has_body, body_len - sign.astype(np.int64), 0)
    has_core = core_len >= 1
    JR = np.arange(W)
    valid_pos = (JR >= core_start[:, None]) & (JR < (core_start + core_len)[:, None])
    dig = (C >= np.uint32(48)) & (C <= np.uint32(57))
    valid_c = has_core & np.all(dig | ~valid_pos, axis=1)
    if bool(valid_c.any()):
        clen_v = core_len[valid_c]
        if bool((clen_v >= 19).any()):
            bad = Uc[is_str][valid_c][np.nonzero(clen_v >= 19)[0][0]]
            raise err(
                f"CPU driver: encode_pattern int32 overflow for value '{bad}'",
                fix="use a wider code dtype path or shorter pattern ints",
                doc="specs/delta-4-pattern-strings.md",
            )
        # Vectorized Horner over absolute codepoint columns (W vector steps,
        # no N loop). Each row's core is contiguous and columns ascend, so a
        # masked per-column update keeps digit order per row; where() discards
        # inactive columns (both branches evaluated, garbage never scattered).
        Cv = C[valid_c]
        cstart_v = core_start[valid_c]
        clen_all = core_len[valid_c]
        pos_v = (JR >= cstart_v[:, None]) & (JR < (cstart_v + clen_all)[:, None])
        mag = np.zeros(valid_c.sum(), dtype=np.int64)
        for j in range(W):
            step = np.where(
                pos_v[:, j], (Cv[:, j].astype(np.int64) - np.int64(48)), np.int64(0))
            mag = np.where(pos_v[:, j], mag * np.int64(10) + step, mag)
        neg_v = sign[valid_c]
        over = (~neg_v & (mag > np.int64(2 ** 31 - 1))) | (neg_v & (mag > np.int64(2 ** 31)))
        if bool(over.any()):
            bad = Uc[is_str][valid_c][np.nonzero(over)[0][0]]
            raise err(
                f"CPU driver: encode_pattern int32 overflow for value '{bad}'",
                fix="use a wider code dtype path or shorter pattern ints",
                doc="specs/delta-4-pattern-strings.md",
            )
        value = np.where(neg_v, -mag, mag).astype(np.int32)
        full_idx = np.nonzero(nonnull)[0][np.nonzero(is_str)[0][np.nonzero(valid_c)[0]]]
        codes[full_idx] = value
        valid[full_idx] = True
        width = int(clen_v.max())
    return codes, valid, width


def _text_norm(values, err):
    """Shared TEXT normalization: 1-D str|None column -> (n, full_idx, U).

    full_idx: positions of valid str rows; U: <U array of those rows.
    Non-str scalars -> explicit error (no silent coercion, same rule as
    dictionary_encode). None -> invalid (DELTA-3), never an error.
    Dict ENC shapes never reach here (each op routes them to its D-scale
    path first); a stray dict is an explicit error, never a key-list.
    """
    if isinstance(values, dict):
        raise err(
            "CPU driver: TEXT op got a dict carrier (ENC shape or mapping).",
            fix="pass the flat str column, or route ENC shapes via the op's "
            "dictionary path (text_contains / text_regex_replace)",
            doc="specs/05-storage-encoding.md",
        )
    if not isinstance(values, (np.ndarray, list, tuple)):
        try:
            values = list(values)
        except TypeError:
            raise err(
                f"CPU driver: TEXT op needs a string column, got {type(values).__name__}",
                fix="pass a numpy <U / list[str|None] column",
                doc="specs/05-storage-encoding.md",
            ) from None
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1:
        raise err(
            f"CPU driver: TEXT op needs a rank-1 column, got shape {arr.shape}",
            fix="pass a flat string column",
            doc="specs/02-semantic-ir.md",
        )
    n = arr.size
    if n == 0:
        return 0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype="U1")
    try:
        nonnull = np.asarray(arr != None, dtype=bool)  # noqa: E711 -- elementwise C loop
    except TypeError as e:
        raise err(
            f"CPU driver: TEXT op needs a string column (exotic scalars: {e})",
            fix="pass str or None per row, never raw numbers",
            doc="specs/05-storage-encoding.md",
        ) from None
    if not bool(nonnull.any()):
        return n, np.zeros(0, dtype=np.int64), np.zeros(0, dtype="U1")
    s = arr[nonnull]
    try:
        U = s.astype("U")
    except (TypeError, ValueError) as e:
        raise err(
            f"CPU driver: TEXT op needs a string column ({e})",
            fix="pass str or None per row, never raw numbers",
            doc="specs/05-storage-encoding.md",
        ) from None
    if isinstance(values, np.ndarray) and values.dtype.kind == "U":
        is_str = np.ones(s.size, dtype=bool)
    else:
        try:
            is_str = np.asarray(s == U, dtype=bool)
        except TypeError as e:
            raise err(
                f"CPU driver: TEXT op needs a string column (exotic scalars: {e})",
                fix="pass str or None per row, never raw numbers",
                doc="specs/05-storage-encoding.md",
            ) from None
    if not bool(is_str.all()):
        bad = s[np.nonzero(~is_str)[0][0]]
        raise err(
            f"CPU driver: TEXT op needs a string column, got "
            f"{type(bad).__name__} value {bad!r}",
            fix="pass str or None per row, never raw numbers",
            doc="specs/05-storage-encoding.md",
        )
    full_idx = np.nonzero(nonnull)[0]
    return n, full_idx, U


def _text_data_offs(U):
    """Valid-row strs -> (data, offs) for the data+offs ABI (bulk, no per-row loop).

    Arrow path: one C-speed bulk transcode <U/object -> UTF-8 + zero-copy
    buffer views (no .tolist(), no per-row .encode, no b"".join, no cumsum).
    Byte-identical to build_text_buffers(U.tolist()). No-pyarrow path keeps
    the old prep verbatim (fallback/WASM parity). True zero-copy from <U is
    impossible (UCS4 4B/char layout); the remaining copy is the mandatory
    UTF-8 transcode, done once in C++.
    """
    if _HAS_PA:
        bufs = pa.array(U).buffers()
        return bufs[2], np.ascontiguousarray(
            np.frombuffer(bufs[1], dtype=np.int32))
    return _native_build_text(list(U))


def _native_text_length(U, data_offs=None):
    """Native TEXT length over valid-row U (<U array, no tolist); None -> reference owns it."""
    try:
        if _text_available():
            data, offs = data_offs if data_offs is not None else _text_data_offs(U)
            return _native_text_len(data, offs)
    except Exception:  # noqa: BLE001 -- fallback owns it
        pass
    return None


def _native_text_contains(U, substr):
    """Native TEXT contains over valid-row U (<U array, no tolist); None -> reference owns it."""
    try:
        if _text_available():
            data, offs = _text_data_offs(U)
            return _native_text_hit(data, offs, substr)
    except Exception:  # noqa: BLE001 -- fallback owns it
        pass
    return None


def _native_text_affix(U, needle, kind):
    """Native TEXT affix/equals over valid-row U (<U array, no tolist); None -> reference owns it."""
    try:
        if _text_available():
            data, offs = _text_data_offs(U)
            if kind == "startswith":
                return _native_text_head(data, offs, needle)
            if kind == "endswith":
                return _native_text_tail(data, offs, needle)
            return _native_text_eq(data, offs, needle)
    except Exception:  # noqa: BLE001 -- fallback owns it
        pass
    return None


def _text_length_vec(values, err):
    """TEXT length (code points, NOT bytes) -> (lens int32[N], valid bool[N]).

    Native first (Rust nf_text_length over UTF-8 buffers); numpy
    np.char.str_len reference owns every native failure. None/non-str
    rows -> 0 + invalid (DELTA-3), never an error. Dictionary ENC
    shapes run the D-scale path (length over D uniques once, numeric
    broadcast over codes); flat columns keep the verbatim path below.
    Zero-copy: header-only split (no strings list) + buffer-direct
    Arrow scan (no eager full-body decode); strings materialize only
    on the numpy fallback.
    """
    hdr = _dict_header(values)
    if hdr is not None:
        ca, row_valid, data, offs, d = hdr
        n = int(ca.size)
        lens = np.zeros(n, dtype=np.int32)
        if d == 0 or not bool(row_valid.any()):
            return lens, np.ascontiguousarray(row_valid)
        dl = None
        arr = _dict_arrow_array(data, offs, d)
        if arr is not None:
            try:
                m = pc.utf8_length(arr).to_numpy(zero_copy_only=False)
                dl = np.ascontiguousarray(
                    np.asarray(m, dtype=np.int32))
            except Exception:  # noqa: BLE001 -- numpy fallback owns it
                dl = None
        if dl is None:
            full = _dict_column_parts(values)
            if full is not None:
                dl = np.ascontiguousarray(
                    np.char.str_len(np.asarray(full[0], dtype=str)),
                    dtype=np.int32)
        if dl is not None:
            ix = np.nonzero(row_valid)[0]
            lens[ix] = dl[ca[ix].astype(np.intp)]
            return lens, np.ascontiguousarray(row_valid)
        # else: unusable dict shape after all -> flat path owns it (as before)
    n, full_idx, U = _text_norm(values, err)
    lens = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=bool)
    if full_idx.size == 0:
        return lens, valid
    got = _native_text_length(U)
    if got is None:
        got = np.char.str_len(U).astype(np.int32)
    lens[full_idx] = np.ascontiguousarray(got, dtype=np.int32)
    valid[full_idx] = True
    return lens, valid


def _dict_header(values):
    """ENC header split -> (codes int32[N], row_valid, data, offs, d) or None.

    No strings-list materialization (40ms/2.6M @10M): D-scale Arrow
    consumers scan buffers directly; strings are built lazily only on
    the numpy fallback. TEXT-shape validation matches
    _dict_column_parts exactly (non-TEXT dicts and malformed offset
    shapes -> None, caller keeps its flat path verbatim).
    """
    if not isinstance(values, dict):
        return None
    codes = values.get("codes")
    if codes is None:
        return None
    body = values.get("dictionary")
    dvals = values.get("values")
    data = offs = None
    if body is not None:
        data = getattr(body, "utf8_data", None)
        if data is None:
            data = getattr(body, "data", None)
        offs = getattr(body, "offsets", None)
    if data is None or offs is None:
        data = values.get("utf8_data") or values.get("data")
        offs = values.get("offsets")
    d = None
    if isinstance(dvals, (list, tuple)):
        if len(dvals) == 0:
            d = 0
        elif all(isinstance(x, str) for x in dvals[:8]):
            d = len(dvals)
        else:
            return None  # non-TEXT (e.g. int64) dict: flat path owns the error
    if d is None:
        if data is None or offs is None or not isinstance(data, (bytes, bytearray)):
            return None  # not an ENC shape: flat path owns it
        try:
            o = np.asarray(offs, dtype=np.int32)
            if o.size == 0 or int(o[0]) != 0 or int(o[-1]) != len(data):
                return None
            if bool((o[1:] < o[:-1]).any()):
                return None
            d = int(o.size) - 1
        except (UnicodeDecodeError, ValueError):
            return None
    ca = np.ascontiguousarray(np.asarray(codes, dtype=np.int32))
    n = int(ca.size)
    row_valid = (ca >= 0) & (ca < d) if d > 0 else np.zeros(n, dtype=bool)
    row_valid = np.ascontiguousarray(row_valid)
    vma = values.get("validity")
    if vma is not None:
        vma = np.asarray(vma, dtype=bool)
        if vma.size == n:
            row_valid = row_valid & vma
    return ca, row_valid, data, offs, d


def _dict_arrow_array(data, offs, d):
    """Zero-copy Arrow string array over dictionary body buffers or None.

    No eager full-body UTF-8 decode (675ms/302MB @10M): ENC bodies are
    valid UTF-8 by construction and Arrow validates during the scan;
    any failure returns None and the caller keeps its numpy fallback
    (which owns every error explicitly). Buffers wrap by reference
    (bytes/bytearray/int32-contiguous pass without copy).
    """
    if not _HAS_PA:
        return None
    if not isinstance(data, (bytes, bytearray)):
        return None
    if offs is None or d <= 0:
        return None
    try:
        o = np.asarray(offs, dtype=np.int32)
        if o.size != d + 1 or int(o[0]) != 0 or int(o[-1]) != len(data):
            return None
        if bool((o[1:] < o[:-1]).any()):
            return None
        o = np.ascontiguousarray(o)
        return pa.Array.from_buffers(pa.string(), d,
                                     [None, pa.py_buffer(o),
                                      pa.py_buffer(data)])
    except Exception:  # noqa: BLE001 -- numpy fallback owns it
        return None


def _dict_bytescan(data, offs, d, substr):
    """Native byte-substring scan over (data, offs) -> bool[D] or None.

    Byte search == code-point contains for valid UTF-8 (self-synchronizing);
    ENC bodies arrive valid from dictionary_encode. Zero-copy read views, no
    per-string objects, no UTF-8 re-validation pass. Any doubt (native
    unavailable, malformed offsets, matcher error) -> None: Arrow lane owns
    it next. Lanes: (1) Rust nf_text_contains via _lib.native_cpu
    (ctypes ABI, bit-exact vs the retired JIT lane, ~9.5x over numpy on
    100k/seed-42 parity probe); (2) Arrow match_substring; (3) numpy
    char.find oracle. Generic over needle length and D.
    """
    try:
        o = np.ascontiguousarray(np.asarray(offs, dtype=np.int32))
        if o.size != int(d) + 1 or int(o[0]) != 0:
            return None
        if int(o[-1]) != len(data) or bool((o[1:] < o[:-1]).any()):
            return None
        nb = substr.encode("utf-8")
        if len(nb) == 0:
            return np.ones(int(d), dtype=bool)
        try:
            from _lib.native_cpu import text_contains_buffers as _tcb
        except ImportError:
            return None
        try:
            out = _tcb(data, o, nb)
        except RuntimeError:
            return None
        return np.ascontiguousarray(np.asarray(out, dtype=bool))
    except Exception:  # noqa: BLE001 -- Arrow lane owns it
        return None


def _dict_column_parts(values):
    """ENC-shape split -> (strings[D], codes int32[N], row_valid, data, offs).

    Accepts (a) ENC output {codes, dictionary body|values[D], validity}
    and (b) raw body dict {utf8_data|data, offsets, codes, validity}.
    Row validity = codes in range AND ENC validity (DELTA-3).
    Returns None when values is not a TEXT dict ENC shape: non-dict
    carriers, missing codes, non-TEXT (e.g. int64) dictionaries and
    malformed bodies all return None and the caller keeps its flat
    path (which owns every error explicitly). Self-contained (no
    cross-Extension import).
    """
    hdr = _dict_header(values)
    if hdr is None:
        return None
    ca, row_valid, data, offs, _d = hdr
    dvals = values.get("values")
    strings = None
    if isinstance(dvals, (list, tuple)):
        strings = [] if len(dvals) == 0 else list(dvals)
    if strings is None:
        try:
            o = np.ascontiguousarray(np.asarray(offs, dtype=np.int32))
            mv = memoryview(bytes(data))
            strings = [bytes(mv[int(a):int(b)]).decode("utf-8")
                       for a, b in zip(o[:-1].tolist(), o[1:].tolist())]
        except (UnicodeDecodeError, ValueError):
            return None
    return strings, ca, row_valid, data, offs


def _dict_contains_codes(values, substr):
    """D-scale contains over ENC shapes -> (hit, valid) or None.

    String work happens on D unique strings only (native byte-scan first,
    Arrow vector second, one numpy pass fallback), broadcast to rows via
    numeric fancy-indexing over codes (no Python loop over N). Row validity =
    codes in range AND ENC validity; invalid rows never match (3VL).
    Returns None when values is not a dict ENC shape (caller keeps the
    flat path verbatim). Self-contained (no cross-Extension import).
    Zero-copy: header-only split (no strings list) + buffer-direct
    scans (no eager full-body decode); strings materialize only
    on the numpy fallback.
    """
    hdr = _dict_header(values)
    if hdr is None:
        return None
    ca, row_valid, data, offs, d = hdr
    n = int(ca.size)
    hit = np.zeros(n, dtype=bool)
    if substr == "":
        hit[row_valid] = True
        return hit, row_valid
    if d == 0:
        return hit, row_valid
    allow = _dict_bytescan(data, offs, d, substr)
    if allow is None:
        arr = _dict_arrow_array(data, offs, d)
        if arr is not None:
            try:
                m = pc.match_substring(arr, substr).to_numpy(zero_copy_only=False)
                allow = np.ascontiguousarray(np.asarray(m, dtype=bool))
            except Exception:  # noqa: BLE001 -- numpy fallback owns it
                allow = None
    if allow is None:
        full = _dict_column_parts(values)
        if full is None:
            return None  # not a usable dict shape: flat path owns the error
        try:
            allow = (np.char.find(np.asarray(full[0], dtype=str), substr) != -1)
        except (TypeError, ValueError):
            return None  # non-str dictionary: flat path owns the error
        allow = np.ascontiguousarray(allow, dtype=bool)
    ix = np.nonzero(row_valid)[0]
    if ix.size > 0:
        hit[ix] = allow[ca[ix].astype(np.intp)]
    return hit, row_valid


def _text_contains_vec(values, substr, err):
    """TEXT contains (UTF-8 substring, case-sensitive) -> (hit, valid).

    Empty substr matches every valid row. Invalid rows never match
    (3VL). Dictionary ENC shapes run the D-scale path (string work on D
    uniques, numeric broadcast over codes); flat columns keep the
    native-first verbatim path below.
    """
    if not isinstance(substr, str):
        raise err(
            f"CPU driver: text_contains substr must be str, got {type(substr).__name__}",
            fix="pass a str literal",
            doc="specs/05-storage-encoding.md",
        )
    via_dict = _dict_contains_codes(values, substr)
    if via_dict is not None:
        return via_dict
    n, full_idx, U = _text_norm(values, err)
    hit = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    if full_idx.size == 0:
        return hit, valid
    got = _native_text_contains(U, substr)
    if got is None:
        if substr == "":
            got = np.ones(full_idx.size, dtype=np.uint8)
        else:
            got = (np.char.find(U, substr) != -1).astype(np.uint8)
    hit[full_idx] = np.ascontiguousarray(got, dtype=np.uint8).astype(bool)
    valid[full_idx] = True
    return hit, valid


def _text_affix_vec(values, needle, kind, err):
    """TEXT startswith/endswith/equals -> (hit, valid). Native first."""
    if not isinstance(needle, str):
        raise err(
            f"CPU driver: text_{kind} needle must be str, got {type(needle).__name__}",
            fix="pass a str literal",
            doc="specs/05-storage-encoding.md",
        )
    n, full_idx, U = _text_norm(values, err)
    hit = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    if full_idx.size == 0:
        return hit, valid
    got = _native_text_affix(U, needle, kind)
    if got is None:
        if kind == "startswith":
            got = np.char.startswith(U, needle).astype(np.uint8)
        elif kind == "endswith":
            got = np.char.endswith(U, needle).astype(np.uint8)
        else:
            got = (U == needle).astype(np.uint8)
    hit[full_idx] = np.ascontiguousarray(got, dtype=np.uint8).astype(bool)
    valid[full_idx] = True
    return hit, valid


def _text_regex_replace_vec(values, pattern, repl, err):
    """TEXT regexp_replace -> (out object[N], valid bool[N]).

    CPU-only v2: dictionary fast path (D-scale regex + codes broadcast)
    + flat list-comprehension fallback.

    DuckDB regexp_replace parity (frozen): first match only, backrefs,
    empty-pattern-at-0, case-sensitive. Invalid (None/non-str)
    rows -> None + invalid (DELTA-3), never a match, never an
    error. Bad pattern -> explicit error (never silent NULL).
    pattern/repl must be str (else explicit error).

    Dictionary fast path: if values is a dict with utf8_data+offsets+codes
    (DictionaryBody representation), decode D unique strings, apply regex
    to D only, broadcast result via numpy fancy-indexing over codes.
    Flat fallback: list comprehension over object array (eliminates
    per-row np.asarray overhead and explicit loop index).
    """
    import re as _re
    import time as _t
    _tv = _t.perf_counter()
    if not isinstance(pattern, str):
        raise err(
            f"CPU driver: text_regex_replace pattern must be str, got {type(pattern).__name__}",
            fix="pass a str literal",
            doc="specs/05-storage-encoding.md",
        )
    if not isinstance(repl, str):
        raise err(
            f"CPU driver: text_regex_replace repl must be str, got {type(repl).__name__}",
            fix="pass a str literal",
            doc="specs/05-storage-encoding.md",
        )
    try:
        rx = _re.compile(pattern)
    except _re.error as e:
        raise err(
            f"CPU driver: text_regex_replace bad pattern {pattern!r}: {e}",
            fix="pass a valid Python/DuckDB-compatible regex",
            doc="specs/05-storage-encoding.md",
        ) from None
    _tv_ms = (_t.perf_counter() - _tv) * 1000

    # --- Dictionary fast path: D-scale regex, broadcast via codes ---
    # Accepts (a) raw body dict {utf8_data|data, offsets, codes, validity}
    # and (b) ENC output {codes, values[D list], dictionary body, validity}.
    # String work happens on values[D] only, never on rows[N]; broadcast
    # to rows is numeric fancy-indexing over codes (no Python loop over N).
    _dd = _do = _dc = _dv = None
    _dvals = None
    if isinstance(values, dict):
        if "dictionary" in values and "codes" in values:
            _body = values.get("dictionary")
            _dd = getattr(_body, "utf8_data", None)
            if _dd is None:
                _dd = getattr(_body, "data", None)
            _do = getattr(_body, "offsets", None)
            _dc = values.get("codes")
            _dv = values.get("validity")
            _vv = values.get("values")
            if isinstance(_vv, (list, tuple)) and len(_vv) > 0 and all(
                    isinstance(_x, str) for _x in _vv[:8]):
                _dvals = list(_vv)
        else:
            _dd = values.get("utf8_data") or values.get("data")
            _do = values.get("offsets")
            _dc = values.get("codes")
            _dv = values.get("validity")
    else:
        for _dk, _ok in (("utf8_data", "offsets"), ("data", "offsets")):
            _gd = getattr(values, _dk, None)
            _go = getattr(values, _ok, None)
            if _gd is not None and _go is not None:
                _dd, _do = _gd, _go
                break
        _dc = getattr(values, "codes", None)
        _dv = getattr(values, "validity", None)
    if (_dd is not None and _do is not None and _dc is not None
            and isinstance(_dd, (bytes, bytearray))) or (
            _dvals is not None and _dc is not None):
        # DECODE phase: D unique strings from values list (preferred, no
        # re-decode) or utf8 body (D-scale boundary materialization).
        _t0 = _t.perf_counter()
        _ca = np.ascontiguousarray(np.asarray(_dc, dtype=np.int32))
        _n = int(_ca.size)
        if _dvals is not None:
            _ds = _dvals
            _d = len(_ds)
        else:
            _db = bytes(_dd)
            _oa = np.ascontiguousarray(np.asarray(_do, dtype=np.int32))
            _d = int(_oa.size - 1)
            _mv = memoryview(_db)
            _ds = [bytes(_mv[int(_oa[_k]):int(_oa[_k + 1])]).decode("utf-8")
                   for _k in range(_d)]
        _td = (_t.perf_counter() - _t0) * 1000

        # MATCH phase: regex on D unique strings only
        _t0 = _t.perf_counter()
        _rd = [None] * _d
        for _k in range(_d):
            try:
                _rd[_k] = rx.sub(repl, _ds[_k], count=1)
            except _re.error as e:
                raise err(
                    f"CPU driver: text_regex_replace repl failed: {e}",
                    fix="pass a valid replacement string",
                    doc="specs/05-storage-encoding.md",
                ) from None
        _tm = (_t.perf_counter() - _t0) * 1000

        # ALLOC + OUTPUT: broadcast via codes (numpy fancy indexing)
        _t0 = _t.perf_counter()
        _out = np.empty(_n, dtype=object)
        _out[:] = None
        _val = np.zeros(_n, dtype=bool)
        _vm = (_ca >= 0) & (_ca < _d)
        if _dv is not None:
            _vma = np.asarray(_dv, dtype=bool)
            if _vma.size == _n:
                _vm = _vm & _vma
        _ix = np.nonzero(_vm)[0]
        if _ix.size > 0:
            _ra = np.empty(_d, dtype=object)
            for _k in range(_d):
                _ra[_k] = _rd[_k]
            _out[_ix] = _ra[_ca[_ix].astype(np.intp)]
            _val[_ix] = True
        _ta = (_t.perf_counter() - _t0) * 1000
        print(f"  [regex_dict] decode={_td:.3f} match={_tm:.3f} "
              f"alloc={_ta:.3f} total={_tv_ms + _td + _tm + _ta:.3f} "
              f"D={_d} N={_n}")
        return np.ascontiguousarray(_out, dtype=object), np.ascontiguousarray(_val)

    # --- Flat fallback: list comprehension (no per-row index overhead) ---
    if not isinstance(values, (np.ndarray, list, tuple)):
        try:
            values = list(values)
        except TypeError:
            raise err(
                f"CPU driver: TEXT op needs a string column, got {type(values).__name__}",
                fix="pass a numpy <U / list[str|None] column",
                doc="specs/05-storage-encoding.md",
            ) from None
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1:
        raise err(
            f"CPU driver: TEXT op needs a rank-1 column, got shape {arr.shape}",
            fix="pass a flat string column",
            doc="specs/02-semantic-ir.md",
        )
    n = int(arr.size)

    _t0 = _t.perf_counter()
    try:
        raw = [rx.sub(repl, v, count=1) if isinstance(v, str) else None
               for v in arr]
    except _re.error as e:
        raise err(
            f"CPU driver: text_regex_replace repl failed: {e}",
            fix="pass a valid replacement string",
            doc="specs/05-storage-encoding.md",
        ) from None
    _tm = (_t.perf_counter() - _t0) * 1000

    _t0 = _t.perf_counter()
    out = np.empty(n, dtype=object)
    out[:] = raw
    valid = np.fromiter((r is not None for r in raw), dtype=bool, count=n)
    _ta = (_t.perf_counter() - _t0) * 1000
    print(f"  [regex_flat] match={_tm:.3f} alloc={_ta:.3f} "
          f"total={_tv_ms + _tm + _ta:.3f} N={n}")
    return np.ascontiguousarray(out, dtype=object), np.ascontiguousarray(valid)


def _group_index(keys, plan_groupby, naggs=1):
    """Grouping traversal (strategy owned by Planner, executed by groupindex).

    Private helper: returns (gi, decision, gi_ms). gi keeps the frozen shape
    {strategy, ukeys(sorted), starts|inverse, n}; decision carries the
    Planner reason; gi_ms times probe+traversal only (aggregate excluded).
    State is minimal: ukeys + starts|inverse only, never per-row index lists.
    naggs = value aggregates sharing this grouping (generic Planner
    observable for fused-state sizing).
    """
    t0 = time.perf_counter()
    gi, decision = _generic_group_index(keys, plan_groupby, naggs=naggs)
    return gi, decision, (time.perf_counter() - t0) * 1000


def _carry_result(out, bufs, carry, params, to_dict):
    """Primary internal result is ColumnCarry; dict is explicit compat only.

    bufs[out+"#carry"] always holds the carry (observable stages). bufs[out]
    holds the carry when params result='carry', else the explicit dict
    materialization (compat_threads>1 chunks the key range). Default 'dict'
    keeps the frozen output shape; hot path skips dict via result='carry'.
    """
    bufs[out + "#carry"] = carry
    if params.get("result", "dict") == "carry":
        bufs[out] = carry
        return
    try:
        ct = int(params.get("compat_threads", 1) or 1)
    except (TypeError, ValueError):
        ct = 1
    bufs[out] = to_dict(max(1, ct))


def _fused_soa_by_code(k64, vcols, m, ukeys, counts_m):
    """Fused SoA multi-aggregate over resident codes (dense_by_code groups).

    One grouping traversal (counts bincount, computed once in group_index and
    carried in gi -- never recomputed here), one generic aggregate state:
    shared counts (SoA, one array) + one sums array per value column. Layout
    is SoA (separate arrays) by measurement (SoA wins over interleaved AoS on
    the 10M fused matrix), not by legacy. Mean is always derived sum/count,
    never a second pass. k64/counts_m are read-only shared working buffers
    (fresh per grouping, never caller inputs); compaction is forward-only;
    hash/unique paths keep their two-buffer flow (arbitrary permutation is
    never done in place).
    """
    k64 = np.asarray(k64)
    m = int(m)
    pos = np.asarray(ukeys, dtype=np.int64)
    counts = np.asarray(counts_m)[pos]
    sums_list = []
    for v in vcols:
        v = np.asarray(v)
        if np.issubdtype(v.dtype, np.integer):
            w = v.astype(np.int64, copy=False)
            s = np.bincount(k64, weights=w, minlength=m)[pos].astype(np.int64)
        else:
            w = v.astype(np.float64, copy=False)
            s = np.bincount(k64, weights=w, minlength=m)[pos]
        sums_list.append(s)
    return ukeys, counts, sums_list

def _fused_sums_counts(v, gi):
    """One generic state per column: exact int64 sums / float64 sums + counts.

    Single traversal of the value column (reduceat on sorted runs or bincount
    on inverse codes). Mean is always derived sum/count, never a second pass.
    """
    ng = gi["ukeys"].size
    if ng == 0:  # empty input: no groups (sorted-run ends[-1] would crash)
        zd = np.int64 if np.issubdtype(v.dtype, np.integer) else np.float64
        return np.zeros(0, dtype=zd), np.zeros(0, dtype=np.int64)
    if np.issubdtype(v.dtype, np.integer):
        w = v.astype(np.int64, copy=False)
        if gi["inverse"] is None:
            sums = np.add.reduceat(w, gi["starts"])
        else:
            sums = np.bincount(gi["inverse"], weights=w, minlength=ng).astype(np.int64)
    else:
        w = v.astype(np.float64, copy=False)
        if gi["inverse"] is None:
            sums = np.add.reduceat(w, gi["starts"])
        else:
            sums = np.bincount(gi["inverse"], weights=w, minlength=ng)
    if gi["inverse"] is None:
        ends = np.empty(gi["starts"].size, dtype=np.int64)
        ends[:-1] = gi["starts"][1:]
        ends[-1] = gi["n"]
        counts = (ends - gi["starts"]).astype(np.int64)
    else:
        counts = np.bincount(gi["inverse"], minlength=ng).astype(np.int64)
    return sums, counts


def _fused_sums_only(v, gi):
    """Sums lane only; counts are shared (computed once per grouping).

    Same sums contract as _fused_sums_counts (int64 exact / float64,
    reduceat on sorted runs, bincount on inverse codes). gi with
    inverse None requires starts (sorted_run_index shape); fused
    single-pass gi (starts None) is single-column only and never
    reaches here (caller guards).
    """
    v = np.asarray(v)
    ng = gi["ukeys"].size
    if ng == 0:
        zd = np.int64 if np.issubdtype(v.dtype, np.integer) else np.float64
        return np.zeros(0, dtype=zd)
    if np.issubdtype(v.dtype, np.integer):
        w = v.astype(np.int64, copy=False)
        if gi["inverse"] is None:
            return np.add.reduceat(w, gi["starts"])
        return np.bincount(gi["inverse"], weights=w,
                           minlength=ng).astype(np.int64)
    w = v.astype(np.float64, copy=False)
    if gi["inverse"] is None:
        return np.add.reduceat(w, gi["starts"])
    return np.bincount(gi["inverse"], weights=w, minlength=ng)


def _group_counts(gi):
    """Shared group sizes (same math as _fused_sums_counts counts lane)."""
    ng = gi["ukeys"].size
    if ng == 0:
        return np.zeros(0, dtype=np.int64)
    if "k64" in gi and gi.get("k64") is not None:
        pos = np.asarray(gi["ukeys"], dtype=np.int64)
        return np.asarray(gi["counts_m"])[pos].astype(np.int64)
    if gi["inverse"] is None:
        ends = np.empty(gi["starts"].size, dtype=np.int64)
        ends[:-1] = gi["starts"][1:]
        ends[-1] = gi["n"]
        return (ends - gi["starts"]).astype(np.int64)
    return np.bincount(gi["inverse"], minlength=ng).astype(np.int64)


def _lex_vals(dictionary):
    """D-scale dictionary values -> [str] or None (local copy, no imports)."""
    try:
        if isinstance(dictionary, dict):
            if isinstance(dictionary.get("values"), (list, tuple)):
                return list(dictionary["values"])
            for _dk, _ok in (("utf8_data", "offsets"), ("data", "offsets")):
                if _dk in dictionary and _ok in dictionary:
                    o = np.ascontiguousarray(
                        np.asarray(dictionary[_ok], dtype=np.int32))
                    mv = memoryview(bytes(dictionary[_dk]))
                    return [bytes(mv[int(a):int(b)]).decode("utf-8")
                            for a, b in zip(o[:-1].tolist(), o[1:].tolist())]
            return None
        if isinstance(dictionary, (list, tuple)):
            return list(dictionary)
        if hasattr(dictionary, "to_list"):
            return list(dictionary.to_list())
    except (UnicodeDecodeError, ValueError, TypeError, IndexError):
        return None
    return None


def _group_minmax(v, gi, kind, dictionary=None):
    """Single-pass O(N) grouped min/max via traversal codes (C-speed).

    Reuses the grouping traversal (gi from _group_index/_routed_aggregate):
    sorted runs -> minimum/maximum.reduceat; inverse codes -> minimum.at /
    maximum.at scatter; dense_by_code (k64/m) -> dense scatter + compact.
    Validity already filtered by the caller (keep/compact); empty -> empty
    (same dtype). Dtype preserved (int32/int64/float32/float64); NaN is a
    value (minimum/maximum propagate, same as global a.min()/a.max()).
    String path (TEXT codes or str arrays): lexical rank, not raw codes --
    dictionary lexical order (argsort once, D-scale) maps codes to ranks,
    min/max runs over ranks, then maps back to the original code space;
    str arrays use their sorted-unique rank the same way. Numeric path
    below is verbatim (dictionary None + numeric dtype).
    """
    v = np.asarray(v)
    ng = gi["ukeys"].size
    if ng == 0:
        return np.zeros(0, dtype=v.dtype)
    if kind not in ("min", "max"):
        raise ValueError(f"unknown minmax kind '{kind}'")
    is_min = kind == "min"
    # String values (str arrays): lexical rank path (Q22 correctness).
    if v.dtype.kind in "USO":
        try:
            if v.size == 0:
                return np.ascontiguousarray(v[:0])
            uniq, rank = np.unique(v, return_inverse=True)
            rank = np.ascontiguousarray(rank.astype(np.int64, copy=False))
        except (TypeError, ValueError):
            rank = None
        if rank is not None:
            if "k64" in gi and gi.get("k64") is not None:
                k64 = np.asarray(gi["k64"])
                m = int(gi["m"])
                pos = np.asarray(gi["ukeys"], dtype=np.int64)
                fill = int(rank.max()) + 1 if is_min else -1
                dense = np.full(m, fill, dtype=np.int64)
                if is_min:
                    np.minimum.at(dense, k64, rank)
                else:
                    np.maximum.at(dense, k64, rank)
                return np.ascontiguousarray(uniq[dense[pos]])
            if gi["inverse"] is None:
                w = np.ascontiguousarray(rank)
                rr = (np.minimum.reduceat(w, gi["starts"]) if is_min
                      else np.maximum.reduceat(w, gi["starts"]))
                return np.ascontiguousarray(uniq[rr])
            inv = np.asarray(gi["inverse"])
            fill = int(rank.max()) + 1 if is_min else -1
            out = np.full(ng, fill, dtype=np.int64)
            if is_min:
                np.minimum.at(out, inv, rank)
            else:
                np.maximum.at(out, inv, rank)
            return np.ascontiguousarray(uniq[out])
    # Integer codes with an unsorted dictionary carrier: lexical rank over
    # the dictionary order (Q22 raw-code vs lexical fix); codes returned in
    # the original code space so decode stays valid. No dictionary (None)
    # -> legacy numeric path verbatim below.
    if dictionary is not None and v.dtype.kind in "iu":
        vals = _lex_vals(dictionary)
        if vals:
            try:
                order = np.argsort(np.array(vals, dtype=str),
                                   kind="stable").astype(np.int64)
                rank_of = np.empty(len(vals), dtype=np.int64)
                rank_of[order] = np.arange(len(vals), dtype=np.int64)
                if int(v.min()) >= 0 and int(v.max()) < len(vals):
                    rank = np.ascontiguousarray(rank_of[np.asarray(v)])
                    if "k64" in gi and gi.get("k64") is not None:
                        k64 = np.asarray(gi["k64"])
                        m = int(gi["m"])
                        pos = np.asarray(gi["ukeys"], dtype=np.int64)
                        fill = int(rank.max()) + 1 if is_min else -1
                        dense = np.full(m, fill, dtype=np.int64)
                        if is_min:
                            np.minimum.at(dense, k64, rank)
                        else:
                            np.maximum.at(dense, k64, rank)
                        back = np.ascontiguousarray(order[dense[pos]])
                        return np.ascontiguousarray(back.astype(v.dtype,
                                                               copy=False))
                    if gi["inverse"] is None:
                        w = np.ascontiguousarray(rank)
                        rr = (np.minimum.reduceat(w, gi["starts"]) if is_min
                              else np.maximum.reduceat(w, gi["starts"]))
                        return np.ascontiguousarray(
                            order[rr].astype(v.dtype, copy=False))
                    inv = np.asarray(gi["inverse"])
                    fill = int(rank.max()) + 1 if is_min else -1
                    out = np.full(ng, fill, dtype=np.int64)
                    if is_min:
                        np.minimum.at(out, inv, rank)
                    else:
                        np.maximum.at(out, inv, rank)
                    return np.ascontiguousarray(
                        order[out].astype(v.dtype, copy=False))
            except (TypeError, ValueError, IndexError):
                pass
    if "k64" in gi and gi.get("k64") is not None:
        k64 = np.asarray(gi["k64"])
        m = int(gi["m"])
        pos = np.asarray(gi["ukeys"], dtype=np.int64)
        if v.dtype.kind in "iu":
            info = np.iinfo(v.dtype)
            fill = info.max if is_min else info.min
        else:
            fill = np.inf if is_min else -np.inf
        dense = np.full(m, fill, dtype=v.dtype)
        if is_min:
            np.minimum.at(dense, k64, v)
        else:
            np.maximum.at(dense, k64, v)
        return np.ascontiguousarray(dense[pos])
    if gi["inverse"] is None:
        w = np.ascontiguousarray(v)
        return np.ascontiguousarray(
            np.minimum.reduceat(w, gi["starts"]) if is_min
            else np.maximum.reduceat(w, gi["starts"]))
    inv = np.asarray(gi["inverse"])
    if v.dtype.kind in "iu":
        info = np.iinfo(v.dtype)
        fill = info.max if is_min else info.min
    else:
        fill = np.inf if is_min else -np.inf
    out = np.full(ng, fill, dtype=v.dtype)
    if is_min:
        np.minimum.at(out, inv, v)
    else:
        np.maximum.at(out, inv, v)
    return np.ascontiguousarray(out)


def _tuple_sums(v, inverse, ng):
    """Sums lane over composite inverse codes (Q36 tuple path).

    Integer columns accumulate in true int64 (exact); float columns in
    float64 (same as bincount weights path). Vectorized single pass.
    """
    v = np.asarray(v)
    ng = int(ng)
    if ng == 0:
        zd = np.int64 if np.issubdtype(v.dtype, np.integer) else np.float64
        return np.zeros(0, dtype=zd)
    if np.issubdtype(v.dtype, np.integer):
        w = v.astype(np.int64, copy=False)
        return np.bincount(np.asarray(inverse), weights=w,
                           minlength=ng).astype(np.int64)
    w = v.astype(np.float64, copy=False)
    return np.bincount(np.asarray(inverse), weights=w, minlength=ng)


def _where_ref(mask, true_v, false_v, mask_valid=None):
    """Fused row-wise select (CASE): where(mask, true, false), C-speed.

    mask: BoolMask (BitPack or bool array); mask_valid: sidecar (3VL:
    invalid mask rows select the false branch, never an error). true_v /
    false_v: equal-length value columns (any numeric/bool dtype; promotion
    follows np.where). Empty / all-true / all-false valid. Returns
    (out_values, out_valid_or_None): validity = chosen-branch validity
    per row (None iff all chosen rows valid).
    """
    m = _to_bool_array(mask)
    n = m.size
    if np.asarray(true_v).size != n or np.asarray(false_v).size != n:
        raise ValueError(
            f"where values/mask size mismatch {np.asarray(true_v).size}/"
            f"{np.asarray(false_v).size} != {n}")
    eff = m if mask_valid is None else m & np.asarray(mask_valid, dtype=bool)
    eff = np.ascontiguousarray(eff, dtype=bool)
    t = true_v.to_array() if isinstance(true_v, _BitPack) else np.asarray(true_v)
    f = false_v.to_array() if isinstance(false_v, _BitPack) else np.asarray(false_v)
    out = np.where(eff, t, f)
    return np.ascontiguousarray(out), eff


def _count_distinct_sorted_dedup(kc, vc, err):
    """COUNT DISTINCT per group via sorted_dedup (vector-only, no row loops).

    kc/vc: compact int arrays (validity already filtered by caller).
    1. _sort_perm over (keys, values) -> rows ordered by group, dedup adjacent.
    2. One vector pass: group boundaries (sk change flags) + distinct
       transitions (value change within group) -> nunique per group.
    Returns (ukeys sorted int64, nunique int64, sort_ms, scan_ms).
    Empty -> empty (no-group edge, same as groupby).
    """
    kc = np.ascontiguousarray(np.asarray(kc)).ravel()
    vc = np.ascontiguousarray(np.asarray(vc)).ravel()
    n = int(kc.size)
    if n == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                0.0, 0.0)
    t0 = time.perf_counter()
    try:
        # Fast lane: all-ascending, no validity mask (this call site always
        # passes descending=[False, False], mask None after validity
        # compaction). Single lexsort (primary kc, secondary vc) replaces
        # two successive stable argsorts (~2x on 1M). Tie order among
        # identical (k, v) rows may differ from _sort_perm, but the scan
        # below depends only on the (sk, sv) sequences, which are
        # identical either way. Any failure -> proven _sort_perm owns it.
        perm = np.lexsort((
            np.ascontiguousarray(vc).ravel(),
            np.ascontiguousarray(kc).ravel()))
        perm = np.ascontiguousarray(perm)
    except (TypeError, ValueError, MemoryError):
        perm = _sort_perm([kc, vc], [False, False], None, err)
    sort_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    sk = kc[np.asarray(perm)]
    sv = vc[np.asarray(perm)]
    new_group = sk[1:] != sk[:-1]
    ev = np.empty(n, dtype=bool)
    ev[0] = True
    if n > 1:
        ev[1:] = new_group | (sv[1:] != sv[:-1])
    # Run-length over the distinct stream (generic: same nunique, fewer
    # passes). ev rows carry sorted group labels, so per-group distinct
    # counts are run-lengths of sk[ev]: no N-sized gid cumsum and no
    # weighted N-bincount (both ~10ms on 1M). ukeys = run heads.
    sk_ev = sk[ev]
    nev = int(sk_ev.size)
    chg = sk_ev[1:] != sk_ev[:-1] if nev > 1 else np.zeros(0, dtype=bool)
    gid_ev = np.empty(nev, dtype=np.int64)
    gid_ev[0] = 0
    if nev > 1:
        gid_ev[1:] = np.cumsum(chg, dtype=np.int64)
    ng = int(gid_ev[-1]) + 1
    starts_ev = np.empty(ng, dtype=np.int64)
    starts_ev[0] = 0
    if ng > 1:
        starts_ev[1:] = np.flatnonzero(chg).astype(np.int64) + 1
    ukeys = np.ascontiguousarray(sk_ev[starts_ev].astype(np.int64, copy=False))
    nunique = np.ascontiguousarray(
        np.bincount(gid_ev, minlength=ng).astype(np.int64))
    scan_ms = (time.perf_counter() - t1) * 1000
    return ukeys, nunique, sort_ms, scan_ms


def _unique_sorted(keys):
    """Sorted unique ascending (np.unique parity) -> int array.

    Radix lane (LSD radix via stable argsort, O(N)) with exact
    fallback: 2.5x @1M / 3.9x @10M int64 (ClickBench UserID,
    array_equal exact), generic over int dtypes and N (no thresholds,
    no fixed widths). Stable argsort + adjacent-dedup yields the same
    ascending set as np.unique. Any failure -> proven np.unique owns it.
    """
    n = int(np.asarray(keys).size)
    if n == 0:
        return np.unique(keys)
    try:
        perm = np.argsort(keys, kind="stable")
        s = np.asarray(keys)[np.asarray(perm)]
        m = np.empty(n, dtype=bool)
        m[0] = True
        if n > 1:
            m[1:] = s[1:] != s[:-1]
        return np.ascontiguousarray(s[m])
    except (TypeError, ValueError, MemoryError):
        return np.unique(keys)


def cpu_execute_impl(nodes, accum_dtype, canonical_dtype=None, format_error=None,
                     plan_groupby=None, plan_pack=None, check_int32_range=None,
                     grouped_distinct_hash=None, GroupedHashMiss=None):
    """execute(packets) -> {out: buffer}; packets are Planner nodes (single chunk)."""
    if canonical_dtype is None:
        # Standalone fallback (same shape as Core table); injected Core wins.
        def canonical_dtype(name):
            table = {"int32": "int32", "float32": "float32", "f32": "float32",
                      "float64": "float64", "f64": "float64"}
            if name == "bool":
                return {"logical": "bool", "width": 1}
            if isinstance(name, str) and name.startswith("enum:"):
                try:
                    w = int(name.split(":")[1])
                except (IndexError, ValueError):
                    w = -1
                if w in (1, 2, 4, 8):
                    return {"logical": name, "width": w}
            if name not in table:
                raise ValueError(f"unknown dtype '{name}'")
            return {"logical": table[name]}
    err = format_error or _fallback_err

    def as_dtype(name):
        return _as_dtype(name, canonical_dtype)

    # P4: producer-consumer map for fused pack+groupby (generic IR shapes,
    # never query names; Planner untouched). _uses counts readers of each
    # buffer; _consumer lists them. _fused_done marks groupby outs already
    # computed at their pack node; _fused_pack_outs marks pack outs with
    # no materialized packed-key buffer (single consumer fused away).
    _consumer = {}
    _uses = {}
    for _q in nodes:
        if not isinstance(_q, dict):
            continue
        for _i in (_q.get("inputs") or []):
            _uses[_i] = _uses.get(_i, 0) + 1
            _consumer.setdefault(_i, []).append(_q)
    _fused_done = set()
    _fused_pack_outs = set()
    # P5: fuse map→reduce pairs (single-consumer chain; Planner untouched).
    # When a map node's output is consumed by exactly one reduce node, the
    # intermediate buffer allocation is avoided by inlining the map into the
    # reduce handler.  Safe for any map fn × any reduce op (validity
    # propagated from map's inputs; dtype follows the original map output).
    _fused_mr_skip = set()          # map outs to skip in main loop
    _fused_mr = {}                  # reduce_out → {fn, value, map_inputs}
    for _p in nodes:
        if not isinstance(_p, dict):
            continue
        if _p.get("kernel_id", _p.get("op")) != "reduce":
            continue
        _inp = (_p.get("inputs") or [None])[0]
        if _inp is None:
            continue
        _prods = _consumer.get(_inp, [])
        if len(_prods) != 1:
            continue
        _mapn = _prods[0]
        if _mapn.get("kernel_id", _mapn.get("op")) != "map":
            continue
        # Fuse: skip map node, annotate reduce with map params.
        _fused_mr_skip.add(_mapn["out"])
        _fused_mr[_p["out"]] = {
            "fn": _mapn["params"]["fn"],
            "value": _mapn["params"].get("value"),
            "map_inputs": list(_mapn["inputs"]),
        }
    bufs = {}
    for p in nodes:
        op = p.get("kernel_id", p.get("op"))
        params = p["params"]
        # P5: skip map nodes fused into a downstream reduce.
        if op == "map" and p["out"] in _fused_mr_skip:
            continue
        if op == "series":
            dtype_name = params.get("dtype", "int32")
            if isinstance(dtype_name, str) and dtype_name.startswith("enum:"):
                # Packed-enum names validate here: bad widths raise the
                # width-specific contract (never the generic numeric error).
                width = canonical_dtype(dtype_name).get("width")
                bufs[p["out"]] = _series_packed(
                    params["values"], width, err, f"series '{p['out']}'")
            else:
                width = _packed_width(dtype_name, canonical_dtype)
                if width is not None:
                    bufs[p["out"]] = _series_packed(
                        params["values"], width, err, f"series '{p['out']}'")
                else:
                    want = as_dtype(dtype_name)
                    arr = _as_column(params["values"], want, err,
                                     f"series '{p['out']}'", check_int32_range)
                    bufs[p["out"]] = arr
            if params.get("validity") is not None:
                mask = _as_column(params["validity"], np.dtype(bool), err,
                                  f"series '{p['out']}' validity")
                if mask.size != bufs[p["out"]].size:
                    raise err(
                        f"CPU driver: series validity size {mask.size} != values {bufs[p['out']].size}",
                        fix="pass validity matching values length",
                        doc="specs/delta-3-null-contract.md",
                    )
                bufs[p["out"] + "#validity"] = mask
        elif op == "encode_pattern":
            codes, valid, width = _encode_pattern_vec(
                params["values"], params["prefix"], err)
            bufs[p["out"]] = codes
            bufs[p["out"] + "#validity"] = valid
            bufs[p["out"] + "#pattern"] = {"prefix": params["prefix"], "width": width}
        elif op == "text_length":
            lens, valid = _text_length_vec(params["values"], err)
            bufs[p["out"]] = lens
            bufs[p["out"] + "#validity"] = valid
        elif op == "text_contains":
            hit, valid = _text_contains_vec(
                params["values"], params["substr"], err)
            bufs[p["out"]] = hit
            bufs[p["out"] + "#validity"] = valid
        elif op == "text_startswith":
            hit, valid = _text_affix_vec(
                params["values"], params["prefix"], "startswith", err)
            bufs[p["out"]] = hit
            bufs[p["out"] + "#validity"] = valid
        elif op == "text_endswith":
            hit, valid = _text_affix_vec(
                params["values"], params["suffix"], "endswith", err)
            bufs[p["out"]] = hit
            bufs[p["out"] + "#validity"] = valid
        elif op == "text_equals":
            hit, valid = _text_affix_vec(
                params["values"], params["key"], "equals", err)
            bufs[p["out"]] = hit
            bufs[p["out"] + "#validity"] = valid
        elif op == "text_regex_replace":
            out, valid = _text_regex_replace_vec(
                params["values"], params["pattern"], params["repl"], err)
            bufs[p["out"]] = out
            bufs[p["out"] + "#validity"] = valid
        elif op == "pack_keys":
            ins = [bufs[i] for i in p["inputs"]]
            for arr in ins:
                if not np.issubdtype(arr.dtype, np.integer):
                    raise err(
                        "CPU driver: pack_keys needs int32 code series",
                        fix="encode categoricals to int32 codes first (encode_pattern)",
                        doc="specs/delta-2-composite-keys.md",
                    )
            mode = params.get("mode", "pack")
            if mode == "pack":
                if len(ins) == 1:
                    bufs[p["out"]] = ins[0].astype(np.int64, copy=False)
                else:
                    # In-place composite (E-track: 2 allocs + 5 passes instead
                    # of 3 allocs + temp chains; measured 1.8x on 10M pack).
                    # astype() defaults copy=True: fresh buffers, inputs never
                    # mutated (copy=False would alias int64 inputs -> no).
                    # The 0xFFFFFFFF mask is semantics (negative int32 keys
                    # stay bijective); dropping it collides hi bits.
                    comp = ins[0].astype(np.int64)
                    np.left_shift(comp, np.int64(32), out=comp)
                    lo = ins[1].astype(np.int64)
                    np.bitwise_and(lo, np.int64(0xFFFFFFFF), out=lo)
                    np.bitwise_or(comp, lo, out=comp)
                    bufs[p["out"]] = comp
            elif mode == "radix":
                auto = params.get("radix") is None
                rad = params.get("radix")
                _prod2 = (len(ins) == 2 and ins[0].dtype == np.int32
                          and ins[1].dtype == np.int32 and ins[0].size)
                if rad is None and not (auto and _prod2):
                    rad = [int(a.max()) + 1 for a in ins]
                if rad is not None and len(rad) != len(ins):
                    raise err(
                        f"CPU driver: pack_keys radix len {len(rad)} != inputs {len(ins)}",
                        fix="pass radix per input or radix=None for auto",
                        doc="specs/delta-2-composite-keys.md",
                    )
                comp = None
                if _prod2:
                    # Production int32-direct path: Planner-gated on generic
                    # observables (column minima/maxima + M2, no dataset
                    # branches); legacy int64 radix stays the fallback.
                    # Minima+maxima come from one concurrent pass (same
                    # numbers as the sequential scans); auto radix reuses
                    # the maxima, explicit radix keeps its own (maxima only
                    # re-probed for the gate, same values as before).
                    k1, k2 = ins[0], ins[1]
                    _tpack = _mt_threads()
                    if int(_tpack) > 1:
                        (_kmin1, _kmax1), (_kmin2, _kmax2) = [
                            b for b in _pool(int(_tpack)).map(
                                _mm_one, (k1, k2))]
                    else:
                        _kmin1, _kmax1 = _mm_one(k1)
                        _kmin2, _kmax2 = _mm_one(k2)
                    if auto:
                        rad = [_kmax1 + 1, _kmax2 + 1]
                    m2 = int(rad[1])
                    kmax1 = int(rad[0]) - 1 if auto else _kmax1
                    kmax2 = int(rad[1]) - 1 if auto else _kmax2
                    kmin1, kmin2 = _kmin1, _kmin2
                    if plan_pack is not None:
                        try:
                            dec = plan_pack(kmin1, kmax1, kmin2, kmax2, m2)
                        except TypeError:
                            dec = {"strategy": "radix",
                                   "reason": "plan_pack gate unavailable: "
                                             "legacy int64 radix"}
                    else:
                        dec = {"strategy": "radix",
                               "reason": "no Planner gate: legacy int64 radix"}
                    bufs[p["out"] + "#pack"] = {
                        "strategy": dec.get("strategy", "radix"),
                        "reason": dec.get("reason", "")}
                    if dec.get("strategy") == "int32_direct":
                        # P4: fused pack+groupby (single-consumer groupby
                        # sum/count/mean, generic guards inside; any doubt
                        # -> proven pack below, groupby branch verbatim).
                        _gnode = None
                        if _uses.get(p["out"], 0) == 1:
                            _cands = _consumer.get(p["out"], [])
                            if (len(_cands) == 1 and _cands[0].get(
                                    "kernel_id", _cands[0].get("op")) == "groupby"):
                                _gnode = _cands[0]
                        _fz = (_fused_pack_consume(
                            p, _gnode, k1, k2, m2, kmin1, kmax1, kmin2,
                            kmax2, bufs, plan_groupby)
                            if _gnode is not None else None)
                        if _fz is not None:
                            _uk, _cc, _ss, _gdec, _gi_ms, _ams, _cms, \
                                _mms, _tn = _fz
                            _gop = _gnode["params"].get("op", "sum")
                            if np.issubdtype(np.asarray(_ss).dtype,
                                             np.integer):
                                _ss = np.ascontiguousarray(_ss).astype(
                                    np.int64, copy=False)
                            _carry = _ColumnCarry(_uk, _cc, {"v": _ss})
                            _carry_result(
                                _gnode["out"], bufs, _carry,
                                _gnode["params"],
                                lambda ct, _gop=_gop: _carry.to_dict_flat(
                                    _gop, threads=ct))
                            bufs[_gnode["out"] + "#strategy"] = "dense_by_code"
                            bufs[_gnode["out"] + "#groupindex"] = {
                                "strategy": "dense_by_code",
                                "reason": _gdec.get("reason", "") +
                                f"; backend=native_{'mt%d' % _tn if _tn > 1 else 'st'} "
                                "fused-pack carry->dict explicit",
                                "gi_ms": _gi_ms, "agg_ms": _ams,
                                "carry_ms": _cms, "merge_ms": _mms,
                                "threads": _tn, "backend": "native_fused"}
                            bufs[p["out"] + "#pack"] = {
                                "strategy": "int32_direct",
                                "reason": dec.get("reason", "") +
                                "; fused pack+aggregate, no packed buffer"}
                            _fused_done.add(_gnode["out"])
                            _fused_pack_outs.add(p["out"])
                        else:
                            comp = _pack_i32_direct(k1, k2, m2, kmax1, kmax2,
                                                   err)
                if comp is None and p["out"] not in _fused_pack_outs:
                    comp = _pack_radix_i64(ins, rad, err)
                if p["out"] not in _fused_pack_outs:
                    bufs[p["out"]] = comp
            elif mode == "hash":
                # Generic exact path: tuple sidecar first (composite
                # groupby/groupby_multi prefer it; never an int64 scalar
                # pack as the sole grouping path). The int64 hash below
                # is a measurement baseline only: exact per-row Python
                # loop on small N, skipped on wide N where the O(N)
                # Python overhead dominates (sidecar stays exact).
                bufs[p["out"] + "#tuple_cols"] = [np.asarray(a) for a in ins]
                n = int(ins[0].size)
                if n > 65536:
                    bufs[p["out"]] = np.zeros(n, dtype=np.int64)
                else:
                    hs = np.empty(n, dtype=np.int64)
                    cols = [np.asarray(a).tolist() for a in ins]
                    for i, tup in enumerate(zip(*cols)):
                        hs[i] = hash(tup) & 0x7FFFFFFFFFFFFFFF
                    bufs[p["out"]] = hs
            else:
                raise err(
                    f"CPU driver: unknown pack_keys mode '{mode}'",
                    fix="use one of pack/radix/hash",
                    doc="specs/delta-2-composite-keys.md",
                )
            m = _valid_mask(bufs, *p["inputs"])
            if m is not None:
                bufs[p["out"] + "#validity"] = m
        elif op == "map":
            a = bufs[p["inputs"][0]]
            fn = params["fn"]
            b = bufs[p["inputs"][1]] if len(p["inputs"]) > 1 else params.get("value")
            if fn == "pow" and len(p["inputs"]) > 1:
                raise err(
                    "CPU driver: map pow with array exponent: '**' scalar-exp only (spec 01)",
                    fix="pass a scalar exponent",
                    doc="specs/01-public-api.md",
                )
            if fn == "add":
                r = a + b
            elif fn == "sub":
                r = a - b
            elif fn == "mul":
                r = a * b
            elif fn == "div":
                # M7b: above the gate the fused native lane collapses the
                # 4-pass astype(f64)/divide/rint/astype chain into one pass.
                # EXACT parity with the NumPy chain below (rint-then-cast for
                # integer lanes included). Below the gate, and for every other
                # shape the wrapper does not lane natively, NumPy owns it.
                # The scalar form (1 input) stays NumPy: measured 0.12-1.06x.
                if (len(p["inputs"]) > 1 and a.size >= _MAP_DIV_NATIVE_MIN
                        and isinstance(a, np.ndarray)
                        and isinstance(b, np.ndarray)
                        and a.dtype == b.dtype):
                    r = _native_map_div_lane(a, "div", b)
                else:
                    r = None
                if r is None:
                    r = (a.astype(np.float64) / b)
                    if np.issubdtype(a.dtype, np.integer):
                        r = np.rint(r).astype(a.dtype)
            elif fn == "pow":
                r = (a.astype(np.float64) ** b)
                if np.issubdtype(a.dtype, np.integer):
                    r = np.rint(r).astype(a.dtype)
            elif fn == "floor_div":
                # Python floor semantics (np.floor_divide, exact for ints).
                # WGSL-port note: WGSL `/` truncates toward zero, so a GPU
                # port needs a sign-correction prelude for negative inputs
                # (non-negative domains — dates/counts/codes — map 1:1).
                # SPEC 06: Map/MapBinary GPU ✅, elementwise, chunkable=true.
                r = np.floor_divide(a, b)
            elif fn == "mod":
                # Python % semantics (sign of divisor, np.remainder).
                # Same WGSL-port note as floor_div (truncated `%` on GPU).
                r = np.remainder(a, b)
            else:
                raise err(
                    f"CPU driver: unknown map fn '{fn}'",
                    fix="use one of add/sub/mul/div/pow/floor_div/mod",
                    doc="specs/02-semantic-ir.md",
                )
            bufs[p["out"]] = r.astype(a.dtype, copy=False) if fn in ("add", "sub", "mul") else r
            m = _valid_mask(bufs, *p["inputs"])
            if m is not None:
                bufs[p["out"] + "#validity"] = m
        elif op == "compare":
            a = bufs[p["inputs"][0]]
            b = bufs[p["inputs"][1]] if len(p["inputs"]) > 1 else params.get("value")
            cop = params["op"]
            if cop == "==":
                r = (a == b)
            elif cop == "!=":
                r = (a != b)
            elif cop == "<":
                r = (a < b)
            elif cop == "<=":
                r = (a <= b)
            elif cop == ">":
                r = (a > b)
            elif cop == ">=":
                r = (a >= b)
            else:
                raise err(
                    f"CPU driver: unknown compare op '{cop}'",
                    fix="use one of ==/!=/</<=/>/>=",
                    doc="specs/02-semantic-ir.md",
                )
            bufs[p["out"]] = _BitPack.from_bool(
                np.ascontiguousarray(np.asarray(r, dtype=bool)))
            m = _valid_mask(bufs, *p["inputs"])
            if m is not None:
                bufs[p["out"] + "#validity"] = m
        elif op == "filter":
            a = bufs[p["inputs"][0]]
            mk = bufs[p["inputs"][1]]
            mk_valid = bufs.get(p["inputs"][1] + "#validity")
            try:
                r, eff = _filter_ref(a, mk, mk_valid)
            except (TypeError, ValueError) as e:
                raise err(
                    f"CPU driver: filter {e}",
                    fix="pass a BoolMask (compare output) matching values length",
                    doc="specs/02-semantic-ir.md",
                ) from None
            native_r = _native_filter(a, eff)
            if native_r is not None:
                r = native_r
            bufs[p["out"]] = r
            v_valid = bufs.get(p["inputs"][0] + "#validity")
            if v_valid is not None:
                bufs[p["out"] + "#validity"] = np.asarray(v_valid)[eff]
        elif op == "mask":
            mop = params["op"]
            if mop not in ("and", "or", "not"):
                raise err(
                    f"CPU driver: unknown mask op '{mop}'",
                    fix="use one of and/or/not",
                    doc="specs/02-semantic-ir.md",
                )
            a = bufs[p["inputs"][0]]
            b = bufs[p["inputs"][1]] if len(p["inputs"]) > 1 else None
            try:
                r = _bit_combine(a, b, mop)
            except (TypeError, ValueError) as e:
                raise err(
                    f"CPU driver: mask {e}",
                    fix="pass BoolMask inputs (compare outputs) of equal length",
                    doc="specs/02-semantic-ir.md",
                ) from None
            bufs[p["out"]] = r
            m = _valid_mask(bufs, *p["inputs"])
            if m is not None:
                bufs[p["out"] + "#validity"] = m
        elif op == "where":
            if len(p["inputs"]) != 3:
                raise err(
                    f"CPU driver: where needs [mask, true, false], got {p['inputs']!r}",
                    fix="pass ir_where(out, 'm', 't', 'f')",
                    doc="specs/02-semantic-ir.md",
                )
            mk, tv, fv = (bufs[p["inputs"][0]], bufs[p["inputs"][1]],
                          bufs[p["inputs"][2]])
            mk_valid = bufs.get(p["inputs"][0] + "#validity")
            try:
                r, eff = _where_ref(mk, tv, fv, mk_valid)
            except (TypeError, ValueError) as e:
                raise err(
                    f"CPU driver: where {e}",
                    fix="pass a BoolMask + equal-length true/false columns",
                    doc="specs/02-semantic-ir.md",
                ) from None
            bufs[p["out"]] = r
            t_valid = bufs.get(p["inputs"][1] + "#validity")
            f_valid = bufs.get(p["inputs"][2] + "#validity")
            if t_valid is not None or f_valid is not None:
                n = int(r.size)
                ta = (np.ones(n, dtype=bool) if t_valid is None
                      else np.asarray(t_valid, dtype=bool))
                fa = (np.ones(n, dtype=bool) if f_valid is None
                      else np.asarray(f_valid, dtype=bool))
                ov = np.where(eff, ta, fa)
                if not bool(np.all(ov)):
                    bufs[p["out"] + "#validity"] = np.ascontiguousarray(ov)
        elif op == "gather":
            a = bufs[p["inputs"][0]]
            idx = bufs[p["inputs"][1]]
            bufs[p["out"]] = _gather_ref(a, idx, err)
            v_valid = bufs.get(p["inputs"][0] + "#validity")
            if v_valid is not None:
                bufs[p["out"] + "#validity"] = np.asarray(v_valid)[np.asarray(idx)]
        elif op == "sort":
            cols = [_sort_key_array(bufs[i], err, f"sort key '{i}'")
                    for i in p["inputs"]]
            desc = params.get("descending", [False] * len(cols))
            if len(desc) != len(cols):
                raise err(
                    f"CPU driver: sort descending len {len(desc)} != keys {len(cols)}",
                    fix="pass one bool per sort key",
                    doc="specs/02-semantic-ir.md",
                )
            bufs[p["out"]] = _sort_perm(cols, [bool(d) for d in desc],
                                        _valid_mask(bufs, *p["inputs"]), err)
        elif op == "slice":
            a = bufs[p["inputs"][0]]
            bufs[p["out"]] = _slice_take(a, params.get("limit"),
                                         params.get("offset", 0), err)
            v_valid = bufs.get(p["inputs"][0] + "#validity")
            if v_valid is not None:
                n = (v_valid.n if isinstance(v_valid, _BitPack)
                     else np.asarray(v_valid).size)
                off = int(params.get("offset", 0))
                if off < n:
                    lim = params.get("limit")
                    end = n if lim is None else min(n, off + int(lim))
                    bufs[p["out"] + "#validity"] = np.asarray(v_valid)[off:end]
                else:
                    bufs[p["out"] + "#validity"] = np.zeros(0, dtype=bool)
        elif op == "rolling_sum":
            a = bufs[p["inputs"][0]].astype(np.float64)
            window, mp = params["window"], params["min_periods"]
            n = a.size
            out = np.full(n, np.nan)
            if n:
                m = _valid_mask(bufs, p["inputs"][0])
                if m is not None:
                    a = np.where(np.asarray(m, dtype=bool), a, np.nan)
                # M7a: vectorised. sliding_window_view is a stride view (no
                # n*window materialisation) and .sum(axis=1) runs the SAME
                # pairwise reduction per row as the old per-window w.sum(), so
                # the value is bit-identical and ANY-NaN still poisons.
                # Head region i < k-1 (k = min(window, n)) is a growing prefix,
                # so it stays a short scalar loop of at most window-1 steps.
                k = min(window, n)
                for i in range(k - 1):
                    if i + 1 >= mp:
                        w = a[:i + 1]
                        if not np.isnan(w).any():
                            out[i] = w.sum()
                if k >= mp:
                    v = np.lib.stride_tricks.sliding_window_view(a, k)
                    out[k - 1:] = v.sum(axis=1)
            src = bufs[p["inputs"][0]]
            if np.issubdtype(src.dtype, np.integer) and not np.isnan(out).any():
                out = np.rint(out).astype(src.dtype)
            bufs[p["out"]] = out
        elif op == "shift":
            a = bufs[p["inputs"][0]]
            periods = params.get("periods", 0)
            if isinstance(periods, bool) or not isinstance(periods, int) \
                    or periods < 0:
                raise err(
                    f"CPU driver: shift needs periods>=0 int, got {periods!r}",
                    fix="pass periods>=0 (negative is explicit v1 error)",
                    doc="specs/02-semantic-ir.md",
                )
            v_valid = bufs.get(p["inputs"][0] + "#validity")
            r, rv = _shift_ref(a, v_valid, int(periods), err)
            bufs[p["out"]] = r
            if rv is not None:
                bufs[p["out"] + "#validity"] = rv
        elif op == "cumsum":
            if params:
                raise err(
                    f"CPU driver: cumsum takes no params, got {params!r}",
                    fix="pass ir_cumsum(out, inp) (always inclusive, v1)",
                    doc="specs/02-semantic-ir.md",
                )
            a = bufs[p["inputs"][0]]
            v_valid = bufs.get(p["inputs"][0] + "#validity")
            r, rv = _cumsum_ref(a, v_valid, err)
            bufs[p["out"]] = r
            if rv is not None:
                bufs[p["out"] + "#validity"] = rv
        elif op == "groupby":
            if p["out"] in _fused_done:
                continue  # P4: already computed at the pack node (fused)
            v = bufs[p["inputs"][0]]
            k = bufs[p["inputs"][1]]
            if v.size != k.size:
                raise err(
                    f"CPU driver: groupby values/keys size mismatch {v.size} != {k.size}",
                    fix="pass equal-length values and keys",
                    doc="specs/02-semantic-ir.md",
                )
            _tcols = bufs.get(p["inputs"][1] + "#tuple_cols")
            if _tcols is not None:
                # Generic tuple/composite path (Q36 4-col, CPU-only):
                # validity AND over values + packed keys (pack validity
                # already ANDs the 4 columns: NULL excluded), lex-sort
                # structured unique with sorted tuple ukeys, int64
                # sums/count. Never an int64 scalar pack as sole path.
                gop = params["op"]
                if gop not in ("sum", "count", "mean"):
                    raise err(
                        f"CPU driver: composite groupby op '{gop}' needs single-col keys",
                        fix="use sum/count/mean on tuple keys (min/max stay single-col)",
                        doc="specs/delta-2-composite-keys.md",
                    )
                t0 = time.perf_counter()
                m = _valid_mask(bufs, *p["inputs"])
                keep = None if m is None else np.nonzero(np.asarray(m, dtype=bool))[0]
                ccols = [(np.asarray(c) if keep is None else np.asarray(c)[keep])
                         for c in _tcols]
                vv = np.asarray(v) if keep is None else np.asarray(v)[keep]
                ukeys, inv, counts = _composite_tuple_index(ccols)
                gi_ms = (time.perf_counter() - t0) * 1000
                t_a = time.perf_counter()
                sums = _tuple_sums(vv, inv, len(ukeys))
                agg_ms = (time.perf_counter() - t_a) * 1000
                # Chain: tuple traversal -> ColumnCarry -> explicit dict only.
                # Same result contract as the packed lane below; the tuple
                # ukeys are the key representation, so a carry costs no int64
                # round trip (see _key_store).
                _carry = _ColumnCarry(ukeys, counts, {"v": sums})
                _carry_result(p["out"], bufs, _carry, params,
                              lambda ct: _carry.to_dict_flat(gop, threads=ct))
                bufs[p["out"] + "#strategy"] = "composite"
                bufs[p["out"] + "#groupindex"] = {
                    "strategy": "composite",
                    "reason": "tuple path: validity AND + lex-sort unique + int64 sums/count",
                    "gi_ms": gi_ms, "agg_ms": agg_ms}
                continue
            gop = params["op"]
            if gop not in ("sum", "count", "mean", "min", "max"):

                raise err(
                    f"CPU driver: unknown groupby op '{gop}'",
                    fix="use one of sum/count/mean/min/max (or groupby_multi for fused)",
                    doc="specs/02-semantic-ir.md",
                )
            m = _valid_mask(bufs, *p["inputs"])
            kk = k if m is None else k[np.nonzero(m)[0]]
            vv = v if m is None else v[np.nonzero(m)[0]]
            if not np.issubdtype(kk.dtype, np.integer):
                raise err(
                    "CPU driver: groupby needs integer keys",
                    fix="encode categoricals to int codes first (encode_pattern/pack_keys)",
                    doc="specs/02-semantic-ir.md",
                )
            if gop in ("min", "max"):
                # Grouped min/max: proven traversal + single-pass O(N)
                # min/max lane (same group_index as sums; native sum-only
                # backends never see min/max). Invalid rows already
                # compacted above (DELTA-3); empty -> empty dict.
                t_g = time.perf_counter()
                gi, gi_dec, gi_ms = _group_index(
                    np.asarray(kk), plan_groupby, naggs=1)
                ukeys = gi["ukeys"]
                counts = _group_counts(gi)
                t_a = time.perf_counter()
                _lex = params.get("dictionary")
                if _lex is None:
                    _lex = bufs.get(p["inputs"][0] + "#dictionary")
                mm = _group_minmax(np.asarray(vv), gi, gop, dictionary=_lex)
                agg_ms = (time.perf_counter() - t_a) * 1000
                kw = {"mins": {"v": mm}} if gop == "min" else {"maxs": {"v": mm}}
                _carry = _ColumnCarry(ukeys, counts, {}, **kw)
                _carry_result(p["out"], bufs, _carry, params,
                              lambda ct: _carry.to_dict_flat(gop, threads=ct))
                bufs[p["out"] + "#strategy"] = gi["strategy"]
                bufs[p["out"] + "#groupindex"] = {
                    "strategy": gi["strategy"],
                    "reason": gi_dec.get("reason", "") + "; minmax single-pass",
                    "gi_ms": gi_ms, "agg_ms": agg_ms}
                continue
            # Native dense path (Planner owns selection; native is execution).
            # Chain: native aggregate -> ColumnCarry -> explicit dict only.
            # Planner rejects / backend absent / int-exact guard fails ->
            # proven chain below runs verbatim (one API, one semantics).
            _hit = None
            if plan_groupby is not None and _native_available():
                _t = _mt_threads()
                _karr, _varr = np.asarray(kk), [np.asarray(vv)]
                _kb, _vb = _probe_all(_karr, _varr, _t)
                _dec, _m, _pms = _dense_gate(_karr, plan_groupby,
                                             naggs=1, _kb=_kb)
                if _dec is not None:
                    try:
                        if _t > 1:
                            _r = _native_mt(_karr, _varr,
                                            int(_m), _t, _vb=_vb)
                            if _r is not None:
                                _uk, _cc, (_ss,), _ams, _cms, _mms = _r
                                _pick = _resolve_on_worse(
                                    _ams + _cms + _mms, None, None)
                                _hit = (_uk, _cc, _ss, _dec, _pms, _ams,
                                        _cms, _mms, _t, _pick)
                        else:
                            _r = _native_st(_karr, _varr,
                                            int(_m), _vb=_vb)
                            if _r is not None:
                                _uk, _cc, (_ss,), _ams, _cms = _r
                                _pick = _resolve_on_worse(_ams + _cms,
                                                          None, None)
                                _hit = (_uk, _cc, _ss, _dec, _pms, _ams,
                                        _cms, 0.0, 1, _pick)
                    except Exception:  # noqa: BLE001
                        _hit = None
            if _hit is not None:
                ukeys, counts, sums, gi_dec, gi_ms, agg_ms, carry_ms, \
                    merge_ms, _tn, _pick = _hit
                if np.issubdtype(np.asarray(vv).dtype, np.integer):
                    sums = np.ascontiguousarray(sums).astype(
                        np.int64, copy=False)
                _carry = _ColumnCarry(ukeys, counts, {"v": sums})
                _carry_result(p["out"], bufs, _carry, params,
                              lambda ct: _carry.to_dict_flat(gop, threads=ct))
                bufs[p["out"] + "#strategy"] = "dense_by_code"
                bufs[p["out"] + "#groupindex"] = {
                    "strategy": "dense_by_code",
                    "reason": gi_dec.get("reason", "") +
                    f"; backend=native_{'mt%d' % _tn if _tn > 1 else 'st'} "
                    f"resolve={_pick} carry->dict explicit",
                    "gi_ms": gi_ms, "agg_ms": agg_ms, "carry_ms": carry_ms,
                    "merge_ms": merge_ms, "threads": _tn, "backend": "native"}
            else:
                # E-track: routed single-pass (sorted-run / dense-by-code fused /
                # small-start hash grouping, one probe set + one Planner gate).
                # None -> proven group_index + separate-aggregate chain verbatim.
                _kind, _pay, gi_ms, agg_ms = _routed_aggregate(
                    np.asarray(kk), [np.asarray(vv)], plan_groupby, naggs=1)
                if _kind == "sorted":
                    gi, counts, sums = _pay
                    ukeys = gi["ukeys"]
                    gi_dec = {"strategy": "sorted",
                              "reason": "sorted input: single-pass run accumulate"}
                elif _kind == "dense":
                    gi, gi_dec, counts, sums_list = _pay
                    ukeys = gi["ukeys"]
                    (sums,) = sums_list
                elif _kind == "hash":
                    gi, gi_dec = _pay[0], _pay[1]
                    t_agg = time.perf_counter()
                    sums, counts = _fused_sums_counts(np.asarray(vv), gi)
                    agg_ms = (time.perf_counter() - t_agg) * 1000
                    ukeys = gi["ukeys"]
                else:
                    gi, gi_dec, gi_ms = _group_index(
                        np.asarray(kk), plan_groupby, naggs=1)
                    t_agg = time.perf_counter()
                    ukeys = gi["ukeys"]
                    if gi["strategy"] == "dense_by_code":
                        ukeys, counts, (sums,) = _fused_soa_by_code(
                            gi["k64"], [np.asarray(vv)], gi["m"], ukeys,
                            gi["counts_m"])
                    else:
                        sums, counts = _fused_sums_counts(np.asarray(vv), gi)
                    agg_ms = (time.perf_counter() - t_agg) * 1000
                _carry = _ColumnCarry(ukeys, counts, {"v": sums})
                _carry_result(p["out"], bufs, _carry, params,
                              lambda ct: _carry.to_dict_flat(gop, threads=ct))
                bufs[p["out"] + "#strategy"] = gi["strategy"]
                bufs[p["out"] + "#groupindex"] = {
                    "strategy": gi["strategy"], "reason": gi_dec.get("reason", ""),
                    "gi_ms": gi_ms, "agg_ms": agg_ms}
        elif op == "groupby_multi":
            cols = params.get("cols")
            if cols is None:  # legacy single-column: inputs [values, keys]
                cols, multi = [p["inputs"][0]], False
                key_name = p["inputs"][1]
                ops_map = {cols[0]: params["ops"]}
            else:  # multi-column: inputs [*values, keys], one traversal
                multi = True
                key_name = p["inputs"][-1]
                ops_map = params["ops"]
            k = bufs[key_name]
            vs = [bufs[c] for c in cols]
            for vv in vs:
                if vv.size != k.size:
                    raise err(
                        f"CPU driver: groupby_multi values/keys size mismatch {vv.size} != {k.size}",
                        fix="pass equal-length values and keys",
                        doc="specs/delta-1-fused-aggregate.md",
                    )
            if not np.issubdtype(k.dtype, np.integer):
                raise err(
                    "CPU driver: groupby_multi needs integer keys",
                    fix="encode categoricals to int codes first (encode_pattern/pack_keys)",
                    doc="specs/delta-2-composite-keys.md",
                )
            _tcols = bufs.get(key_name + "#tuple_cols")
            if _tcols is not None:
                # Generic tuple/composite path (Q36 4-col, CPU-only): one
                # traversal for ALL value columns, validity AND (NULL
                # excluded), lex-sort structured unique with sorted tuple
                # ukeys, int64 sums/count, mean derived. Never an int64
                # scalar pack as the sole path.
                if any("min" in ops_map[c] or "max" in ops_map[c]
                       for c in cols):
                    raise err(
                        "CPU driver: composite groupby_multi needs sum/count/mean",
                        fix="use sum/count/mean on tuple keys (min/max stay single-col)",
                        doc="specs/delta-2-composite-keys.md",
                    )
                t0 = time.perf_counter()
                m = _valid_mask(bufs, *p["inputs"])
                keep = None if m is None else np.nonzero(np.asarray(m, dtype=bool))[0]
                ccols = [(np.asarray(c) if keep is None else np.asarray(c)[keep])
                         for c in _tcols]
                vks = [(np.asarray(vv) if keep is None else np.asarray(vv)[keep])
                       for vv in vs]
                ukeys, inv, counts = _composite_tuple_index(ccols)
                gi_ms = (time.perf_counter() - t0) * 1000
                t_a = time.perf_counter()
                sums_d = {}
                for c, vk in zip(cols, vks):
                    s = _tuple_sums(vk, inv, len(ukeys))
                    if np.issubdtype(np.asarray(vk).dtype, np.integer):
                        s = np.ascontiguousarray(s).astype(np.int64, copy=False)
                    sums_d[c] = s
                agg_ms = (time.perf_counter() - t_a) * 1000
                # Chain: tuple traversal -> ColumnCarry -> explicit dict only,
                # same result contract as the packed lane below. Mean stays
                # derived sum/count, and it is now derived where it is asked
                # for (ColumnCarry.means, cached) instead of eagerly here.
                _carry = _ColumnCarry(ukeys, counts, sums_d)
                if not multi:
                    _c0 = cols[0]
                    _carry_result(p["out"], bufs, _carry, params,
                                  lambda ct, _c0=_c0: _carry.to_dict_single(
                                      _c0, ops_map[_c0], threads=ct))
                else:
                    _carry_result(p["out"], bufs, _carry, params,
                                  lambda ct: _carry.to_dict_multi(
                                      cols, ops_map, threads=ct))
                bufs[p["out"] + "#strategy"] = "composite"
                bufs[p["out"] + "#groupindex"] = {
                    "strategy": "composite",
                    "reason": "tuple path: validity AND + lex-sort unique + int64 sums/count",
                    "gi_ms": gi_ms, "agg_ms": agg_ms}
                continue
            m = _valid_mask(bufs, *p["inputs"])
            keep = None if m is None else np.nonzero(m)[0]
            kk = k if keep is None else k[keep]
            vks = [(vv if keep is None else vv[keep]) for vv in vs]
            _need_mm = any("min" in ops_map[c] or "max" in ops_map[c]
                           for c in cols)
            if _need_mm:
                # Fused min/max (+sum/count/mean co-aggs): one grouping
                # traversal, one min/max lane per min/max column (same
                # _fused_sums_* pattern); sums lane only where needed.
                t_g = time.perf_counter()
                gi, gi_dec, gi_ms = _group_index(
                    np.asarray(kk), plan_groupby, naggs=len(cols))
                ukeys, ng = gi["ukeys"], gi["ukeys"].size
                counts = _group_counts(gi)
                t_a = time.perf_counter()
                sums_d, mins_d, maxs_d = {}, {}, {}
                for c, vk in zip(cols, vks):
                    ops = ops_map[c]
                    va = np.asarray(vk)
                    _lexc = bufs.get(c + "#dictionary")
                    if _lexc is None and isinstance(params.get("dictionaries"),
                                                   dict):
                        _lexc = params["dictionaries"].get(c)
                    if "min" in ops:
                        mins_d[c] = _group_minmax(va, gi, "min",
                                                  dictionary=_lexc)
                    if "max" in ops:
                        maxs_d[c] = _group_minmax(va, gi, "max",
                                                  dictionary=_lexc)
                    if "sum" in ops or "mean" in ops:
                        if gi["strategy"] == "dense_by_code" and "k64" in gi:
                            (_u, _cc, (_s,)) = _fused_soa_by_code(
                                gi["k64"], [va], gi["m"], ukeys,
                                gi["counts_m"])
                            sums_d[c] = _s
                        else:
                            sums_d[c] = _fused_sums_only(va, gi)
                agg_ms = (time.perf_counter() - t_a) * 1000
                _carry = _ColumnCarry(ukeys, counts, sums_d,
                                      mins_d or None, maxs_d or None)
                if not multi:
                    c = cols[0]
                    _carry_result(p["out"], bufs, _carry, params,
                                  lambda ct: _carry.to_dict_single(
                                      c, ops_map[c], threads=ct))
                else:
                    _carry_result(p["out"], bufs, _carry, params,
                                  lambda ct: _carry.to_dict_multi(
                                      cols, ops_map, threads=ct))
                bufs[p["out"] + "#strategy"] = gi["strategy"]
                bufs[p["out"] + "#groupindex"] = {
                    "strategy": gi["strategy"],
                    "reason": gi_dec.get("reason", "") + "; minmax single-pass",
                    "gi_ms": gi_ms, "agg_ms": agg_ms}
                continue
            # Native fused path (Planner owns selection; native is execution).
            # Chain: native multi-aggregate -> ColumnCarry per col ->
            # explicit dict only. Fallback below runs verbatim.
            _hit = None
            if plan_groupby is not None and _native_available():
                _t = _mt_threads()
                _karr = np.asarray(kk)
                _varrs = [np.asarray(vk) for vk in vks]
                _kb, _vb = _probe_all(_karr, _varrs, _t)
                _dec, _m, _pms = _dense_gate(_karr, plan_groupby,
                                             naggs=len(cols), _kb=_kb)
                if _dec is not None:
                    try:
                        if _t > 1:
                            _r = _native_mt(
                                _karr, _varrs, int(_m), _t, _vb=_vb)
                            if _r is not None:
                                _uk, _cc, _ss, _ams, _cms, _mms = _r
                                _pick = _resolve_on_worse(
                                    _ams + _cms + _mms, None, None)
                                _hit = (_uk, _cc, _ss, _dec, _pms, _ams,
                                        _cms, _mms, _t, _pick)
                        else:
                            _r = _native_st(
                                _karr, _varrs, int(_m), _vb=_vb)
                            if _r is not None:
                                _uk, _cc, _ss, _ams, _cms = _r
                                _pick = _resolve_on_worse(_ams + _cms,
                                                          None, None)
                                _hit = (_uk, _cc, _ss, _dec, _pms, _ams,
                                        _cms, 0.0, 1, _pick)
                    except Exception:  # noqa: BLE001
                        _hit = None
            if _hit is not None:
                ukeys, counts, sums_list, gi_dec, gi_ms, agg_ms, carry_ms, \
                    merge_ms, _tn, _pick = _hit
                ukeys, ng = ukeys, ukeys.size
                sums_d = {}
                for c, sums, vk in zip(cols, sums_list, vks):
                    if np.issubdtype(np.asarray(vk).dtype, np.integer):
                        sums = np.ascontiguousarray(sums).astype(
                            np.int64, copy=False)
                    sums_d[c] = sums
                gi = {"strategy": "dense_by_code", "ukeys": ukeys}
                gi_dec = {"strategy": "dense_by_code",
                          "reason": gi_dec.get("reason", "") +
                          f"; backend=native_{'mt%d' % _tn if _tn > 1 else 'st'} "
                          f"resolve={_pick} carry->dict explicit"}
                _nmeta = {"carry_ms": carry_ms, "merge_ms": merge_ms,
                          "threads": _tn, "backend": "native",
                          "resolve": _pick}
            else:
                # E-track: routed single-pass (one probe set + one Planner gate).
                # None -> proven group_index + separate-aggregate chain verbatim.
                _kind, _pay, gi_ms, agg_ms = _routed_aggregate(
                    np.asarray(kk), [np.asarray(vk) for vk in vks],
                    plan_groupby, naggs=len(cols))
                if _kind == "dense":
                    gi, gi_dec, counts, sums_list = _pay
                    ukeys, ng = gi["ukeys"], gi["ukeys"].size
                    sums_d = {}
                    for c, sums in zip(cols, sums_list):
                        sums_d[c] = sums
                elif _kind in ("hash", "sorted"):
                    # 'hash': small-start grouping, proven per-column agg
                    # (counts computed once, shared across columns).
                    # 'sorted': single-column run kernel contract
                    # (sorted_fused_aggregate is 1-col only); multi-col
                    # sorted input downgrades to the proven chain below.
                    if _kind == "sorted" and len(cols) != 1:
                        gi, gi_dec, gi_ms = _group_index(
                            np.asarray(kk), plan_groupby, naggs=len(cols))
                        ukeys, ng = gi["ukeys"], gi["ukeys"].size
                        sums_d = {}
                        t_agg = time.perf_counter()
                        _vks_arr = [np.asarray(vk) for vk in vks]
                        _s0, counts = _fused_sums_counts(_vks_arr[0], gi)
                        sums_d[cols[0]] = _s0
                        for c, vk in zip(cols[1:], _vks_arr[1:]):
                            sums_d[c] = _fused_sums_only(vk, gi)
                        agg_ms = (time.perf_counter() - t_agg) * 1000
                    elif _kind == "hash":
                        gi, gi_dec = _pay[0], _pay[1]
                        ukeys, ng = gi["ukeys"], gi["ukeys"].size
                        sums_d = {}
                        t_agg = time.perf_counter()
                        _vks_arr = [np.asarray(vk) for vk in vks]
                        _s0, counts = _fused_sums_counts(_vks_arr[0], gi)
                        sums_d[cols[0]] = _s0
                        for c, vk in zip(cols[1:], _vks_arr[1:]):
                            sums_d[c] = _fused_sums_only(vk, gi)
                        agg_ms = (time.perf_counter() - t_agg) * 1000
                    else:
                        gi, counts, sums = _pay
                        gi_dec = {"strategy": "sorted",
                                  "reason": "sorted input: single-pass run accumulate"}
                        ukeys, ng = gi["ukeys"], gi["ukeys"].size
                        sums_d = {cols[0]: sums}
                        # agg_ms already times the fused kernel in _pay.
                else:
                    gi, gi_dec, gi_ms = _group_index(np.asarray(kk), plan_groupby,
                                                      naggs=len(cols))
                    ukeys, ng = gi["ukeys"], gi["ukeys"].size
                    sums_d = {}
                    t_agg = time.perf_counter()
                    if gi["strategy"] == "dense_by_code":
                        _uk, counts, sums_list = _fused_soa_by_code(
                            gi["k64"], [np.asarray(vk) for vk in vks],
                            gi["m"], ukeys, gi["counts_m"])  # one traversal, shared counts (SoA)
                        for c, sums in zip(cols, sums_list):
                            sums_d[c] = sums
                    else:
                        _vks_arr = [np.asarray(vk) for vk in vks]
                        _s0, counts = _fused_sums_counts(_vks_arr[0], gi)
                        sums_d[cols[0]] = _s0
                        for c, vk in zip(cols[1:], _vks_arr[1:]):
                            sums_d[c] = _fused_sums_only(vk, gi)
                    agg_ms = (time.perf_counter() - t_agg) * 1000
                carry_ms, merge_ms, _tn, _pick = 0.0, 0.0, 1, "fallback"
                _nmeta = None
            _carry = _ColumnCarry(ukeys, counts, sums_d)
            if not multi:
                c = cols[0]
                _carry_result(p["out"], bufs, _carry, params,
                              lambda ct: _carry.to_dict_single(
                                  c, ops_map[c], threads=ct))
            else:
                _carry_result(p["out"], bufs, _carry, params,
                              lambda ct: _carry.to_dict_multi(
                                  cols, ops_map, threads=ct))
            bufs[p["out"] + "#strategy"] = gi["strategy"]
            _ginfo = {
                "strategy": gi["strategy"], "reason": gi_dec.get("reason", ""),
                "gi_ms": gi_ms, "agg_ms": agg_ms}
            if _nmeta is not None:
                _ginfo.update(_nmeta)
            elif carry_ms or merge_ms or _tn > 1 or _pick != "fallback":
                _ginfo.update({"carry_ms": carry_ms, "merge_ms": merge_ms,
                               "threads": _tn, "backend": "numpy-fallback",
                               "resolve": _pick})
            bufs[p["out"] + "#groupindex"] = _ginfo
        elif op == "group_count_distinct":
            # CPU-op (NOT a new IR): {key: nunique} dict, keys sorted.
            # Contract: _valid_mask -> compact -> _sort_perm(keys,values) ->
            # one vector scan (group bounds + distinct transitions).
            # Co-aggregates (sum/count/mean) are separate nodes, never here.
            # chunkable=false in v1 (spec 12: single-pass order contract,
            # not a math ban -- chunked backends must re-sort per chunk).
            v = bufs[p["inputs"][0]]
            k = bufs[p["inputs"][1]]
            if v.size != k.size:
                raise err(
                    f"CPU driver: group_count_distinct values/keys size mismatch {v.size} != {k.size}",
                    fix="pass equal-length values and keys",
                    doc="specs/12-relational-family.md",
                )
            if not np.issubdtype(np.asarray(k).dtype, np.integer):
                raise err(
                    "CPU driver: group_count_distinct needs integer keys",
                    fix="encode categoricals to int codes first (encode_pattern/pack_keys)",
                    doc="specs/12-relational-family.md",
                )
            if not np.issubdtype(np.asarray(v).dtype, np.integer):
                raise err(
                    f"CPU driver: group_count_distinct needs integer values, got {np.asarray(v).dtype}",
                    fix="encode categoricals (e.g. UserID factorize) to int codes first",
                    doc="specs/12-relational-family.md",
                )
            t_f = time.perf_counter()
            m = _valid_mask(bufs, *p["inputs"])
            keep = None if m is None else np.nonzero(np.asarray(m, dtype=bool))[0]
            kk = k if keep is None else np.asarray(k)[keep]
            vv = v if keep is None else np.asarray(v)[keep]
            filter_ms = (time.perf_counter() - t_f) * 1000
            _gh = grouped_distinct_hash
            _gm = GroupedHashMiss
            if _gh is not None and _gm is not None:
                try:
                    ukeys, nunique, sort_ms, scan_ms = _gh(
                        np.asarray(kk), np.asarray(vv))
                    _dist_strategy = "pair_hash"
                    _dist_reason = ("pair open-addressing hash over (keys,values) "
                                    "as two independent arrays (no domain "
                                    "factorize, full-key compare) + unique over "
                                    "occupied keys")
                except _gm:
                    ukeys, nunique, sort_ms, scan_ms = _count_distinct_sorted_dedup(
                        np.asarray(kk), np.asarray(vv), err)
                    _dist_strategy = "sorted_dedup"
                    _dist_reason = ("valid-mask compact + sort(keys,values) + "
                                    "single vector scan (group bounds + value "
                                    "transitions)")
            else:
                ukeys, nunique, sort_ms, scan_ms = _count_distinct_sorted_dedup(
                    np.asarray(kk), np.asarray(vv), err)
                _dist_strategy = "sorted_dedup"
                _dist_reason = ("valid-mask compact + sort(keys,values) + "
                                "single vector scan (group bounds + value "
                                "transitions)")
            _carry = _ColumnCarry(ukeys, nunique, {"v": nunique})
            _carry_result(p["out"], bufs, _carry, params,
                          lambda ct: _carry.to_dict_flat("count", threads=ct))
            bufs[p["out"] + "#strategy"] = _dist_strategy
            bufs[p["out"] + "#groupindex"] = {
                "strategy": _dist_strategy,
                "reason": _dist_reason,
                "filter_ms": filter_ms, "sort_ms": sort_ms,
                "scan_ms": scan_ms}
        elif op == "reduce":
            rop = params["op"]
            skipna = params.get("skipna", False)
            ddof = params.get("ddof", 0)
            # P5: fused map→reduce — inline map, skip intermediate buffer.
            _fm = _fused_mr.get(p["out"])
            if _fm is not None:
                a = bufs[_fm["map_inputs"][0]]
                _fn = _fm["fn"]
                _bv = (_fm["map_inputs"][1]
                       if len(_fm["map_inputs"]) > 1
                       else _fm.get("value"))
                if _fn == "add":
                    a = a + _bv
                elif _fn == "sub":
                    a = a - _bv
                elif _fn == "mul":
                    a = a * _bv
                elif _fn == "div":
                    a = (a.astype(np.float64) / _bv)
                    if np.issubdtype(a.dtype, np.integer):
                        a = np.rint(a).astype(a.dtype)
                elif _fn == "pow":
                    a = (a.astype(np.float64) ** _bv)
                    if np.issubdtype(a.dtype, np.integer):
                        a = np.rint(a).astype(a.dtype)
                elif _fn == "floor_div":
                    a = np.floor_divide(a, _bv)
                elif _fn == "mod":
                    a = np.remainder(a, _bv)
                else:
                    raise err(
                        f"CPU driver: fused map unknown fn '{_fn}'",
                        fix="use one of add/sub/mul/div/pow/floor_div/mod",
                        doc="specs/02-semantic-ir.md",
                    )
                # Validity: use map's input validity (same mask as the
                # intermediate would have had).
                m = _valid_mask(bufs, *_fm["map_inputs"])
            else:
                a = bufs[p["inputs"][0]]
                m = _valid_mask(bufs, *p["inputs"])
            if m is not None:
                if skipna:
                    a = a[m]
                    if a.size == 0 and rop != "count":
                        import warnings

                        warnings.warn(
                            f"CPU driver: reduce '{rop}' over all-NA input -> NaN "
                            "(delta-3). Fix: ensure at least one valid row."
                        )
                        bufs[p["out"]] = float("nan")
                        continue
                elif not bool(m.all()) and rop != "count" and a.size:
                    import warnings

                    warnings.warn(
                        f"CPU driver: reduce '{rop}' with NA and skipna=False -> NaN "
                        "(delta-3). Fix: pass skipna=True to exclude NA."
                    )
                    bufs[p["out"]] = float("nan")
                    continue
            _is_int = a.dtype.kind in "iu"
            has_nan = False if _is_int else (bool(np.isnan(a.astype(np.float64)).any()) if a.size else False)
            if has_nan and not skipna and rop != "count":
                import warnings

                warnings.warn(
                    f"CPU driver: reduce '{rop}' with NaN and skipna=False -> NaN "
                    "(spec 08). Fix: pass skipna=True to exclude NaN."
                )
                bufs[p["out"]] = float("nan")
            elif rop == "count":
                if skipna and a.size:
                    if _is_int:
                        bufs[p["out"]] = int(a.size)
                    else:
                        bufs[p["out"]] = int(np.count_nonzero(~np.isnan(a.astype(np.float64))))
                else:
                    bufs[p["out"]] = int(a.size)
            elif rop == "sum":
                if skipna:
                    bufs[p["out"]] = np.nansum(a, dtype=np.float64).item()
                else:
                    acc = accum_dtype(str(a.dtype))
                    bufs[p["out"]] = np.sum(a, dtype=as_dtype(acc)).item()
            elif rop == "mean":
                bufs[p["out"]] = float(np.nanmean(a) if skipna else np.sum(a, dtype=np.float64) / a.size)
            elif rop == "min":
                bufs[p["out"]] = (np.nanmin(a) if skipna else a.min()).item()
            elif rop == "max":
                bufs[p["out"]] = (np.nanmax(a) if skipna else a.max()).item()
            elif rop in ("var", "std"):
                f = np.nanvar if skipna else np.var
                if rop == "std":
                    f = np.nanstd if skipna else np.std
                bufs[p["out"]] = float(f(a.astype(np.float64), ddof=ddof))
            else:
                raise err(
                    f"CPU driver: unknown reduce op '{rop}'",
                    fix="use one of sum/count/mean/min/max/var/std",
                    doc="specs/02-semantic-ir.md",
                )
        elif op == "rng_fill_i32":
            n = int(params["n"])
            lo, hi = int(params["lo"]), int(params["hi"])
            seed, stream, off = (int(params["seed"]), int(params["stream"]),
                                 int(params["offset"]))
            mode = 0 if params.get("mode", "bits") == "bits" else -1
            try:
                if _rng_native_available():
                    bufs[p["out"]] = _rng_fill_i32_n(n, seed, stream, off, lo, hi, mode)
                else:
                    raise RuntimeError("rng native absent: fallback owns it")
            except RuntimeError:
                rc, arr = _rng_fill_i32_fb(n, seed, stream, off, lo, hi, mode)
                if rc != 0:
                    raise err(
                        f"CPU driver: rng_fill_i32 rc={rc} (n={n} lo={lo} hi={hi})",
                        fix="pass lo<hi with mode='bits'",
                        doc="specs/02-semantic-ir.md",
                    )
                bufs[p["out"]] = arr
        elif op == "rng_fill_f64":
            n = int(params["n"])
            lo, hi = float(params["lo"]), float(params["hi"])
            seed, stream, off = (int(params["seed"]), int(params["stream"]),
                                 int(params["offset"]))
            try:
                if _rng_native_available():
                    bufs[p["out"]] = _rng_fill_f64_n(n, seed, stream, off, lo, hi)
                else:
                    raise RuntimeError("rng native absent: fallback owns it")
            except RuntimeError:
                rc, arr = _rng_fill_f64_fb(n, seed, stream, off, lo, hi)
                if rc != 0:
                    raise err(
                        f"CPU driver: rng_fill_f64 rc={rc} (n={n} lo={lo} hi={hi})",
                        fix="pass finite lo<hi",
                        doc="specs/02-semantic-ir.md",
                    )
                bufs[p["out"]] = arr
        elif op == "rng_sample_no_replace":
            n, k = int(params["n"]), int(params["k"])
            seed, stream, off = (int(params["seed"]), int(params["stream"]),
                                 int(params["offset"]))
            try:
                if _rng_native_available():
                    bufs[p["out"]] = _rng_sample_n(n, k, seed, stream, off)
                else:
                    raise RuntimeError("rng native absent: fallback owns it")
            except RuntimeError:
                rc, arr = _rng_sample_fb(n, k, seed, stream, off)
                if arr is None or (isinstance(rc, int) and rc < 0):
                    raise err(
                        f"CPU driver: rng_sample_no_replace rc={rc} (n={n} k={k})",
                        fix="pass 0<=k<=n",
                        doc="specs/02-semantic-ir.md",
                    )
                bufs[p["out"]] = arr
        elif op == "rng_permutation":
            n = int(params["n"])
            seed, stream, off = (int(params["seed"]), int(params["stream"]),
                                 int(params["offset"]))
            try:
                if _rng_native_available():
                    bufs[p["out"]] = _rng_permutation_n(n, seed, stream, off)
                else:
                    raise RuntimeError("rng native absent: fallback owns it")
            except RuntimeError:
                rc, arr = _rng_permutation_fb(n, seed, stream, off)
                if arr is None or (isinstance(rc, int) and rc < 0):
                    raise err(
                        f"CPU driver: rng_permutation rc={rc} (n={n})",
                        fix="pass n>=0 within int32",
                        doc="specs/02-semantic-ir.md",
                    )
                bufs[p["out"]] = arr
        elif op == "rng_compat":
            kind = params.get("kind", "runif")
            seed = int(params["seed"])
            if kind == "runif":
                n = int(params["n"])
                lo, hi = float(params["lo"]), float(params["hi"])
                try:
                    if _rng_native_available():
                        bufs[p["out"]] = _rng_compat_runif_n(n, seed, lo, hi)
                    else:
                        raise RuntimeError("rng native absent: fallback owns it")
                except RuntimeError:
                    rc, arr = _rng_compat_runif_fb(n, seed, lo, hi)
                    if rc != 0:
                        raise err(
                            f"CPU driver: rng_compat runif rc={rc} (n={n})",
                            fix="pass finite lo<hi",
                            doc="specs/02-semantic-ir.md",
                        )
                    bufs[p["out"]] = arr
            else:
                n, m = int(params["n"]), int(params["m"])
                try:
                    if _rng_native_available():
                        bufs[p["out"]] = _rng_compat_sample_n(n, m, seed)
                    else:
                        raise RuntimeError("rng native absent: fallback owns it")
                except RuntimeError:
                    rc, arr = _rng_compat_sample_fb(n, m, seed)
                    if rc != 0:
                        raise err(
                            f"CPU driver: rng_compat sample rc={rc} (n={n} m={m})",
                            fix="pass n>0",
                            doc="specs/02-semantic-ir.md",
                        )
                    bufs[p["out"]] = arr
            bufs[p["out"] + "#compat"] = {"kind": kind, "chunkable": False}
        elif op == "map_round":
            a = bufs[p["inputs"][0]]
            try:
                x = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
            except (TypeError, ValueError):
                raise err(
                    "CPU driver: map_round needs a numeric column",
                    fix="pass an int/float Series",
                    doc="specs/02-semantic-ir.md",
                )
            mv = bufs.get(p["inputs"][0] + "#validity")
            vu8 = (None if mv is None
                   else np.ascontiguousarray(np.asarray(mv, dtype=bool)).astype(np.uint8))
            nd = int(params.get("ndigits", 0))
            try:
                if _rng_native_available():
                    vv = (np.ones(x.size, dtype=np.uint8) if vu8 is None else vu8)
                    o, ov = _rng_map_round_n(x, vv, nd)
                else:
                    raise RuntimeError("rng native absent: fallback owns it")
            except RuntimeError:
                rc, o, ov = _rng_map_round_fb(x, vu8, nd)
                if rc != 0:
                    raise err(
                        f"CPU driver: map_round rc={rc} (ndigits={nd})",
                        fix="pass ndigits=0..15",
                        doc="specs/02-semantic-ir.md",
                    )
            bufs[p["out"]] = o
            bufs[p["out"] + "#validity"] = np.ascontiguousarray(
                np.asarray(ov, dtype=bool))
        elif op == "unique_inverse":
            # Sorted-order unique + inverse (chunkable=false). Callers needing
            # inv (rng_api) use this op explicitly; uniq[inv] == values.
            a = bufs[p["inputs"][0]]
            keys = np.ascontiguousarray(np.asarray(a))
            if keys.dtype.kind == "b":
                keys = keys.astype(np.int32)
            if keys.dtype.kind not in "iu":
                raise err(
                    f"CPU driver: unique_inverse needs int32/int64 keys, got {keys.dtype}",
                    fix="encode categoricals to int32 codes first",
                    doc="specs/02-semantic-ir.md",
                )
            mv = bufs.get(p["inputs"][0] + "#validity")
            if mv is None:
                # M7b: fused radix lane above _UNIQUE_NATIVE_MIN (measured
                # 1.08-2.54x at n >= 1e5 across cardinality 2..n, EXACT
                # parity with np.unique(return_inverse=True)). The validity
                # path below is untouched -- it compacts rows first, so its
                # key count is not the array length the gate keys on.
                got = _native_unique_lane(keys)
                if got is not None:
                    ukeys, inv = got
                    inv = np.ascontiguousarray(inv.astype(np.int32))
                else:
                    ukeys, inv = np.unique(keys, return_inverse=True)
                    inv = np.ascontiguousarray(inv.astype(np.int32))
            else:
                m = np.ascontiguousarray(np.asarray(mv, dtype=bool))
                ukeys, inv_v = np.unique(keys[m], return_inverse=True)
                inv = np.full(keys.shape, -1, dtype=np.int32)
                inv[m] = inv_v.astype(np.int32)
            ng = int(ukeys.size)
            bufs[p["out"]] = np.ascontiguousarray(ukeys)
            bufs[p["out"] + "#inv"] = np.ascontiguousarray(inv)
            bufs[p["out"] + "#ng"] = ng
        elif op == "unique":
            # DISTINCT set only (no inverse): scalar-distinct compositions
            # (unique+reduce(count)) never consume #inv; skipping the
            # N-sized inverse build plus the radix lane (LSD radix via
            # stable argsort, _unique_sorted) serve all scalar COUNT
            # DISTINCT (grouped counts stay in group_count_distinct).
            # Contract: out (sorted uniq) + #ng.
            a = bufs[p["inputs"][0]]
            keys = np.ascontiguousarray(np.asarray(a))
            if keys.dtype.kind == "b":
                keys = keys.astype(np.int32)
            if keys.dtype.kind not in "iu":
                raise err(
                    f"CPU driver: unique needs int32/int64 keys, got {keys.dtype}",
                    fix="encode categoricals to int32 codes first",
                    doc="specs/02-semantic-ir.md",
                )
            mv = bufs.get(p["inputs"][0] + "#validity")
            if mv is None:
                ukeys = _unique_sorted(keys)
            else:
                m = np.ascontiguousarray(np.asarray(mv, dtype=bool))
                ukeys = _unique_sorted(keys[m])
            ng = int(ukeys.size)
            bufs[p["out"]] = np.ascontiguousarray(ukeys)
            bufs[p["out"] + "#ng"] = ng
        elif op == "lookup":
            # Reuses Join mechanics verbatim (same algorithm, local copy: no
            # cross-Extension imports): stable-argsort build -> sorted-unique
            # U[K] (dupe = explicit error, Join contract) -> vectorized
            # searchsorted probe (C-speed, no per-row Python loop).
            b = np.ascontiguousarray(np.asarray(bufs[p["inputs"][0]]).ravel(),
                                     dtype=np.int32)
            q = np.ascontiguousarray(np.asarray(bufs[p["inputs"][1]]).ravel(),
                                     dtype=np.int32)
            if np.asarray(bufs[p["inputs"][0]]).dtype.kind not in "iu" \
                    or np.asarray(bufs[p["inputs"][1]]).dtype.kind not in "iu":
                raise err(
                    f"CPU driver: lookup needs int32/int64 keys, got "
                    f"{np.asarray(bufs[p['inputs'][0]]).dtype}/"
                    f"{np.asarray(bufs[p['inputs'][1]]).dtype}",
                    fix="encode categoricals to int32 codes first",
                    doc="specs/02-semantic-ir.md",
                )
            order = np.argsort(b.astype(np.int64), kind="stable")
            u = b[order]
            if u.size > 1 and bool((u[1:] == u[:-1]).any()):
                raise err(
                    "CPU driver: lookup build keys not unique: dupes collapse",
                    fix="dedupe the build side (unique_inverse) or use "
                        "groupby/Join for many-to-one",
                    doc="specs/02-semantic-ir.md",
                )
            k = int(u.size)
            pos = np.searchsorted(u, q)
            in_range = pos < k
            hit = np.zeros(q.shape, dtype=bool)
            if k:
                hit = in_range & (u[np.where(in_range, pos, 0)] == q)
            out_pos = np.where(hit, pos, -1).astype(np.int64)
            bufs[p["out"]] = np.ascontiguousarray(out_pos)
            bufs[p["out"] + "#hit"] = np.ascontiguousarray(hit)
            bufs[p["out"] + "#k"] = k
        else:
            raise err(
                f"CPU driver: unsupported op '{op}'",
                fix="capability covers series/map/compare/filter/mask/where/gather/sort/slice/reduce/rolling_sum/shift/cumsum/groupby/groupby_multi/group_count_distinct/pack_keys/encode_pattern/text_length/text_contains/text_startswith/text_endswith/text_equals/text_regex_replace/rng_fill_i32/rng_fill_f64/rng_sample_no_replace/rng_permutation/rng_compat/map_round/unique_inverse/unique/lookup",
                doc="specs/06-drivers-gpu-cpu.md",
            )
    return bufs


def cpu_capability_impl():
    """capability() -> {ops, max_dispatch, max_buffer_bytes, chunkable_hints}."""
    return {
        "ops": ["series", "map", "compare", "filter", "mask", "where", "gather", "sort", "slice", "reduce",
                "rolling_sum", "shift", "cumsum", "groupby", "groupby_multi", "group_count_distinct", "pack_keys", "encode_pattern",
                "text_length", "text_contains", "text_startswith", "text_endswith",
                "text_equals", "text_regex_replace",
                "rng_fill_i32", "rng_fill_f64", "rng_sample_no_replace", "rng_permutation",
                "rng_compat", "map_round", "unique_inverse", "unique", "lookup"],
        "max_dispatch": {"x": 2 ** 31 - 1, "y": 65535, "z": 65535},
        "max_buffer_bytes": 2 ** 31 - 1,
        "chunkable_hints": {"series": True, "map": True, "compare": True,
                            "filter": True, "mask": True, "gather": True,
                            "sort": False, "slice": False,
                            "reduce": True, "rolling_sum": True, "shift": False, "cumsum": False,
                            "groupby": False,
                            "groupby_multi": False, "group_count_distinct": False, "pack_keys": True,
                            "encode_pattern": True,
                            "text_length": True, "text_contains": True,
                            "text_startswith": True, "text_endswith": True,
                            "text_equals": True, "text_regex_replace": True,
                            "rng_fill_i32": True, "rng_fill_f64": True,
                            "rng_sample_no_replace": False, "rng_permutation": False,
                            "rng_compat": False, "map_round": True,
                            "unique_inverse": False, "unique": False, "lookup": False},
        "note": "skeleton CPU-oracle: permissive placeholder limits",
    }
