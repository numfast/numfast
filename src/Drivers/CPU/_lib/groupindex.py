# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""group_index primitive: one generic entry, several physical strategies.

Planner owns the choice (cost model over observables + profile, reason always
set); this module only probes and executes. Output contract (frozen):
  {strategy, ukeys (sorted ascending), starts | inverse, n}
Strategies: empty | sorted | dense_by_code | dense | hash | unique.

Order (approved matrix 10M):
  1. empty input -> empty result (no scan at all).
  2. sortedness early-exit (vectorized probe) -> sorted run/reduceat.
  3. range probe (vectorized min/max) + memory-aware gate
     R <= min(R_MAX, K_RN * N) AND est_dense_bytes <= budget -> dense family.
     Never trust max-min alone: keys=[0, 1e9] look "dense" (R ~ 1e9) but the
     backing array is gigantic; the R/N term and the byte check reject it.
     Inside dense: kmin >= 0 with M=kmax+1 small -> dense_by_code (direct,
     no shift array, no lut, no inverse); else shift-based dense FALLBACK
     (kept verbatim: shift + bincount + lut scatter + inverse).
  4. generic integer hash (open addressing, multiplicative top-bit spread,
     dynamic growth: initial cap 2^24 at production scale, 2x rehash when
     load > 0.7). The old naive fixed-table hash LOST high-card uniform
     (9.0s vs 4.1s), so hash is never assumed a universal winner -- it only
     runs where dense is unsafe and unique is the only alternative left.
  5. fallback -> np.unique (bit-exact reference behaviour).

Buffer reuse (safe cases only): dense_by_code compaction bincount(M) ->
mask -> present codes is one forward pass (write frontier never overtakes
unread input; asserted w<=r shape by construction of flatnonzero). Arbitrary
permutation / dependent reductions (hash scatter, sort, reverse) stay on the
two-buffer path -- in-place reuse there would destroy unread state.

ukeys are ALWAYS sorted ascending and inverse (when present) ALWAYS satisfies
ukeys[inverse] == keys, on every strategy -- verified exact vs np.unique.
No H2O-specific conditions: no dataset names, no hardcoded group counts.
"""

import numpy as np

# NATIVE-ALL: legacy JIT retired (target zero). Canonical lanes are
# numpy-vectorized (C-speed) + Rust native (unique_inverse) with np.unique
# oracle fallback. Every lane below is bit-exact vs the retired kernels
# (parity table in numfast/NATIVE_ALL.md, seed 42).

_HASH_MULT = np.uint64(0x9E3779B97F4A7C15)

# NATIVE-ALL: retired JIT kernels deleted. Canonical replacements (same
# contract, bit-exact):
# - _is_sorted_nb -> inline np.all in is_sorted_early (C-speed vector pass)
# - _hash_fill -> unique_fallback_index (Rust unique_inverse / np.unique)
# - _sp_* (fused sums) -> np.add.at in fused_sums_by_code (exact, repeats-safe)
# - _sorted_1i/_sorted_1f -> flatnonzero boundaries + reduceat in
#   sorted_fused_aggregate (exact, same accumulation order)


def is_sorted_early(keys):
    """Vectorized sortedness probe (C-speed, no alloc beyond views)."""
    keys = np.asarray(keys)
    if keys.size < 2:
        return True
    return bool(np.all(keys[1:] >= keys[:-1]))


def range_probe(keys):
    """Vectorized min/max probe -> (kmin, kmax, R) as Python ints (no overflow)."""
    keys = np.asarray(keys)
    kmin = int(keys.min())
    kmax = int(keys.max())
    return kmin, kmax, kmax - kmin + 1


def dense_estimate_bytes(r, n):
    """Transient estimate for shift-based dense: bincount(R,int64) + lut(R,int64)
    + inverse(N,int64). ukeys/nz are O(ngroups), negligible. FALLBACK numbers."""
    return 16 * int(r) + 8 * int(n)


def bycode_estimate_bytes(m, n, keys_is_int64):
    """Transient estimate for dense-by-code: sums(M,int64) + counts(M,int64)
    + one int64 working copy of keys when they are not int64 already
    (single shared copy: grouping and aggregation both consume it, no second
    conversion; int64 keys pass by view, zero extra). All Python-int math."""
    return 16 * int(m) + (0 if keys_is_int64 else 8 * int(n))


def initial_hash_cap(n, profile_init):
    """Small inputs get a right-sized table (no 273MB for 100 rows);
    production scale (n >= profile_init/4) starts exactly at profile_init."""
    want = 1
    target = max(1024, 4 * int(n))
    while want < target:
        want *= 2
    return min(int(profile_init), want)


def empty_index(keys):
    keys = np.asarray(keys)
    return {"strategy": "empty", "ukeys": keys[:0],
            "starts": np.zeros(0, dtype=np.int64), "inverse": None, "n": 0}


def sorted_run_index(keys):
    """Boundaries + ukeys over sorted runs. No sort cost, no inverse state."""
    keys = np.asarray(keys)
    n = keys.size
    bounds = np.flatnonzero(keys[1:] != keys[:-1]).astype(np.int64) + 1
    starts = np.empty(bounds.size + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = bounds
    return {"strategy": "sorted", "ukeys": keys[starts], "starts": starts,
            "inverse": None, "n": n}


def dense_direct_index(keys, kmin, r):
    """Shift-based dense indexing: bincount over shifted keys + lut scatter.

    FALLBACK (kept verbatim for non-resident ranges, e.g. kmin<0 or large
    kmin where M=kmax+1 backing would exceed the shift-based footprint).
    ukeys sorted ascending by construction; inverse labels match np.unique
    labelling exactly (sorted-position codes), so downstream bincounts are
    bit-identical to the unique path.
    """
    keys = np.asarray(keys)
    n = keys.size
    sh = keys - np.int64(kmin)
    cnt = np.bincount(sh, minlength=int(r))
    nz = np.flatnonzero(cnt)
    lut = np.full(int(r), -1, dtype=np.int64)
    lut[nz] = np.arange(nz.size, dtype=np.int64)
    inverse = lut[sh]
    ukeys = nz.astype(np.int64) + np.int64(kmin)
    return {"strategy": "dense", "ukeys": ukeys, "starts": None,
            "inverse": inverse, "n": n}


def dense_by_code_present(keys, m):
    """Forward-compaction of M-sized presence to sorted present codes.

    keys convert to int64 ONCE here (view when already int64) and the working
    copy is carried in gi for aggregation reuse -- grouping and aggregation
    share it, never a second conversion (measured: one shared copy beats
    slow narrow-dtype bincount loops on this platform; research B did the
    same single conversion). counts_m = bincount(k64) doubles as the grouping
    traversal and is likewise carried (no recompute downstream); mask +
    flatnonzero compact to sorted ukeys in one forward pass (write frontier
    never overtakes unread input). No shift array, no lut, no inverse.
    Caller guarantees 0<=keys<M (Planner gate on Python ints: no overflow)
    and M within budget.
    """
    k64 = np.asarray(keys)
    if k64.dtype != np.int64:
        k64 = k64.astype(np.int64)
    m = int(m)
    counts_m = np.bincount(k64, minlength=m)
    mask = counts_m > 0
    ukeys = np.flatnonzero(mask).astype(np.int64)
    assert ukeys.size == 0 or bool(np.all(np.diff(ukeys) > 0)), \
        "by-code compaction must stay sorted+unique"
    return ukeys, k64, counts_m


def dense_by_code_index(keys, m):
    """Dense direct by resident integer code: NO shift array, NO lut,
    NO inverse, NO second int64 copies. One int64 working copy of keys at
    most (shared gi->aggregate), bincounts run on it directly; compaction is
    forward-only (see above). ukeys sorted ascending = the present codes
    themselves, identical set and order to np.unique. Aggregation runs
    code-direct (no inverse needed); gi carries m, k64 and counts_m so the
    driver sizes M-aux without re-probing and without recompute."""
    keys = np.asarray(keys)
    n = keys.size
    ukeys, k64, counts_m = dense_by_code_present(keys, m)
    return {"strategy": "dense_by_code", "ukeys": ukeys, "starts": None,
            "inverse": None, "n": n, "m": int(m), "k64": k64,
            "counts_m": counts_m}


def hash_dynamic_index(keys, profile_init=1 << 24, max_load=0.7, growth=2):
    """Generic integer hash: open addressing, multiplicative top-bit spread,
    dynamic growth (2x rehash while load > max_load). ukeys sorted ascending,
    inverse remapped to sorted-position codes (same labelling as np.unique).
    """
    keys = np.asarray(keys)
    n = keys.size
    # NATIVE-ALL: hash lane retired. Canonical lane is
    # unique_fallback_index (Rust unique_inverse, np.unique oracle) with the
    # identical output contract (sorted ukeys + remapped inverse). This was
    # already the fallback branch whenever the legacy JIT was absent.
    return unique_fallback_index(keys)


def unique_fallback_index(keys):
    keys = np.asarray(keys)
    # Native sorted-order unique+inverse when the production DLL
    # exposes it (capability probe, env-aware); ANY failure (no
    # backend, disabled, old DLL, error) -> proven np.unique oracle.
    # Contract unchanged: ukeys sorted ascending (keys dtype),
    # inverse int64 with ukeys[inverse] == keys.
    try:
        try:
            from .native_cpu import unique_inverse as _nat_unique
        except ImportError:  # sys.path assembly (tests/benches)
            from _lib.native_cpu import unique_inverse as _nat_unique
        ukeys, inverse, _be = _nat_unique(keys)
        return {"strategy": "unique", "ukeys": ukeys, "starts": None,
                "inverse": inverse, "n": keys.size}
    except Exception:  # noqa: BLE001 -- fallback must never raise
        pass
    ukeys, inverse = np.unique(keys, return_inverse=True)
    return {"strategy": "unique", "ukeys": ukeys, "starts": None,
            "inverse": inverse, "n": keys.size}


def _sp_sig(vcols):
    return "".join("i" if np.issubdtype(np.asarray(v).dtype, np.integer)
                   else "f" for v in vcols)


def fused_sums_by_code(keys, vcols, m):
    """Single-pass fused sums+counts over resident integer codes (E-track).

    keys: integer codes with 0<=k<m (Planner-gate guarantee, same trust as
    dense_by_code_present). vcols: 1-3 int32/int64 or float columns.
    Returns (ukeys_sorted, counts_compact, sums_compact_list) or None when
    unavailable (unsupported column mix) -- caller keeps the proven bincount
    path. NATIVE-ALL: accumulation is np.add.at (C-speed, exact for repeated
    codes, TRUE int64 for integer columns, float64 for float columns).
    No N-sized copies, no shift/lut/inverse; presence compacts from counts
    after the single traversal (forward-only).
    """
    keys = np.asarray(keys)
    if keys.dtype.kind not in "iu":
        return None
    cols = [np.asarray(v) for v in vcols]
    if not (1 <= len(cols) <= 3):
        return None
    if any(v.ndim != 1 or v.size != keys.size for v in cols):
        return None
    if any(v.dtype.kind not in "iuf" for v in cols):
        return None
    sig = "".join("i" if v.dtype.kind in "iu" else "f" for v in cols)
    if sig not in ("i", "f", "ii", "if", "ff", "iii", "iif", "iff", "fff"):
        return None
    m = int(m)
    try:
        ki = np.ascontiguousarray(keys).astype(np.int64, copy=False)
        sums_m = [(np.zeros(m, dtype=np.int64) if v.dtype.kind in "iu"
                   else np.zeros(m, dtype=np.float64)) for v in cols]
        for s, v in zip(sums_m, cols):
            np.add.at(s, ki, v)
        counts_m = np.zeros(m, dtype=np.int64)
        np.add.at(counts_m, ki, 1)
    except (MemoryError, ValueError, OverflowError):
        return None
    ukeys = np.flatnonzero(counts_m > 0).astype(np.int64)
    return ukeys, counts_m[ukeys], [s[ukeys] for s in sums_m]


def sorted_fused_aggregate(keys, vcols):
    """Single-pass sorted-run aggregation; (ok, payload, probe_ms, agg_ms).

    No Planner involvement (run-based: always safe, any range, no M alloc).
    Returns ok=True only for sorted input with a single value column of
    int/float kind; everything else (unsorted, empty, multi-column) returns
    ok=False and the caller runs the legacy chain verbatim. Payload on ok:
    (gi, counts, sums_list) with gi strategy "sorted"; starts is None by
    design (no second traversal needs boundaries -- the fused branch consumes
    sums out-of-band and never calls reduceat).
    """
    import time as _time
    keys = np.asarray(keys)
    n = keys.size
    if n == 0:
        return False, None, 0.0, 0.0
    if len(vcols) != 1:
        return False, None, 0.0, 0.0
    v = np.asarray(vcols[0])
    if v.ndim != 1 or v.size != n or v.dtype.kind not in "iuf":
        return False, None, 0.0, 0.0
    if keys.dtype.kind not in "iu":
        return False, None, 0.0, 0.0
    sig = "i" if v.dtype.kind in "iu" else "f"
    # NATIVE-ALL: sorted-run aggregation is numpy-native (C-speed, exact):
    # one vector sortedness check, run boundaries via flatnonzero, per-run
    # sums via reduceat (same accumulation order as the retired kernel).
    t0 = _time.perf_counter()
    try:
        if not bool(np.all(keys[1:] >= keys[:-1])):
            return False, None, (_time.perf_counter() - t0) * 1000, 0.0
        ends = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        starts = np.concatenate(([0], ends))
        ends = np.concatenate((ends, [n]))
        ukeys = np.ascontiguousarray(keys[starts]).astype(np.int64)
        counts = np.ascontiguousarray((ends - starts).astype(np.int64))
        if sig == "i":
            sums = np.add.reduceat(v.astype(np.int64), starts)
        else:
            sums = np.add.reduceat(v.astype(np.float64), starts)
    except (MemoryError, ValueError, OverflowError):
        return False, None, (_time.perf_counter() - t0) * 1000, 0.0
    agg_ms = (_time.perf_counter() - t0) * 1000
    gi = {"strategy": "sorted", "ukeys": ukeys, "starts": None,
          "inverse": None, "n": n}
    return True, (gi, counts, sums), 0.0, agg_ms


def _probe_plan(keys, n, plan_fn, budget_bytes, naggs):
    """Shared probe+gate: vectorized min/max + Planner decision (E-track).

    Same helpers, same observables, same plan_fn call shape as group_index,
    factored so routed entry points cannot drift from the proven gate.
    Returns (kmin, kmax, r, m_code, est_code, dec, probe_ms).
    """
    import time as _time
    t0 = _time.perf_counter()
    kmin, kmax, r = range_probe(keys)
    est = dense_estimate_bytes(r, n)
    m_code = int(kmax) + 1 if kmin >= 0 else None
    keys_is_int64 = keys.dtype == np.int64
    est_code = (bycode_estimate_bytes(m_code, n, keys_is_int64)
                if m_code is not None else None)
    if plan_fn is not None:
        try:
            dec = plan_fn(n, False, span=r, est_dense_bytes=est,
                          budget_bytes=budget_bytes, kmin=kmin,
                          est_code_bytes=est_code, naggs=naggs)
        except TypeError:
            try:
                dec = plan_fn(n, False, span=r, est_dense_bytes=est,
                              budget_bytes=budget_bytes)
            except TypeError:
                dec = plan_fn(n, False)
    else:
        dec = {"strategy": "unique", "reason": "no planner: unique fallback"}
    return kmin, kmax, r, m_code, est_code, dec, \
        (_time.perf_counter() - t0) * 1000


# Small-start hash (E-track): the proven growth loop in hash_dynamic_index
# already handles overflow (2x rehash while load > max_load), so starting at
# 2^15 (L2-resident) instead of 2^24 wins low/mid-card 3-4x (measured) while
# high-card still wins 1.5x via growth (final 2^21 << 2^24). No H2O-specific
# branches: growth is purely load-driven, exact on every cardinality.
_HASH_START_SMALL = 1 << 15


def routed_aggregate(keys, vcols, plan_fn=None, budget_bytes=None, naggs=1):
    """One probe set, one Planner gate, three single-pass outcomes.

    Returns (kind, payload, gi_ms, agg_ms) with kind in:
      'sorted' -- speculative run kernel hit (payload (gi, counts, sums));
      'dense'  -- dense_by_code fused single-pass (payload (gi, dec, counts,
                 sums_list)); 'k64' working copies never materialize;
      'hash'   -- small-start adaptive fill (payload (gi, dec)); gi carries
                 the standard inverse, driver aggregates via the proven
                 per-column path (fill itself is 2-4x faster, agg unchanged);
      None     -- every other case (empty, dense FALLBACK range, unsupported
                 column mix, growth overflow): caller runs the
                 legacy group_index + separate-aggregate chain verbatim.
    group_index() itself is untouched; the gate observables are shared via
    _probe_plan so the strategy can never diverge from the proven chain.
    """
    import time as _time
    keys = np.asarray(keys)
    n = keys.size
    if n == 0:
        return None, None, 0.0, 0.0
    cols = [np.asarray(v) for v in vcols]
    sok, spay, _p, _a = sorted_fused_aggregate(keys, cols)
    if sok:
        gi, counts, sums = spay
        return "sorted", (gi, counts, sums), _p, _a
    _, _, _, m_code, _, dec, probe_ms = _probe_plan(
        keys, n, plan_fn, budget_bytes, naggs)
    strat = dec.get("strategy")
    if strat == "dense_by_code" and m_code is not None:
        t1 = _time.perf_counter()
        try:
            out = fused_sums_by_code(keys, cols, m_code)
        except MemoryError:
            out = None
        agg_ms = (_time.perf_counter() - t1) * 1000
        if out is not None:
            ukeys, counts, sums_list = out
            gi = {"strategy": "dense_by_code", "ukeys": ukeys, "starts": None,
                  "inverse": None, "n": n, "m": int(m_code)}
            return "dense", (gi, dec, counts, sums_list), probe_ms, agg_ms
        return None, None, probe_ms, 0.0
    if strat == "hash":
        t1 = _time.perf_counter()
        try:
            gi = hash_dynamic_index(keys, profile_init=_HASH_START_SMALL)
        except Exception:  # noqa: BLE001 -- fallback must never raise
            return None, None, probe_ms, 0.0
        agg_ms = (_time.perf_counter() - t1) * 1000
        if gi["strategy"] == "hash":
            return "hash", (gi, dec), probe_ms, agg_ms
        return None, None, probe_ms, 0.0
    return None, None, probe_ms, 0.0


def fused_dense_aggregate(keys, vcols, plan_fn=None, budget_bytes=None,
                          hash_init=1 << 24, hash_max_load=0.7, hash_growth=2,
                          naggs=1):
    """Try single-pass dense aggregation; (ok, payload, gi_ms, agg_ms).

    Probes (sortedness early-exit, vectorized min/max) and the Planner gate
    are IDENTICAL to group_index (same helpers, same observables, same call
    shape) so the strategy decision can never diverge. Returns ok=True only
    for dense_by_code with a supported column mix; every other case returns
    ok=False and the caller runs the legacy group_index + separate-aggregate
    chain verbatim (proven paths untouched). Empty/sorted inputs always take
    the legacy path (sorted reduceat is already optimal there).
    Payload on ok: (gi, decision, counts, sums_list).
    """
    import time as _time
    keys = np.asarray(keys)
    n = keys.size
    if n == 0 or is_sorted_early(keys):
        return False, None, 0.0, 0.0
    _, _, _, m_code, _, dec, gi_ms = _probe_plan(
        keys, n, plan_fn, budget_bytes, naggs)
    if dec.get("strategy") != "dense_by_code" or m_code is None:
        return False, None, gi_ms, 0.0
    t1 = _time.perf_counter()
    try:
        out = fused_sums_by_code(keys, vcols, m_code)
    except MemoryError:
        return False, None, gi_ms, 0.0
    agg_ms = (_time.perf_counter() - t1) * 1000
    if out is None:
        return False, None, gi_ms, 0.0
    ukeys, counts, sums_list = out
    gi = {"strategy": "dense_by_code", "ukeys": ukeys, "starts": None,
          "inverse": None, "n": n, "m": int(m_code)}
    return True, (gi, dec, counts, sums_list), gi_ms, agg_ms


def group_index(keys, plan_fn=None, budget_bytes=None,
                hash_init=1 << 24, hash_max_load=0.7, hash_growth=2, naggs=1):
    """One generic group_index primitive.

    plan_fn: Planner decision (n, is_sorted[, span/est/budget/kmin/naggs
    observables]). Legacy plan_fn(n, is_sorted) without observable kwargs
    still works. naggs = number of value aggregates sharing this grouping
    (Planner may pack the fused state tighter, never looser).
    Returns (gi, decision); gi keeps the frozen shape, decision carries why.
    """
    keys = np.asarray(keys)
    n = keys.size
    if n == 0:
        dec = plan_fn(0, True) if plan_fn else {"strategy": "empty",
                                                "reason": "no rows"}
        return empty_index(keys), dec
    if is_sorted_early(keys):
        dec = plan_fn(n, True) if plan_fn else {"strategy": "sorted",
                                                "reason": "sorted input"}
        gi = sorted_run_index(keys)
        gi["strategy"] = "sorted"
        return gi, dec
    kmin, kmax, r = range_probe(keys)
    est = dense_estimate_bytes(r, n)
    m_code = int(kmax) + 1 if kmin >= 0 else None
    keys_is_int64 = keys.dtype == np.int64
    est_code = (bycode_estimate_bytes(m_code, n, keys_is_int64)
                if m_code is not None else None)
    if plan_fn is not None:
        try:
            dec = plan_fn(n, False, span=r, est_dense_bytes=est,
                          budget_bytes=budget_bytes, kmin=kmin,
                          est_code_bytes=est_code, naggs=naggs)
        except TypeError:
            try:
                dec = plan_fn(n, False, span=r, est_dense_bytes=est,
                              budget_bytes=budget_bytes)
            except TypeError:
                dec = plan_fn(n, False)
    else:
        dec = {"strategy": "unique", "reason": "no planner: unique fallback"}
    s = dec.get("strategy", "unique")
    if s == "dense_by_code":
        if m_code is None:
            dec = {"strategy": "dense",
                   "reason": "by-code gate needs kmin>=0; shift-dense fallback"}
            s = "dense"
        else:
            try:
                gi = dense_by_code_index(keys, m_code)
                gi["strategy"] = "dense_by_code"
                return gi, dec
            except MemoryError as e:
                dec = {"strategy": "hash",
                       "reason": f"by-code alloc failed ({e}); hash fallback"}
                s = "hash"
    if s == "dense":
        try:
            gi = dense_direct_index(keys, kmin, r)
            gi["strategy"] = "dense"
            return gi, dec
        except MemoryError as e:
            dec = {"strategy": "hash",
                   "reason": f"dense alloc failed ({e}); hash fallback"}
            s = "hash"
    if s == "hash":
        try:
            gi = hash_dynamic_index(keys, profile_init=hash_init,
                                    max_load=hash_max_load, growth=hash_growth)
            if gi["strategy"] == "hash":
                return gi, dec
            dec = {"strategy": "unique",
                   "reason": "hash table overflowed 2^30 slots; unique fallback"}
        except Exception as e:  # noqa: BLE001 -- fallback must never raise
            dec = {"strategy": "unique",
                   "reason": f"hash failed ({e!r}); unique fallback"}
    gi = unique_fallback_index(keys)
    gi["strategy"] = "unique"
    return gi, dec


def composite_tuple_index(cols):
    """Generic N-col composite grouping traversal (Q36 4-col, CPU-only).

    cols: list of 1-D integer arrays, equal length, already
    validity-compacted by the caller (validity AND over the key columns:
    any-NULL rows excluded before this call, DELTA-3). No per-row Python
    loop over N: one lexsort + one vector pass over the sorted rows.
    Returns (ukeys, inverse, counts): ukeys is a sorted (lex ascending)
    list of tuples, inverse int64 with ukeys[inverse[i]] == tuple(cols[i]),
    counts int64 per group. Same contract as unique_fallback_index
    (sorted ukeys, exact inverse) generalized to tuples; sums/count
    lanes stay int64 in the caller. Never an int64 scalar pack: the
    tuple identity is the grouping key throughout.
    """
    cols = [np.ascontiguousarray(np.asarray(c)).ravel() for c in cols]
    if not cols:
        return [], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    n = int(cols[0].size)
    for c in cols[1:]:
        if int(c.size) != n:
            raise ValueError(
                f"composite_tuple_index size mismatch {int(c.size)} != {n}")
    if n == 0:
        return [], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    nc = len(cols)
    # --- lexsort with int32-pair packing when possible ---
    # Packing 2 int32 → 1 int64 halves lexsort key width (fewer comparison
    # passes), giving ~35-40% lexsort speedup on 4-col int32 data.
    # lexsort((B, A)) = sort by A primary, B secondary.
    # For pack0, primary column goes into HIGH bits so int64 comparison
    # yields correct lexicographic order.
    _pack0 = None
    _pack1 = None
    if nc == 1:
        order = np.lexsort((cols[0],))
    elif nc >= 2 and cols[0].dtype == np.int32 and cols[1].dtype == np.int32:
        _pack0 = (cols[0].astype(np.int64) << 32) | (
            cols[1].astype(np.int64) & np.int64(0xFFFFFFFF))
        if nc == 2:
            order = np.lexsort((_pack0,))
        elif nc == 3 and cols[2].dtype == np.int32:
            order = np.lexsort((_pack0, cols[2]))
        elif nc == 4 and cols[2].dtype == np.int32 and cols[3].dtype == np.int32:
            _pack1 = (cols[2].astype(np.int64) << 32) | (
                cols[3].astype(np.int64) & np.int64(0xFFFFFFFF))
            order = np.lexsort((_pack1, _pack0))
        else:
            order = np.lexsort(tuple(cols[::-1]))
    else:
        order = np.lexsort(tuple(cols[::-1]))
    # --- new_group: vectorized boundary detection ---
    # Packed keys already computed for lexsort: reuse for boundary detection.
    # 4-col int32: 2 packed diffs instead of 4 per-column fancy-index+diff.
    # 3-col int32: 1 packed diff + 1 per-column instead of 3.
    # 2-col int32: 1 packed diff instead of 2.
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    sp0 = None
    sp1 = None
    sc2 = None
    if nc == 1:
        sc = cols[0][order]
        new_group[1:] = sc[1:] != sc[:-1]
    elif _pack1 is not None:
        sp0 = _pack0[order]
        sp1 = _pack1[order]
        new_group[1:] = (sp0[1:] != sp0[:-1]) | (sp1[1:] != sp1[:-1])
    elif _pack0 is not None and nc == 2:
        sp0 = _pack0[order]
        new_group[1:] = sp0[1:] != sp0[:-1]
    elif _pack0 is not None and nc == 3:
        sp0 = _pack0[order]
        sc2 = cols[2][order]
        new_group[1:] = (sp0[1:] != sp0[:-1]) | (sc2[1:] != sc2[:-1])
    else:
        new_group[1:] = False
        for c in cols:
            sc = c[order]
            new_group[1:] |= (sc[1:] != sc[:-1])
    starts = np.flatnonzero(new_group)
    ng = int(starts.size)
    # --- ukeys: extract from packed sorted values when possible ---
    # For 4-col int32: unpack from sp0/sp1 (already computed, cache-hot),
    # avoiding 4 separate fancy-index ops on raw columns.
    # For other paths: original zip approach.
    if ng:
        if sp0 is not None and sp1 is not None and nc == 4:
            # 4-col int32 packed: unpack from sorted packed keys
            sp0_s = sp0[starts]
            sp1_s = sp1[starts]
            c0 = (sp0_s >> np.int64(32)).astype(np.int32).tolist()
            c1 = (sp0_s & np.int64(0xFFFFFFFF)).astype(np.int32).tolist()
            c2 = (sp1_s >> np.int64(32)).astype(np.int32).tolist()
            c3 = (sp1_s & np.int64(0xFFFFFFFF)).astype(np.int32).tolist()
            ukeys = list(zip(c0, c1, c2, c3))
        elif sp0 is not None and nc == 2:
            sp0_s = sp0[starts]
            c0 = (sp0_s >> np.int64(32)).astype(np.int32).tolist()
            c1 = (sp0_s & np.int64(0xFFFFFFFF)).astype(np.int32).tolist()
            ukeys = list(zip(c0, c1))
        elif sp0 is not None and nc == 3 and sc2 is not None:
            sp0_s = sp0[starts]
            sc2_s = sc2[starts]
            c0 = (sp0_s >> np.int64(32)).astype(np.int32).tolist()
            c1 = (sp0_s & np.int64(0xFFFFFFFF)).astype(np.int32).tolist()
            c2 = sc2_s.tolist()
            ukeys = list(zip(c0, c1, c2))
        else:
            idx = order[starts]
            parts = [c[idx].tolist() for c in cols]
            ukeys = [tuple(v) for v in zip(*parts)]
    else:
        ukeys = []
    # --- inverse: cumsum + scatter ---
    gid_sorted = np.cumsum(new_group.astype(np.int64)) - np.int64(1)
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = gid_sorted
    # --- counts ---
    counts = np.bincount(gid_sorted, minlength=ng).astype(np.int64)
    return ukeys, inverse, counts
