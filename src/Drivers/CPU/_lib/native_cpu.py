# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Native CPU capability: Rust primitives (ctypes, no PyO3) + numpy fallback.

STANDALONE module: only numpy/ctypes stdlib imports, no Builder/Extension
imports (NO INTERNAL IMPORTS rule). Production path untouched — research
scripts load this file directly by path until the primitives are proven;
manifest wiring (alias/mods/setup) happens only after proof.

Contract: every wrapper has identical outputs with backend available or
not (available=False never breaks correctness, only speed). Backend
absence (missing DLL, NUMFAST_NATIVE_DISABLE=1) -> numpy fallback.
"""

# ONE documented exception to the NO-INTERNAL-IMPORTS rule above:
# `_default_native_dll` imports numfast._lib.native_env LAZILY, inside the
# function body, because an eager module-scope import runs while
# numfast/__init__ is still executing and numfast._lib/__init__ eagerly builds
# Series/Table -- a cycle. The rule exists so this file stays loadable BY PATH;
# a lazy import cannot break that, and without it this file's only candidate on
# a non-Windows host is a Windows path.


import ctypes
import os

import numpy as np

def _default_native_dll():
    """The native binary this PLATFORM may load, or the legacy name if unreachable.

    This module is STANDALONE -- research scripts load it directly by path, with
    no `numfast` package booted -- so the import is INSIDE the function. At
    module scope it would run while `numfast/__init__` is still executing (this
    file is what the boot imports), and `numfast._lib.__init__` eagerly builds
    Series/Table: an eager import from here is a cycle. A lazy one is not -- by
    the time any of this runs the package is either fully imported or not
    imported at all, and both are safe.

    The ImportError arm is the research-script case: this file loaded by path
    with `numfast` not importable at all. There the literal is still the best
    available answer, and keeping it is strictly better than raising at import.
    """
    try:
        from numfast._lib.native_env import default_for
    except ImportError:
        return _LEGACY_DLL_DEFAULT
    return default_for(__file__) or _LEGACY_DLL_DEFAULT


#: Where the binary is: a Windows checkout path, kept ONLY as the last resort.
_LEGACY_DLL_DEFAULT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "numfast-native", "target",
    "x86_64-pc-windows-gnu", "release", "numfast_native.dll"))

_DLL_DEFAULT = _default_native_dll()

_lib = None
_why = "unprobed"
_probe_env = None  # memoized (disable-flag, dll-path); same semantics, no CDLL reload
_lib_gen = 0  # monotonic generation of the loaded CDLL; see _abi_key
_select_ok = None  # memoized select/mask entry points (new ABI, optional until rebuilt)
_shift_ok = None  # memoized shift entry points (same optional-ABI discipline)
_map_ok = None  # memoized map entry points (same optional-ABI discipline)
_cumsum_ok = None  # memoized cumsum entry points (same optional-ABI discipline)


def _abi_key():
    """Memo key for every optional-ABI memo: WHICH loaded object it describes.

    argtypes/restype live ON the CDLL object (they are attributes of the
    _FuncPtr entries ctypes caches in the CDLL's own __dict__), so such a memo
    is valid only for the exact object it configured. `_probe_env` -- the
    disable flag plus the dll path -- is NOT that identity: a disable/restore
    cycle rebuilds a FRESH CDLL under the SAME environment key, so an
    env-keyed memo handed back an un-argtyped CDLL and every 64-bit pointer
    went through as a C int (truncated, sign-extended) into Rust -> SIGSEGV.
    That is why the key is the object and not the environment.

    `_lib_gen` is bumped on every rebind of `_lib` and never repeats within one
    process, so a key recorded against an older CDLL can never compare equal to
    a live one. `id(_lib)` does not have that property -- CPython reuses the
    address of a freed object, so a new CDLL can land on the old one's id and a
    stale memo would match again.

    One key, one mechanism: every `_req_*` optional-ABI memo uses this, so a
    memo added later cannot forget to be invalidated.
    """
    return (_lib_gen, _lib is not None)


def _probe():
    global _lib, _why, _probe_env, _lib_gen
    global _select_ok, _shift_ok, _map_ok, _cumsum_ok
    global _rng_ok
    key = (os.environ.get("NUMFAST_NATIVE_DISABLE"),
           os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT))
    if key == _probe_env and _why != "unprobed":
        return
    _probe_env = key
    # Reaching this point IS the rebind: a fresh CDLL, or None. Bump the
    # generation so every `_req_*` memo keyed by _abi_key() re-probes against
    # the new object instead of returning its stale verdict.
    _lib_gen += 1
    _select_ok = None  # DLL identity changed -> re-probe select entry points
    _shift_ok = None  # DLL identity changed -> re-probe shift entry points
    _map_ok = None  # DLL identity changed -> re-probe map entry points
    _cumsum_ok = None  # DLL identity changed -> re-probe cumsum entry points
    # argtypes are set on the CDLL OBJECT, so a new CDLL has none. Without
    # this reset `_rng_ok` stays True and `_req_rng()` hands out the fresh
    # un-argtyped lib: every RNG call then raises ctypes.ArgumentError
    # ("int too long to convert"), which the driver's `except RuntimeError`
    # does not catch. One line, argument-type plumbing, no semantics.
    # This memo has no key of its own, so it is invalidated here by hand; the
    # keyed memos get the same guarantee from _abi_key() and must NOT be listed
    # here as well -- a hand reset that left its stale _KEY would pin them None.
    _rng_ok = None  # DLL identity changed -> re-probe RNG entry points
    if os.environ.get("NUMFAST_NATIVE_DISABLE") == "1":
        _lib, _why = None, "disabled-by-env"
        return
    path = os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT)
    try:
        lib = ctypes.CDLL(path)
        lib.nf_group_sum_count.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
        lib.nf_group_sum_count.restype = ctypes.c_int32
        lib.nf_group_multi_sum_count.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 2 + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
        lib.nf_group_multi_sum_count.restype = ctypes.c_int32
        lib.nf_pack_i32_direct.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]
        lib.nf_pack_i32_direct.restype = ctypes.c_int32
        lib.nf_pattern_encode.argtypes = [ctypes.c_void_p, ctypes.c_size_t] * 3 + [ctypes.c_void_p] * 4
        lib.nf_pattern_encode.restype = ctypes.c_int32
        lib.nf_sorted_run_i64.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
        lib.nf_sorted_run_i64.restype = ctypes.c_int64
        lib.nf_sorted_run_f64.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
        lib.nf_sorted_run_f64.restype = ctypes.c_int64
        lib.nf_carry_build_i64.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
        lib.nf_carry_build_i64.restype = ctypes.c_int64
        lib.nf_carry_build_f64.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
        lib.nf_carry_build_f64.restype = ctypes.c_int64
        _lib, _why = lib, "loaded:" + path
    except (OSError, AttributeError) as e:
        _lib, _why = None, "unavailable:%s" % e


def available():
    """True when the Rust backend is loaded (call every time: env-aware)."""
    _probe()
    return _lib is not None


def why():
    _probe()
    return _why


def _req_native():
    _probe()
    if _lib is None:
        raise RuntimeError("native_cpu backend %s" % _why)
    return _lib


def _req_select():
    """Select/mask entry points (new ABI): lib or None (fallback owns it).

    Optional until the DLL is rebuilt with select kernels: missing symbols
    disable only select/mask native paths, never the proven groupby/pack
    paths (no whole-backend unavailable cascade).
    """
    global _select_ok
    _probe()
    if _lib is None:
        return None
    if _select_ok is None:
        try:
            _lib.nf_select_count.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            _lib.nf_select_count.restype = ctypes.c_int64
            _lib.nf_select_scatter_i32.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t, ctypes.c_void_p]
            _lib.nf_select_scatter_i32.restype = ctypes.c_int32
            _lib.nf_select_scatter_f32.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t, ctypes.c_void_p]
            _lib.nf_select_scatter_f32.restype = ctypes.c_int32
            _lib.nf_select_scatter_f64.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t, ctypes.c_void_p]
            _lib.nf_select_scatter_f64.restype = ctypes.c_int32
            _lib.nf_select_scatter_i64.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t, ctypes.c_void_p]
            _lib.nf_select_scatter_i64.restype = ctypes.c_int32
            _lib.nf_select_scatter_u8.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t, ctypes.c_void_p]
            _lib.nf_select_scatter_u8.restype = ctypes.c_int32
            _lib.nf_mask_and.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t, ctypes.c_void_p]
            _lib.nf_mask_and.restype = ctypes.c_int32
            _lib.nf_mask_or.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t, ctypes.c_void_p]
            _lib.nf_mask_or.restype = ctypes.c_int32
            _lib.nf_mask_not.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            _lib.nf_mask_not.restype = ctypes.c_int32
            # Touch one symbol to force lazy Windows lookup now, not mid-query.
            _ = _lib.nf_select_count
            _select_ok = True
        except (OSError, AttributeError, ValueError):
            _select_ok = False
    return _lib if _select_ok else None


def select_available():
    """True when select/mask native entry points are present (env-aware)."""
    return _req_select() is not None


# ---------------- numpy fallbacks (correctness reference) ----------------

def _fb_sum_count(keys, vals, g):
    return (np.bincount(keys, weights=vals, minlength=g).astype(np.float64),
            np.bincount(keys, minlength=g).astype(np.int64))


def _fb_multi(keys, cols, g):
    sums = [np.bincount(keys, weights=c, minlength=g) for c in cols]
    return sums, np.bincount(keys, minlength=g).astype(np.int64)


def _fb_pack(k1, k2, m2):
    out = np.empty(k1.size, dtype=np.int32)
    np.multiply(k1, np.int32(m2), out=out)
    np.add(out, k2, out=out)
    return out


def _fb_pattern_buffers(data, offs, prefix):
    n = offs.size - 1
    codes = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=np.uint8)
    ends = np.cumsum(np.diff(offs.astype(np.int64)))
    rows = np.split(np.frombuffer(data, dtype=np.uint8), ends[:-1])
    p = prefix.encode() if isinstance(prefix, str) else bytes(prefix)
    width = -1
    for i, r in enumerate(rows):
        b = bytes(r)
        if not b.startswith(p):
            continue
        body = b[len(p):]
        neg = body.startswith(b"-")
        core = body[1:] if neg else body
        if not core or any(not (48 <= x <= 57) for x in core):
            continue
        mag = int(core)
        if mag > (2**31 if neg else 2**31 - 1):
            raise OverflowError(i)
        codes[i] = -mag if neg else mag
        valid[i] = 1
        width = max(width, len(core))
    return codes, valid, width


def _fb_sorted(keys, vals):
    uk, starts = np.unique(keys, return_index=True)
    n = keys.size
    return uk.astype(np.int64), np.add.reduceat(vals, starts), np.diff(np.append(starts, n)).astype(np.int64)


def _fb_select(values, eff):
    values = np.asarray(values)
    eff = np.asarray(eff, dtype=bool)
    if eff.size != values.size:
        raise ValueError(
            f"filter values/mask size mismatch {values.size} != {eff.size}")
    return values[eff]


def _fb_mask_and(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype != np.dtype(bool) or b.dtype != np.dtype(bool):
        raise TypeError("mask input must be bool")
    if a.size != b.size:
        raise ValueError(f"mask size mismatch {a.size} != {b.size}")
    return a & b


def _fb_mask_or(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype != np.dtype(bool) or b.dtype != np.dtype(bool):
        raise TypeError("mask input must be bool")
    if a.size != b.size:
        raise ValueError(f"mask size mismatch {a.size} != {b.size}")
    return a | b


def _fb_mask_not(a):
    a = np.asarray(a)
    if a.dtype != np.dtype(bool):
        raise TypeError("mask input must be bool")
    return ~a


def _fb_carry(counts_m, sums_m):
    pos = np.flatnonzero(counts_m)
    return pos.astype(np.int64), counts_m[pos], sums_m[pos]


def resolve_on_worse(native_ms, alt_native_ms=None, fallback_ms=None):
    """Native-worse policy (pure, no hardcode): native -> alt-native -> fallback.

    Caller measures (or reuses gate numbers); this only orders attempts:
    returns 'native' if it is fastest or no measurement says otherwise,
    else 'alt' when an alternate native algorithm (e.g. carry vs fused,
    sorted vs dense) beats both native-direct and the fallback, else
    'fallback'. No thresholds, no dataset names.
    """
    cands = {"native": native_ms}
    if alt_native_ms is not None:
        cands["alt"] = alt_native_ms
    if fallback_ms is not None:
        cands["fallback"] = fallback_ms
    return min(cands, key=lambda k: cands[k])


# Variant entry points (int families, additive research ABI in one DLL).
# V_DENSE=0: i64 = unchecked dense scatter (checked add); i32 = range-checked
# scatter (checked add). rc: 0 ok, -2 BAD_RANGE, -3 BAD_VARIANT, -4 OVERFLOW.
_V_LIB_OK = None
_V_LIB_KEY = None


def _req_variant():
    """Variant int kernels (additive ABI): lib or None (fallback owns it).

    Never disables the proven groupby/pack paths: missing symbols only
    disable int-direct lanes, never the f64 backend.
    """
    global _V_LIB_OK, _V_LIB_KEY
    _probe()
    key = _abi_key()
    if _V_LIB_KEY == key:
        return _lib if _V_LIB_OK else None
    _V_LIB_KEY = key
    _V_LIB_OK = False
    if _lib is None:
        return None
    try:
        _lib.nf_group_variant_i64.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _lib.nf_group_variant_i64.restype = ctypes.c_int32
        _lib.nf_group_variant_i32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _lib.nf_group_variant_i32.restype = ctypes.c_int32
        _ = _lib.nf_group_variant_i64  # force Windows lookup now
        _V_LIB_OK = True
    except (OSError, AttributeError, ValueError):
        _V_LIB_OK = False
    return _lib if _V_LIB_OK else None


def variant_available():
    """True when int-direct variant kernels are present (env-aware)."""
    return _req_variant() is not None


def i32_gate_ok(vals, n=None, _b=None):
    """int32-direct gate on precomputed (mn,mx) or one C-speed probe.

    Conservative: every group sum <= n*maxabs (all rows in one group),
    so n*maxabs < 2**31-1 proves no int32 lane can overflow. No dataset
    names, no cardinalities: pure (n, bounds) observables.
    """
    vals = np.ascontiguousarray(vals, dtype=np.int32) \
        if np.asarray(vals).dtype != np.dtype(np.int32) else np.ascontiguousarray(vals)
    if _b is None:
        if vals.size == 0:
            return True
        try:
            mn, mx = int(vals.min()), int(vals.max())
        except (ValueError, OverflowError):
            return False
    else:
        mn, mx = _b
    nn = vals.size if n is None else int(n)
    return nn * max(abs(mn), abs(mx)) < 2 ** 31 - 1


def _fb_i32(keys, vals, g):
    sums = np.bincount(keys, weights=vals.astype(np.int64),
                       minlength=g).astype(np.int32)
    return sums, np.bincount(keys, minlength=g).astype(np.int64)


def _fb_i64(keys, ticks, g):
    return (np.bincount(keys, weights=ticks.astype(np.int64),
                        minlength=g).astype(np.int64),
            np.bincount(keys, minlength=g).astype(np.int64))


def sum_count_i32(keys, vals, g):
    """int32-direct dense aggregate: zero-copy int32 in/out, exact ints.

    Overflow is an explicit OverflowError (never wraps); out-of-range key
    raises RuntimeError. Backend absent -> exact numpy fallback.
    """
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    vals = np.ascontiguousarray(vals, dtype=np.int32)
    lib = _req_variant()
    if lib is None:
        return _fb_i32(keys, vals, g)
    sums = np.zeros(g, dtype=np.int32)
    counts = np.zeros(g, dtype=np.int64)
    rc = lib.nf_group_variant_i32(keys.ctypes.data, vals.ctypes.data,
                                  keys.size, 0, sums.ctypes.data,
                                  counts.ctypes.data, g)
    if rc == -4:
        raise OverflowError("sum_count_i32: int32 accumulator overflow")
    assert rc == 0, rc
    return sums, counts


def sum_count_i64(keys, ticks, g):
    """int64-direct dense aggregate: zero-copy int64 in/out, exact ints.

    Same contract as sum_count_i32 with int64 lanes (no 2**53 caveat).
    """
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    ticks = np.ascontiguousarray(ticks, dtype=np.int64)
    lib = _req_variant()
    if lib is None:
        return _fb_i64(keys, ticks, g)
    sums = np.zeros(g, dtype=np.int64)
    counts = np.zeros(g, dtype=np.int64)
    # V_CHECKED (1), never V_DENSE (0): variant=0 reaches dense_scatter_i64
    # (scaled.rs:28-48) which indexes sums[widen_usize(keys[i])] with no
    # key_in_range, so one out-of-range dense key -- and widen_usize
    # sign-extends, so a NEGATIVE key becomes usize::MAX -- is a Rust
    # panic, and Cargo.toml sets panic="abort", so the whole CPython
    # process dies (0xC0000409, no traceback, no finally). variant=1
    # reaches checked_scatter_i64 (scaled.rs:53-76) which returns
    # BAD_RANGE (-2) instead. Same loop, one #[inline(always)] compare.
    # The in-range result is bit-identical (integer adds commute).
    rc = lib.nf_group_variant_i64(keys.ctypes.data, ticks.ctypes.data,
                                  keys.size, 1, sums.ctypes.data,
                                  counts.ctypes.data, g)
    if rc == -4:
        raise OverflowError("sum_count_i64: int64 accumulator overflow")
    if rc == -2:
        raise RuntimeError("sum_count_i64: dense key outside [0, %d)" % g)
    assert rc == 0, rc
    return sums, counts


def mixed_sum_count(keys, cols, g, _bounds=None):
    """Per-column dtype routing, zero conversions on the int fast lane.

    int32 col + i32 gate (bounds from caller probe or one C scan) ->
    int32-direct kernel (zero-copy); other int kinds -> int64-direct
    (int64 zero-copy, narrower widened once); floats -> proven f64 path.
    One key pass per column (measured: narrow passes beat 1 strided
    fused pass + conversions on low-card; bandwidth-tied on high-card).
    Any lane failure (overflow/range/backend) -> None: caller runs the
    proven f64 multi path verbatim. Returns ([sums...], counts) with
    native per-col dtypes, or None.
    """
    try:
        keys = np.ascontiguousarray(keys, dtype=np.int32)
        raw = [np.ascontiguousarray(c) for c in cols]
        if not available() or _req_variant() is None:
            return None
        n = int(keys.size)
        outs, counts = [], None
        for i, c in enumerate(raw):
            b = None if _bounds is None else _bounds[i]
            if c.dtype == np.dtype(np.int32):
                if i32_gate_ok(c, n, _b=b):
                    s, cc = sum_count_i32(keys, c, g)
                else:
                    # Beyond int32 lanes but int64-proven by caller gate:
                    # widen once, exact int64 lane (never f64 lossy).
                    s, cc = sum_count_i64(
                        keys, np.ascontiguousarray(c, dtype=np.int64), g)
            elif c.dtype.kind in "iu":
                if c.dtype != np.dtype(np.int64):
                    c = np.ascontiguousarray(c, dtype=np.int64)
                s, cc = sum_count_i64(keys, c, g)
            elif c.dtype.kind == "f":
                c = np.ascontiguousarray(c, dtype=np.float64)
                s, cc = fused_sum_count(keys, c, g)
            else:
                return None
            outs.append(s)
            if counts is None:
                counts = cc
        return outs, counts
    except (OverflowError, RuntimeError, AssertionError, MemoryError,
            ValueError):
        return None


# P2 (sum-only) entry points: additive research ABI in the same DLL
# (nf_group_mixed_sum_only + nf_group_count_only). Missing symbols (old
# DLL) -> None: caller runs the proven path verbatim. Same memo pattern
# as _req_variant (first touch configures; warm single-threaded).
_MIX_LIB_OK = None
_MIX_LIB_KEY = None


def _req_mixed():
    """Sum-only kernels (additive ABI): lib or None (fallback owns it).

    Never disables the proven groupby/pack paths: missing symbols only
    disable the sum-only fast lane, never the f64/int backends.
    """
    global _MIX_LIB_OK, _MIX_LIB_KEY
    _probe()
    key = _abi_key()
    if _MIX_LIB_KEY == key:
        return _lib if _MIX_LIB_OK else None
    _MIX_LIB_KEY = key
    _MIX_LIB_OK = False
    if _lib is None:
        return None
    try:
        _lib.nf_group_mixed_sum_only.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _lib.nf_group_mixed_sum_only.restype = ctypes.c_int32
        _lib.nf_group_count_only.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
            ctypes.c_size_t]
        _lib.nf_group_count_only.restype = ctypes.c_int32
        _ = _lib.nf_group_mixed_sum_only  # force Windows lookup now
        _MIX_LIB_OK = True
    except (OSError, AttributeError, ValueError):
        _MIX_LIB_OK = False
    return _lib if _MIX_LIB_OK else None


def mixed_available():
    """True when sum-only kernels are present (env-aware)."""
    return _req_mixed() is not None


# Q5-P3 (owner-group) entry point: additive research ABI in the same DLL
# (nf_group_owner_2i32_1f64). Missing symbol (old DLL) -> False: caller
# runs the proven S-states path verbatim. Same memo pattern as _req_mixed
# (first touch configures; warm single-threaded).
_OWNER_LIB_OK = None
_OWNER_LIB_KEY = None


def _req_owner():
    """Owner-shard kernel (additive ABI): lib or None (fallback owns it).

    Never disables the proven groupby/pack paths: missing symbol only
    disables the owner-group fast lane, never the f64/int backends.
    """
    global _OWNER_LIB_OK, _OWNER_LIB_KEY
    _probe()
    key = _abi_key()
    if _OWNER_LIB_KEY == key:
        return _lib if _OWNER_LIB_OK else None
    _OWNER_LIB_KEY = key
    _OWNER_LIB_OK = False
    if _lib is None:
        return None
    try:
        _lib.nf_group_owner_2i32_1f64.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _lib.nf_group_owner_2i32_1f64.restype = ctypes.c_int32
        _ = _lib.nf_group_owner_2i32_1f64  # force Windows lookup now
        _OWNER_LIB_OK = True
    except (OSError, AttributeError, ValueError):
        _OWNER_LIB_OK = False
    return _lib if _OWNER_LIB_OK else None


def owner_available():
    """True when the owner-shard kernel is present (env-aware)."""
    return _req_owner() is not None


def owner_shard_into(keys, a, b, c, lo, hi, m, s1, s2, s3, cc):
    """One owner shard over caller-shared outputs (zero-copy, no merge).

    keys/a/b/c: int32/int32/int32/float64 (contiguous views, no copies);
    s1/s2 int32, s3 float64, cc int64 shared dense state (only lanes
    [lo, hi) written — disjoint across concurrent shards, each lane
    zeroed exactly once by its owner). True on rc==0; False on backend /
    range / overflow failure (fallback owns the query, never wraps).
    """
    try:
        lib = _req_owner()
        if lib is None:
            return False
        keys = np.ascontiguousarray(keys, dtype=np.int32)
        av = np.ascontiguousarray(a, dtype=np.int32)
        bv = np.ascontiguousarray(b, dtype=np.int32)
        cv = np.ascontiguousarray(c, dtype=np.float64)
        rc = lib.nf_group_owner_2i32_1f64(
            keys.ctypes.data, av.ctypes.data, bv.ctypes.data,
            cv.ctypes.data, keys.size, int(lo), int(hi),
            np.ascontiguousarray(s1).ctypes.data,
            np.ascontiguousarray(s2).ctypes.data,
            np.ascontiguousarray(s3).ctypes.data,
            np.ascontiguousarray(cc).ctypes.data, int(m))
        return rc == 0
    except (OverflowError, RuntimeError, AssertionError, MemoryError,
            ValueError):
        return False


# P4 (fused pack+aggregate) entry points: additive ABI in the same DLL
# (nf_pack_sum_count_i32 + nf_pack_sum_count_f64). Missing symbols (old
# DLL) -> fallback owns it: caller runs the proven pack+groupby path
# verbatim. Same memo pattern as _req_mixed (first touch configures;
# warm single-threaded).
_FUSE_LIB_OK = None
_FUSE_LIB_KEY = None


def _req_fused():
    """Fused pack+aggregate kernels (additive ABI): lib or None.

    Never disables the proven pack/groupby paths: missing symbols only
    disable the fused fast lane, never the pack or aggregate backends.
    """
    global _FUSE_LIB_OK, _FUSE_LIB_KEY
    _probe()
    key = _abi_key()
    if _FUSE_LIB_KEY == key:
        return _lib if _FUSE_LIB_OK else None
    _FUSE_LIB_KEY = key
    _FUSE_LIB_OK = False
    if _lib is None:
        return None
    try:
        _lib.nf_pack_sum_count_i32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _lib.nf_pack_sum_count_i32.restype = ctypes.c_int32
        _lib.nf_pack_sum_count_f64.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _lib.nf_pack_sum_count_f64.restype = ctypes.c_int32
        _ = _lib.nf_pack_sum_count_i32  # force Windows lookup now
        _FUSE_LIB_OK = True
    except (OSError, AttributeError, ValueError):
        _FUSE_LIB_OK = False
    return _lib if _FUSE_LIB_OK else None


def fused_available():
    """True when fused pack+aggregate kernels are present (env-aware)."""
    return _req_fused() is not None


def _fb_fused(k1, k2, m2, vals, g):
    keys = (np.ascontiguousarray(k1, dtype=np.int32).astype(np.int64)
            * np.int64(m2)
            + np.ascontiguousarray(k2, dtype=np.int32).astype(np.int64))
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    vals = np.ascontiguousarray(vals)
    if vals.dtype.kind in "iu":
        w = vals.astype(np.int64, copy=False)
        sums = np.bincount(keys, weights=w, minlength=g)
    else:
        w = vals.astype(np.float64, copy=False)
        sums = np.bincount(keys, weights=w, minlength=g)
    return sums, np.bincount(keys, minlength=g).astype(np.int64)


def pack_sum_count_i32(k1, k2, m2, vals, g):
    """Fused pack + int32 sum+count: no packed-key buffer, exact ints.

    Overflow is an explicit OverflowError (never wraps); pack/key range
    raises RuntimeError. Backend absent -> exact numpy fallback.
    """
    k1 = np.ascontiguousarray(k1, dtype=np.int32)
    k2 = np.ascontiguousarray(k2, dtype=np.int32)
    vals = np.ascontiguousarray(vals, dtype=np.int32)
    lib = _req_fused()
    if lib is None:
        s, c = _fb_fused(k1, k2, m2, vals, g)
        return s.astype(np.int32), c
    sums = np.zeros(g, dtype=np.int32)
    counts = np.zeros(g, dtype=np.int64)
    rc = lib.nf_pack_sum_count_i32(k1.ctypes.data, k2.ctypes.data,
                                   int(m2), vals.ctypes.data, k1.size,
                                   sums.ctypes.data, counts.ctypes.data, g)
    if rc == -4:
        raise OverflowError("pack_sum_count_i32: int32 accumulator overflow")
    assert rc == 0, rc
    return sums, counts


def pack_sum_count_f64(k1, k2, m2, vals, g):
    """Fused pack + float64 sum+count: no packed-key buffer.

    Same contract as pack_sum_count_i32 with float64 lanes (no 2**53
    caveat); accumulation order matches dense_scatter (bit-exact).
    """
    k1 = np.ascontiguousarray(k1, dtype=np.int32)
    k2 = np.ascontiguousarray(k2, dtype=np.int32)
    vals = np.ascontiguousarray(vals, dtype=np.float64)
    lib = _req_fused()
    if lib is None:
        return _fb_fused(k1, k2, m2, vals, g)
    sums = np.zeros(g, dtype=np.float64)
    counts = np.zeros(g, dtype=np.int64)
    rc = lib.nf_pack_sum_count_f64(k1.ctypes.data, k2.ctypes.data,
                                   int(m2), vals.ctypes.data, k1.size,
                                   sums.ctypes.data, counts.ctypes.data, g)
    assert rc == 0, rc
    return sums, counts


def count_only(keys, g):
    """Keys-only occupancy: counts[0..g], exact (native or numpy)."""
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    lib = _req_mixed()
    if lib is None:
        return np.bincount(keys, minlength=g).astype(np.int64)
    counts = np.zeros(g, dtype=np.int64)
    rc = lib.nf_group_count_only(keys.ctypes.data, keys.size,
                                 counts.ctypes.data, g)
    assert rc == 0, rc
    return counts


def mixed_sum_only_count(keys, cols, g, _bounds=None):
    """P2: decoupled counts + single sum-only pass over int+float lanes.

    One keys-only count pass plus ONE pass over all value lanes (no
    counts RMW inside the value loop). Generic over lane counts: any mix
    of int32 (i32-gated) + float columns; int64/other columns -> None
    (proven per-column path owns them, never a silent lossy route).
    i32 gate failure / overflow / range / backend -> None: caller runs
    the proven path verbatim. Returns ([sums...], counts) with native
    per-col dtypes (int32 sums stay int32, floats float64), or None.
    f64 accumulation order matches dense_scatter (bit-exact).
    """
    try:
        keys = np.ascontiguousarray(keys, dtype=np.int32)
        raw = [np.ascontiguousarray(c) for c in cols]
        lib = _req_mixed()
        if lib is None:
            return None
        n = int(keys.size)
        i32c, f64c, order = [], [], []
        for i, c in enumerate(raw):
            b = None if _bounds is None else _bounds[i]
            if c.dtype == np.dtype(np.int32):
                if not i32_gate_ok(c, n, _b=b):
                    return None
                order.append(("i", len(i32c)))
                i32c.append(c)
            elif c.dtype.kind == "f":
                order.append(("f", len(f64c)))
                f64c.append(np.ascontiguousarray(c, dtype=np.float64))
            else:
                return None
        ni, nf = len(i32c), len(f64c)
        # SoA flats (one assembly copy; measured part of the win).
        i32a = (np.ascontiguousarray(np.stack(i32c).reshape(-1))
                if ni else np.zeros(0, dtype=np.int32))
        f64a = (np.ascontiguousarray(np.stack(f64c).reshape(-1))
                if nf else np.zeros(0, dtype=np.float64))
        s32 = np.zeros(ni * g, dtype=np.int32)
        s64 = np.zeros(nf * g, dtype=np.float64)
        rc = lib.nf_group_mixed_sum_only(
            keys.ctypes.data,
            i32a.ctypes.data if ni else None,
            f64a.ctypes.data if nf else None, n, ni, nf,
            s32.ctypes.data if ni else None,
            s64.ctypes.data if nf else None, g)
        if rc != 0:
            return None
        counts = count_only(keys, g)
        outs = [None] * len(raw)
        for j, (k, ci) in enumerate(order):
            outs[j] = (s32[ci * g:(ci + 1) * g] if k == "i"
                       else s64[ci * g:(ci + 1) * g])
        return outs, counts
    except (OverflowError, RuntimeError, AssertionError, MemoryError,
            ValueError):
        return None


def fused_sum_count(keys, vals, g):
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    vals = np.ascontiguousarray(vals, dtype=np.float64)
    if not available():
        return _fb_sum_count(keys, vals, g)
    lib = _req_native()
    sums = np.zeros(g)
    counts = np.zeros(g, dtype=np.int64)
    rc = lib.nf_group_sum_count(keys.ctypes.data, vals.ctypes.data, keys.size,
                                sums.ctypes.data, counts.ctypes.data, g)
    assert rc == 0, rc
    return sums, counts


def multi_sum_count(keys, cols, g):
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    cols = [np.ascontiguousarray(c, dtype=np.float64) for c in cols]
    ncols = len(cols)
    if not available():
        return _fb_multi(keys, cols, g)
    lib = _req_native()
    vflat = np.ascontiguousarray(np.stack(cols).reshape(-1))
    sums = np.zeros(ncols * g)
    counts = np.zeros(g, dtype=np.int64)
    rc = lib.nf_group_multi_sum_count(keys.ctypes.data, vflat.ctypes.data, keys.size,
                                      ncols, sums.ctypes.data, counts.ctypes.data, g)
    assert rc == 0, rc
    return [sums[i * g:(i + 1) * g] for i in range(ncols)], counts


def select_scatter(values, eff):
    values = np.asarray(values)
    eff = np.ascontiguousarray(np.asarray(eff, dtype=bool))
    if eff.size != values.size:
        raise ValueError(
            f"filter values/mask size mismatch {values.size} != {eff.size}")
    lib = _req_select()
    if lib is None:
        return _fb_select(values, eff)
    dt = values.dtype
    if dt == np.dtype(np.int32):
        fn, out = lib.nf_select_scatter_i32, np.empty(values.size, dtype=np.int32)
        vptr = np.ascontiguousarray(values).ctypes.data
    elif dt == np.dtype(np.int64):
        fn, out = lib.nf_select_scatter_i64, np.empty(values.size, dtype=np.int64)
        vptr = np.ascontiguousarray(values).ctypes.data
    elif dt == np.dtype(np.float32):
        fn, out = lib.nf_select_scatter_f32, np.empty(values.size, dtype=np.float32)
        vptr = np.ascontiguousarray(values).ctypes.data
    elif dt == np.dtype(np.float64):
        fn, out = lib.nf_select_scatter_f64, np.empty(values.size, dtype=np.float64)
        vptr = np.ascontiguousarray(values).ctypes.data
    elif dt == np.dtype(bool):
        fn, out = lib.nf_select_scatter_u8, np.empty(values.size, dtype=np.uint8)
        vptr = np.ascontiguousarray(values.view(np.uint8)).ctypes.data
    else:
        # No native lane for this dtype (exactness first): reference owns it.
        return _fb_select(values, eff)
    m8 = eff.view(np.uint8)
    m = int(lib.nf_select_count(m8.ctypes.data, m8.size))
    assert m >= 0, m
    rc = fn(vptr, m8.ctypes.data, values.size, out.ctypes.data)
    assert rc == m, (rc, m)
    got = out[:m]
    return got.view(np.bool_) if dt == np.dtype(bool) else got


def mask_and(a, b):
    if np.asarray(a).dtype != np.dtype(bool) or np.asarray(b).dtype != np.dtype(bool):
        raise TypeError("mask input must be bool")
    a = np.ascontiguousarray(a, dtype=bool).view(np.uint8)
    b = np.ascontiguousarray(b, dtype=bool).view(np.uint8)
    if a.size != b.size:
        raise ValueError(f"mask size mismatch {a.size} != {b.size}")
    lib = _req_select()
    if lib is None:
        return _fb_mask_and(a.view(np.bool_), b.view(np.bool_))
    out = np.empty(a.size, dtype=np.uint8)
    rc = lib.nf_mask_and(a.ctypes.data, b.ctypes.data, a.size, out.ctypes.data)
    assert rc == 0, rc
    return out.view(np.bool_)


def mask_or(a, b):
    if np.asarray(a).dtype != np.dtype(bool) or np.asarray(b).dtype != np.dtype(bool):
        raise TypeError("mask input must be bool")
    a = np.ascontiguousarray(a, dtype=bool).view(np.uint8)
    b = np.ascontiguousarray(b, dtype=bool).view(np.uint8)
    if a.size != b.size:
        raise ValueError(f"mask size mismatch {a.size} != {b.size}")
    lib = _req_select()
    if lib is None:
        return _fb_mask_or(a.view(np.bool_), b.view(np.bool_))
    out = np.empty(a.size, dtype=np.uint8)
    rc = lib.nf_mask_or(a.ctypes.data, b.ctypes.data, a.size, out.ctypes.data)
    assert rc == 0, rc
    return out.view(np.bool_)


def mask_not(a):
    if np.asarray(a).dtype != np.dtype(bool):
        raise TypeError("mask input must be bool")
    a = np.ascontiguousarray(a, dtype=bool).view(np.uint8)
    lib = _req_select()
    if lib is None:
        return _fb_mask_not(a.view(np.bool_))
    out = np.empty(a.size, dtype=np.uint8)
    rc = lib.nf_mask_not(a.ctypes.data, a.size, out.ctypes.data)
    assert rc == 0, rc
    return out.view(np.bool_)


def _req_shift():
    """Shift entry points (nf_shift_i32/f32/f64): lib or None (fallback owns it).

    Same optional-ABI discipline as _req_select: missing symbols
    disable only the shift native path, never the proven select/mask
    or groupby/pack paths (no whole-backend unavailable cascade).
    """
    global _shift_ok
    _probe()
    if _lib is None:
        return None
    if _shift_ok is None:
        try:
            _lib.nf_shift_i32.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_size_t, ctypes.c_void_p]
            _lib.nf_shift_i32.restype = ctypes.c_int32
            _lib.nf_shift_f32.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_size_t, ctypes.c_void_p]
            _lib.nf_shift_f32.restype = ctypes.c_int32
            _lib.nf_shift_f64.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_size_t, ctypes.c_void_p]
            _lib.nf_shift_f64.restype = ctypes.c_int32
            # Touch one symbol to force lazy Windows lookup now, not mid-query.
            _ = _lib.nf_shift_i32
            _shift_ok = True
        except (OSError, AttributeError, ValueError):
            _shift_ok = False
    return _lib if _shift_ok else None


def shift_available():
    """True when shift native entry points are present (env-aware)."""
    return _req_shift() is not None


def _fb_shift(values, periods):
    values = np.ascontiguousarray(np.asarray(values))
    n = values.size
    out = np.zeros(n, dtype=values.dtype)
    if periods < n:
        out[periods:] = values[:n - periods]
    return out


def shift_scatter(values, periods):
    """Data-only positional right-shift (IR `shift` kernel contract).

    Validity sidecar travels host-side (same carry rule as CPU
    `_shift_ref` / GPU `shift_take`); this wrapper never sees validity.
    periods == 0 copies, periods >= n zero-fills, empty -> empty.
    int32/float32/float64 native, other dtypes -> reference owns them.
    """
    values = np.asarray(values)
    periods = int(periods)
    if periods < 0:
        raise ValueError(f"shift periods must be >= 0, got {periods}")
    dt = values.dtype
    if dt == np.dtype(np.int32):
        cdt, fn_name = np.int32, "nf_shift_i32"
    elif dt == np.dtype(np.float32):
        cdt, fn_name = np.float32, "nf_shift_f32"
    elif dt == np.dtype(np.float64):
        cdt, fn_name = np.float64, "nf_shift_f64"
    else:
        # No native lane for this dtype (exactness first): reference owns it.
        return _fb_shift(values, periods)
    src = np.ascontiguousarray(values, dtype=cdt)
    out = np.empty(src.size, dtype=cdt)
    if src.size == 0:
        return out
    lib = _req_shift()
    if lib is None:
        return _fb_shift(src, periods)
    rc = getattr(lib, fn_name)(src.ctypes.data, src.size, periods, out.ctypes.data)
    assert rc == 0, rc
    return out


def _req_cumsum():
    """Cumsum entry points (nf_cumsum_i32/f32/f64): lib or None (fallback owns it).

    Same optional-ABI discipline as _req_select/_req_shift/_req_map:
    missing symbols disable only the cumsum native path, never the
    proven select/mask/shift/map or groupby/pack paths (no
    whole-backend unavailable cascade).
    """
    global _cumsum_ok
    _probe()
    if _lib is None:
        return None
    if _cumsum_ok is None:
        try:
            _lib.nf_cumsum_i32.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.c_void_p]
            _lib.nf_cumsum_i32.restype = ctypes.c_int32
            _lib.nf_cumsum_f32.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.c_void_p]
            _lib.nf_cumsum_f32.restype = ctypes.c_int32
            _lib.nf_cumsum_f64.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.c_void_p]
            _lib.nf_cumsum_f64.restype = ctypes.c_int32
            # Touch one symbol to force lazy Windows lookup now, not mid-query.
            _ = _lib.nf_cumsum_i32
            _cumsum_ok = True
        except (OSError, AttributeError, ValueError):
            _cumsum_ok = False
    return _lib if _cumsum_ok else None


def cumsum_available():
    """True when cumsum native entry points are present (env-aware)."""
    return _req_cumsum() is not None


def _fb_cumsum(values):
    """NumPy reference for IR `cumsum` (mirrors the CPU driver branch exactly).

    Data plane only; validity travels host-side (0-fill before the
    call, same as CPU `_cumsum_ref`). int32 wraps mod 2**32 via an
    int64 accumulator; floats accumulate in their own dtype.
    """
    values = np.ascontiguousarray(np.asarray(values))
    if values.dtype == np.dtype(np.int32):
        acc = np.cumsum(values.astype(np.int64, copy=False), dtype=np.int64)
        w = acc % np.int64(2 ** 32)
        return np.ascontiguousarray(np.where(
            w >= np.int64(2 ** 31), w - np.int64(2 ** 32), w).astype(np.int32))
    return np.ascontiguousarray(np.cumsum(values, dtype=values.dtype))


def cumsum_scatter(values):
    """Data-only inclusive prefix sum (IR `cumsum` kernel contract).

    Validity sidecar travels host-side (0-fill invalid rows before
    the call + per-row carry, same rule as CPU `_cumsum_ref`); this
    wrapper never sees validity. int32 wraps mod 2**32 (never
    saturate/trap); floats accumulate in their own dtype (existing
    numerical contract). Empty -> empty. int32/float32/float64
    native, other dtypes -> reference owns them.
    """
    values = np.asarray(values)
    dt = values.dtype
    if dt == np.dtype(np.int32):
        cdt, fn_name = np.int32, "nf_cumsum_i32"
    elif dt == np.dtype(np.float32):
        cdt, fn_name = np.float32, "nf_cumsum_f32"
    elif dt == np.dtype(np.float64):
        cdt, fn_name = np.float64, "nf_cumsum_f64"
    else:
        # No native lane for this dtype (exactness first): reference owns it.
        return _fb_cumsum(values)
    src = np.ascontiguousarray(values, dtype=cdt)
    out = np.empty(src.size, dtype=cdt)
    if src.size == 0:
        return out
    lib = _req_cumsum()
    if lib is None:
        return _fb_cumsum(src)
    rc = getattr(lib, fn_name)(src.ctypes.data, src.size, out.ctypes.data)
    assert rc == 0, rc
    return out


_MAP_OPS = {"add": 0, "sub": 1, "mul": 2, "div": 3, "pow": 4,
            "floor_div": 5, "mod": 6}
_MAP_SYMBOLS = ("nf_map_i32", "nf_map_scalar_i32", "nf_map_fscalar_i32",
                "nf_map_f32", "nf_map_scalar_f32",
                "nf_map_f32_divpow", "nf_map_scalar_f32_divpow",
                "nf_map_f64", "nf_map_scalar_f64")


def _req_map():
    """Map entry points (nf_map_* x9): lib or None (fallback owns it).

    Same optional-ABI discipline as _req_select/_req_shift: missing symbols
    disable only the map native path, never the proven select/mask/shift
    or groupby/pack paths (no whole-backend unavailable cascade).
    """
    global _map_ok
    _probe()
    if _lib is None:
        return None
    if _map_ok is None:
        try:
            v = ctypes.c_void_p
            z = ctypes.c_size_t
            u = ctypes.c_uint32
            i = ctypes.c_int32
            d = ctypes.c_double
            _lib.nf_map_i32.argtypes = [v, v, z, u, v]
            _lib.nf_map_i32.restype = ctypes.c_int32
            _lib.nf_map_scalar_i32.argtypes = [v, z, i, u, v]
            _lib.nf_map_scalar_i32.restype = ctypes.c_int32
            _lib.nf_map_fscalar_i32.argtypes = [v, z, d, u, v]
            _lib.nf_map_fscalar_i32.restype = ctypes.c_int32
            _lib.nf_map_f32.argtypes = [v, v, z, u, v]
            _lib.nf_map_f32.restype = ctypes.c_int32
            _lib.nf_map_scalar_f32.argtypes = [v, z, d, u, v]
            _lib.nf_map_scalar_f32.restype = ctypes.c_int32
            _lib.nf_map_f32_divpow.argtypes = [v, v, z, u, v]
            _lib.nf_map_f32_divpow.restype = ctypes.c_int32
            _lib.nf_map_scalar_f32_divpow.argtypes = [v, z, d, u, v]
            _lib.nf_map_scalar_f32_divpow.restype = ctypes.c_int32
            _lib.nf_map_f64.argtypes = [v, v, z, u, v]
            _lib.nf_map_f64.restype = ctypes.c_int32
            _lib.nf_map_scalar_f64.argtypes = [v, z, d, u, v]
            _lib.nf_map_scalar_f64.restype = ctypes.c_int32
            # Touch one symbol to force lazy Windows lookup now, not mid-query.
            _ = _lib.nf_map_i32
            _map_ok = True
        except (OSError, AttributeError, ValueError):
            _map_ok = False
    return _lib if _map_ok else None


def map_available():
    """True when map native entry points are present (env-aware)."""
    return _req_map() is not None


def _fb_map(a, fn, b):
    """NumPy reference for IR `map` (mirrors the CPU driver branch exactly).

    Data plane only; validity travels host-side. `pow` with an array
    exponent raises the same scalar-exp error as the driver.
    """
    a = np.ascontiguousarray(np.asarray(a))
    is_array_exp = isinstance(b, np.ndarray) and b.ndim > 0
    bb = np.ascontiguousarray(b) if is_array_exp else b
    if fn == "pow" and is_array_exp:
        raise ValueError("CPU driver: map pow with array exponent: '**' scalar-exp only "
                         "(spec 01). Fix: pass a scalar exponent. "
                         "")
    if fn == "add":
        r = a + bb
    elif fn == "sub":
        r = a - bb
    elif fn == "mul":
        r = a * bb
    elif fn == "div":
        r = (a.astype(np.float64) / bb)
        if np.issubdtype(a.dtype, np.integer):
            r = np.rint(r).astype(a.dtype)
    elif fn == "pow":
        r = (a.astype(np.float64) ** bb)
        if np.issubdtype(a.dtype, np.integer):
            r = np.rint(r).astype(a.dtype)
    elif fn == "floor_div":
        r = np.floor_divide(a, bb)
    elif fn == "mod":
        r = np.remainder(a, bb)
    else:
        raise ValueError(f"CPU driver: unknown map fn '{fn}'. "
                         "Fix: use one of add/sub/mul/div/pow/floor_div/mod. "
                         "")
    return r.astype(a.dtype, copy=False) if fn in ("add", "sub", "mul") else r


def map_scatter(values, fn, other):
    """Data-only elementwise IR `map` (native when available, else reference).

    Validity sidecar travels host-side (AND of inputs, same as CPU
    `_valid_mask`); this wrapper never sees validity. Array-array lanes
    require both inputs in the lane dtype (mixed dtypes -> reference
    owns them, never a silent cast). int lane + int scalar stays
    in-lane (wrapping); int lane + float scalar rides float64 (NumPy
    value-based casting). f32 div/pow widen to float64 output (CPU
    round-trip contract). Empty -> empty (same out dtype as the CPU
    branch). Other dtypes (int64/bool/...) -> reference owns them.
    """
    if fn not in _MAP_OPS:
        raise ValueError(f"CPU driver: unknown map fn '{fn}'. "
                         "Fix: use one of add/sub/mul/div/pow/floor_div/mod. "
                         "")
    op = _MAP_OPS[fn]
    a = np.asarray(values)
    dt = a.dtype
    if dt == np.dtype(np.int32):
        # div/pow on int lanes stay int32 (rint+astype, x86 cvttsd2si rule).
        cdt, out_dt = np.int32, np.int32
    elif dt == np.dtype(np.float32):
        cdt = np.float32
        out_dt = np.float64 if fn in ("div", "pow") else np.float32
    elif dt == np.dtype(np.float64):
        cdt, out_dt = np.float64, np.float64
    else:
        # No native lane for this dtype (exactness first): reference owns it.
        return _fb_map(a, fn, other)
    if isinstance(other, str):
        return _fb_map(a, fn, other)  # array name resolved by caller (IR exec)
    if np.isscalar(other) or (isinstance(other, np.ndarray) and other.ndim == 0):
        scalar = other.item() if isinstance(other, np.ndarray) else other
        if isinstance(scalar, (bool, np.bool_)):
            scalar = int(scalar)
        is_int_scalar = isinstance(scalar, (int, np.integer)) and not isinstance(
            scalar, (float, np.floating))
        if not is_int_scalar:
            try:
                float(scalar)
            except (TypeError, ValueError, OverflowError):
                # Non-real scalar (complex/...): NumPy promotion owns it.
                return _fb_map(a, fn, scalar)
        if is_int_scalar and dt == np.dtype(np.int32) and not (
                -2 ** 31 <= int(scalar) <= 2 ** 31 - 1):
            # Out-of-lane int scalar (NumPy promotion lives here): reference.
            return _fb_map(a, fn, scalar)
        if dt == np.dtype(np.int32) and not is_int_scalar and fn in (
                "floor_div", "mod"):
            # Int lane + float scalar floor/mod widen to float64 (NumPy
            # promotion): exact i32->f64 conversion, then the f64 lane.
            aa = np.ascontiguousarray(a, dtype=np.float64)
            out = np.empty(aa.size, dtype=np.float64)
            if aa.size == 0:
                return out
            lib = _req_map()
            if lib is None:
                return _fb_map(a, fn, scalar)
            rc = lib.nf_map_scalar_f64(aa.ctypes.data, aa.size,
                                       float(scalar), op, out.ctypes.data)
            assert rc == 0, rc
            return out
        aa = np.ascontiguousarray(a, dtype=cdt)
        out = np.empty(aa.size, dtype=out_dt)
        if aa.size == 0:
            return out
        lib = _req_map()
        if lib is None:
            return _fb_map(aa, fn, scalar)
        if dt == np.dtype(np.int32):
            if is_int_scalar:
                rc = lib.nf_map_scalar_i32(aa.ctypes.data, aa.size,
                                           int(scalar), op, out.ctypes.data)
            else:
                rc = lib.nf_map_fscalar_i32(aa.ctypes.data, aa.size,
                                            float(scalar), op, out.ctypes.data)
        elif dt == np.dtype(np.float32):
            if fn in ("div", "pow"):
                rc = lib.nf_map_scalar_f32_divpow(aa.ctypes.data, aa.size,
                                                  float(scalar), op,
                                                  out.ctypes.data)
            else:
                rc = lib.nf_map_scalar_f32(aa.ctypes.data, aa.size,
                                           float(scalar), op, out.ctypes.data)
        else:
            rc = lib.nf_map_scalar_f64(aa.ctypes.data, aa.size,
                                       float(scalar), op, out.ctypes.data)
        assert rc == 0, rc
        return out
    b = np.asarray(other)
    if b.dtype != dt:
        # Mixed array dtypes (promotion+truncate lives in NumPy): reference.
        return _fb_map(a, fn, b)
    if fn == "pow":
        # Scalar-exp only (spec 01): same rejection as the CPU driver.
        raise ValueError("CPU driver: map pow with array exponent: '**' scalar-exp only "
                         "(spec 01). Fix: pass a scalar exponent. "
                         "")
    aa = np.ascontiguousarray(a, dtype=cdt)
    bb = np.ascontiguousarray(b, dtype=cdt)
    if aa.size != bb.size:
        raise ValueError(f"map length mismatch {aa.size} != {bb.size}. "
                         "Fix: align lengths before elementwise ops.")
    out = np.empty(aa.size, dtype=out_dt)
    if aa.size == 0:
        return out
    lib = _req_map()
    if lib is None:
        return _fb_map(aa, fn, bb)
    if dt == np.dtype(np.int32):
        rc = lib.nf_map_i32(aa.ctypes.data, bb.ctypes.data, aa.size,
                            op, out.ctypes.data)
    elif dt == np.dtype(np.float32):
        if fn in ("div", "pow"):
            rc = lib.nf_map_f32_divpow(aa.ctypes.data, bb.ctypes.data,
                                       aa.size, op, out.ctypes.data)
        else:
            rc = lib.nf_map_f32(aa.ctypes.data, bb.ctypes.data, aa.size,
                                op, out.ctypes.data)
    else:
        rc = lib.nf_map_f64(aa.ctypes.data, bb.ctypes.data, aa.size,
                            op, out.ctypes.data)
    assert rc == 0, rc
    return out


def pack_i32_direct(k1, k2, m2):
    k1 = np.ascontiguousarray(k1, dtype=np.int32)
    k2 = np.ascontiguousarray(k2, dtype=np.int32)
    if not available():
        return _fb_pack(k1, k2, m2)
    lib = _req_native()
    out = np.zeros(k1.size, dtype=np.int32)
    rc = lib.nf_pack_i32_direct(k1.ctypes.data, k2.ctypes.data, int(m2), k1.size, out.ctypes.data)
    assert rc == 0, rc
    return out


def build_pattern_buffers(strs):
    """strings -> (data bytes, offs int32). C-speed concat, outside timing."""
    enc = [s.encode("utf-8") if isinstance(s, str) else bytes(s) for s in strs]
    offs = np.zeros(len(enc) + 1, dtype=np.int32)
    offs[1:] = np.cumsum([len(b) for b in enc]).astype(np.int64)
    return b"".join(enc), offs


def pattern_encode_buffers(data, offs, prefix):
    n = offs.size - 1
    if not available():
        return _fb_pattern_buffers(data, offs, prefix)
    lib = _req_native()
    p = prefix.encode() if isinstance(prefix, str) else bytes(prefix)
    dp = np.frombuffer(data, dtype=np.uint8)
    pp = np.frombuffer(p, dtype=np.uint8)
    offs = np.ascontiguousarray(offs, dtype=np.int32)
    codes = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=np.uint8)
    w = np.zeros(1, dtype=np.int32)
    e = np.zeros(1, dtype=np.int32)
    rc = lib.nf_pattern_encode(dp.ctypes.data, len(data), offs.ctypes.data, n,
                               pp.ctypes.data, len(p), codes.ctypes.data,
                               valid.ctypes.data, w.ctypes.data, e.ctypes.data)
    assert rc == 0, (rc, int(e[0]))
    return codes, valid, int(w[0])


def sorted_run(keys, vals):
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    n = keys.size
    if vals.dtype.kind in "iu":
        vals = np.ascontiguousarray(vals, dtype=np.int64)
        if not available():
            return _fb_sorted(keys, vals)
        lib = _req_native()
        uk = np.zeros(n, dtype=np.int64)
        s = np.zeros(n, dtype=np.int64)
        c = np.zeros(n, dtype=np.int64)
        ng = int(lib.nf_sorted_run_i64(keys.ctypes.data, vals.ctypes.data, n,
                                       uk.ctypes.data, s.ctypes.data, c.ctypes.data))
        assert ng >= 0, ng
        return uk[:ng], s[:ng], c[:ng]
    vals = np.ascontiguousarray(vals, dtype=np.float64)
    if not available():
        return _fb_sorted(keys, vals)
    lib = _req_native()
    uk = np.zeros(n, dtype=np.int64)
    s = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.int64)
    ng = int(lib.nf_sorted_run_f64(keys.ctypes.data, vals.ctypes.data, n,
                                   uk.ctypes.data, s.ctypes.data, c.ctypes.data))
    assert ng >= 0, ng
    return uk[:ng], s[:ng], c[:ng]


def carry_build(counts_m, sums_m):
    counts_m = np.ascontiguousarray(counts_m, dtype=np.int64)
    m = counts_m.size
    if sums_m.dtype.kind in "iu":
        sums_m = np.ascontiguousarray(sums_m, dtype=np.int64)
        if not available():
            return _fb_carry(counts_m, sums_m)
        lib = _req_native()
        uk = np.zeros(m, dtype=np.int64)
        c = np.zeros(m, dtype=np.int64)
        s = np.zeros(m, dtype=np.int64)
        ng = int(lib.nf_carry_build_i64(counts_m.ctypes.data, sums_m.ctypes.data, m,
                                        uk.ctypes.data, c.ctypes.data, s.ctypes.data))
        assert ng >= 0, ng
        return uk[:ng], c[:ng], s[:ng]
    sums_m = np.ascontiguousarray(sums_m, dtype=np.float64)
    if not available():
        return _fb_carry(counts_m, sums_m)
    lib = _req_native()
    uk = np.zeros(m, dtype=np.int64)
    c = np.zeros(m, dtype=np.int64)
    s = np.zeros(m, dtype=np.float64)
    ng = int(lib.nf_carry_build_f64(counts_m.ctypes.data, sums_m.ctypes.data, m,
                                    uk.ctypes.data, c.ctypes.data, s.ctypes.data))
    assert ng >= 0, ng
    return uk[:ng], c[:ng], s[:ng]


# Unique M1 production candidate (additive, explicit-call only).
# Wraps nf_unique_inverse_i32/i64: sorted-order unique + inverse, same
# labelling as np.unique(keys, return_inverse=True) — uniq sorted
# ascending, uniq[inv] == keys. groupindex.py does NOT call this yet
# (stage 3); scratch/unique_s2.py calls it explicitly for parity/perf
# proof. NumPy stays the proven oracle (fallback owns every failure).
_U_LIB_OK = None
_U_LIB_KEY = None

# Buffer cache: dtype -> {cap, bufs}; plain resize-by-capacity (no
# super-allocator). Caller-owned outputs are COPIES out of the cache
# (gi retains ukeys/inverse long-term; views would corrupt on reuse).
_UCACHE = {}


def _req_unique():
    """Unique kernels (additive ABI): lib or None (fallback owns it).

    Never disables the proven paths: missing symbols only disable the
    unique candidate lane. Memoized by loaded-object generation (see _abi_key).
    """
    global _U_LIB_OK, _U_LIB_KEY
    _probe()
    key = _abi_key()
    if _U_LIB_KEY == key:
        return _lib if _U_LIB_OK else None
    _U_LIB_KEY = key
    _U_LIB_OK = False
    if _lib is None:
        return None
    try:
        _lib.nf_unique_inverse_i32.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p]
        _lib.nf_unique_inverse_i32.restype = ctypes.c_int64
        _lib.nf_unique_inverse_i64.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p]
        _lib.nf_unique_inverse_i64.restype = ctypes.c_int64
        _ = _lib.nf_unique_inverse_i32  # force Windows lookup now
        _U_LIB_OK = True
    except (OSError, AttributeError, ValueError):
        _U_LIB_OK = False
    return _lib if _U_LIB_OK else None


def unique_available():
    """True when unique inverse kernels are present (env-aware)."""
    return _req_unique() is not None


def _unique_bufs(n, dt):
    """Cached (uniq, inv32, perm, tmp0, tmp1, tmp_p) with cap >= n.

    Resize (realloc) only when n exceeds capacity; steady-state calls
    reuse. tmp lanes are u32 for i32 keys, u64 for i64 keys.
    """
    e = _UCACHE.get(dt)
    if e is None or e["cap"] < n:
        tu = np.uint32 if dt == np.dtype(np.int32) else np.uint64
        e = {"cap": n,
             "uniq": np.empty(n, dtype=dt),
             "inv": np.empty(n, dtype=np.int32),
             "perm": np.empty(n, dtype=np.int32),
             "tmp0": np.empty(n, dtype=tu),
             "tmp1": np.empty(n, dtype=tu),
             "tmp_p": np.empty(n, dtype=np.int32)}
        _UCACHE[dt] = e
    return (e["uniq"], e["inv"], e["perm"],
            e["tmp0"], e["tmp1"], e["tmp_p"])


def _fb_unique(keys):
    uk, inv = np.unique(keys, return_inverse=True)
    return uk, inv.astype(np.int64, copy=False)


def unique_inverse(keys):
    """Sorted-order unique + inverse candidate: (uniq, inv64, backend).

    uniq keeps keys dtype (sorted ascending); inv is int64 codes with
    uniq[inv] == keys (same convention as every groupindex strategy).
    backend is "native" only when the DLL symbol actually executed,
    else "numpy" (disabled/missing/failed -> proven oracle, exact).
    Outputs are owned copies (safe to retain past the next call).
    """
    keys = np.ascontiguousarray(np.asarray(keys))
    dt = keys.dtype
    if dt not in (np.dtype(np.int32), np.dtype(np.int64)):
        return _fb_unique(keys) + ("numpy",)
    n = keys.size
    if n == 0:
        return keys[:0], np.zeros(0, dtype=np.int64), "numpy"
    lib = _req_unique()
    if lib is None:
        return _fb_unique(keys) + ("numpy",)
    try:
        uniq, inv, perm, t0, t1, tp = _unique_bufs(n, dt)
        fn = (lib.nf_unique_inverse_i32
              if dt == np.dtype(np.int32) else lib.nf_unique_inverse_i64)
        ng = int(fn(keys.ctypes.data, n, uniq.ctypes.data,
                    inv.ctypes.data, perm.ctypes.data,
                    t0.ctypes.data, t1.ctypes.data, tp.ctypes.data))
        if ng < 0:
            return _fb_unique(keys) + ("numpy",)
        return uniq[:ng].copy(), inv[:n].copy().astype(np.int64), "native"
    except (OSError, RuntimeError, ValueError, AttributeError,
            MemoryError):
        return _fb_unique(keys) + ("numpy",)


# Sort M1 production candidate (additive, explicit-call only).
# Wraps nf_sort_perm_i32/i64 (stable LSD radix argsort perm, M1-proven)
# with the guard-dispatch M2 (per key-step, requested direction first):
# same-dir monotonic -> identity arange (exact stable perm, no DLL call);
# strict cross-monotonic -> reversed arange; non-strict cross, other
# dtypes, n==0 / n>i32max, backend absent, rc!=0 -> (None, "numpy"):
# caller runs the proven base path verbatim (np.argsort stable /
# _stable_desc_idx). cpu._sort_perm wires this per key-step (stage 3);
# scratch/sort_s2.py calls it explicitly for parity/perf proof.
# NATIVE-ALL: legacy JIT probes deleted (target zero). The numpy-vectorized
# monotonic checks below are canonical (C-speed, exact).


def _g_sort_asc(keys):
    if keys.size < 2:
        return True, True
    # Numpy vectorized monotonic check (C-speed, exact).
    # Two vectorized passes (C-speed): >= for ok, != for strict.
    ge = keys[1:] >= keys[:-1]
    ok = bool(np.all(ge))
    if not ok:
        return False, False
    strict = bool(np.all(keys[1:] != keys[:-1]))
    return True, strict


def _g_sort_desc(keys):
    if keys.size < 2:
        return True, True
    le = keys[1:] <= keys[:-1]
    ok = bool(np.all(le))
    if not ok:
        return False, False
    strict = bool(np.all(keys[1:] != keys[:-1]))
    return True, strict


_S_LIB_OK = None
_S_LIB_KEY = None

# Buffer cache: dtype -> {cap, perm, tmp0, tmp1, tmp_p}; same
# resize-by-capacity discipline as _UCACHE (no super-allocator).
# Returned native steps are TRANSIENT views into the cache: the caller
# consumes them synchronously (idx = idx[step]) before the next call.
_SCACHE = {}


def _req_sort():
    """Sort perm kernels (additive ABI): lib or None (fallback owns it).

    Never disables the proven paths: missing symbols only disable the
    sort candidate lane. Memoized by loaded-object generation (see _abi_key).
    """
    global _S_LIB_OK, _S_LIB_KEY
    _probe()
    key = _abi_key()
    if _S_LIB_KEY == key:
        return _lib if _S_LIB_OK else None
    _S_LIB_KEY = key
    _S_LIB_OK = False
    if _lib is None:
        return None
    try:
        _lib.nf_sort_perm_i32.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint8,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p]
        _lib.nf_sort_perm_i32.restype = ctypes.c_int32
        _lib.nf_sort_perm_i64.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint8,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p]
        _lib.nf_sort_perm_i64.restype = ctypes.c_int32
        _ = _lib.nf_sort_perm_i32  # force Windows lookup now
        _S_LIB_OK = True
    except (OSError, AttributeError, ValueError):
        _S_LIB_OK = False
    return _lib if _S_LIB_OK else None


def sort_available():
    """True when sort perm kernels are present (env-aware)."""
    return _req_sort() is not None


def _sort_bufs(n, dt):
    """Cached (perm, tmp0, tmp1, tmp_p) with cap >= n.

    Resize (realloc) only when n exceeds capacity; steady-state calls
    reuse. tmp lanes are u32 for i32 keys, u64 for i64 keys.
    """
    e = _SCACHE.get(dt)
    if e is None or e["cap"] < n:
        tu = np.uint32 if dt == np.dtype(np.int32) else np.uint64
        e = {"cap": n,
             "perm": np.empty(n, dtype=np.int32),
             "tmp0": np.empty(n, dtype=tu),
             "tmp1": np.empty(n, dtype=tu),
             "tmp_p": np.empty(n, dtype=np.int32)}
        _SCACHE[dt] = e
    return e["perm"], e["tmp0"], e["tmp1"], e["tmp_p"]


def sort_perm_step(keys, descending):
    """One key-step stable permutation candidate: (step, backend).

    step is int32 positions (== np.argsort(kind="stable") / cpu
    _stable_desc_idx bit-for-bit) or None when the caller must run the
    proven base path verbatim. backend is "native" only when the DLL
    symbol actually executed (rc==0), else "numpy"/"ident"/"rev"
    (no native call: monotonic fast paths, base fallback, disabled /
    missing / failed backend -> proven oracle owns correctness).
    Native steps are transient cache views (consume synchronously).
    """
    k = np.ascontiguousarray(np.asarray(keys))
    dt = k.dtype
    n = k.size
    if (dt not in (np.dtype(np.int32), np.dtype(np.int64))
            or n == 0 or n > 2 ** 31 - 1):
        return None, "numpy"
    if bool(descending):
        ok_d, _ = _g_sort_desc(k)
        if ok_d:
            return np.arange(n, dtype=np.int32), "ident"
        ok_a, st_a = _g_sort_asc(k)
        if ok_a:
            if st_a:
                return np.arange(n - 1, -1, -1, dtype=np.int32), "rev"
            return None, "numpy"
    else:
        ok_a, _ = _g_sort_asc(k)
        if ok_a:
            return np.arange(n, dtype=np.int32), "ident"
        ok_d, st_d = _g_sort_desc(k)
        if ok_d:
            if st_d:
                return np.arange(n - 1, -1, -1, dtype=np.int32), "rev"
            return None, "numpy"
    lib = _req_sort()
    if lib is None:
        return None, "numpy"
    try:
        perm, t0, t1, tp = _sort_bufs(n, dt)
        fn = (lib.nf_sort_perm_i32
              if dt == np.dtype(np.int32) else lib.nf_sort_perm_i64)
        rc = int(fn(k.ctypes.data, n, 1 if bool(descending) else 0,
                    perm.ctypes.data, t0.ctypes.data,
                    t1.ctypes.data, tp.ctypes.data))
        if rc != 0:
            return None, "numpy"
        return perm[:n], "native"
    except (OSError, RuntimeError, ValueError, AttributeError,
            MemoryError):
        return None, "numpy"


# ---------------- RNG (additive): native-or-raise wrappers ----------------
# Fallback owns correctness in _lib/rng.py (bit-exact mirror); these
# wrappers execute the Rust symbols ONLY when the rebuilt DLL exports
# them, else raise RuntimeError so the caller falls back (same pattern
# as _req_select/_req_sort: missing symbols never break the oracle).

_rng_ok = None  # memoized RNG entry points (new ABI, optional until rebuilt)


def _req_rng():
    """RNG entry points: lib or None (fallback owns it until rebuilt)."""
    global _rng_ok
    _probe()
    if _lib is None:
        return None
    if _rng_ok is None:
        try:
            _lib.nf_rng_fill_i32.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32]
            _lib.nf_rng_fill_i32.restype = ctypes.c_int32
            _lib.nf_rng_fill_f64.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_double, ctypes.c_double]
            _lib.nf_rng_fill_f64.restype = ctypes.c_int32
            _lib.nf_rng_map_round.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
            _lib.nf_rng_map_round.restype = ctypes.c_int32
            _lib.nf_rng_sample_no_replace.argtypes = [
                ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_void_p]
            _lib.nf_rng_sample_no_replace.restype = ctypes.c_int64
            _lib.nf_rng_permutation.argtypes = [
                ctypes.c_size_t,
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_void_p]
            _lib.nf_rng_permutation.restype = ctypes.c_int64
            _lib.nf_rng_compat_runif.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32,
                ctypes.c_double, ctypes.c_double]
            _lib.nf_rng_compat_runif.restype = ctypes.c_int32
            _lib.nf_rng_compat_sample.argtypes = [
                ctypes.c_int64, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_int32]
            _lib.nf_rng_compat_sample.restype = ctypes.c_int32
            _ = _lib.nf_rng_fill_i32  # force lazy Windows lookup now
            _rng_ok = True
        except (OSError, AttributeError, ValueError):
            _rng_ok = False
    return _lib if _rng_ok else None


def rng_available():
    """True when RNG native entry points are present (env-aware)."""
    return _req_rng() is not None


def _rng_native():
    lib = _req_rng()
    if lib is None:
        raise RuntimeError("rng native unavailable (DLL predates RNG symbols)")
    return lib


def rng_fill_i32_native(n, seed, stream, offset, lo, hi, mode=0):
    out = np.zeros(max(0, n), dtype=np.int32)
    rc = int(_rng_native().nf_rng_fill_i32(
        out.ctypes.data, n, int(seed), int(stream), int(offset),
        int(lo), int(hi), int(mode)))
    if rc != 0:
        raise RuntimeError("nf_rng_fill_i32 rc=%d" % rc)
    return out


def rng_fill_f64_native(n, seed, stream, offset, lo, hi):
    out = np.zeros(max(0, n), dtype=np.float64)
    rc = int(_rng_native().nf_rng_fill_f64(
        out.ctypes.data, n, int(seed), int(stream), int(offset),
        float(lo), float(hi)))
    if rc != 0:
        raise RuntimeError("nf_rng_fill_f64 rc=%d" % rc)
    return out


def rng_map_round_native(x, valid, ndigits):
    x = np.ascontiguousarray(x, dtype=np.float64)
    valid = np.ascontiguousarray(valid, dtype=np.uint8)
    out = np.zeros(x.size, dtype=np.float64)
    ov = np.zeros(x.size, dtype=np.uint8)
    rc = int(_rng_native().nf_rng_map_round(
        x.ctypes.data, valid.ctypes.data, x.size,
        int(ndigits), out.ctypes.data, ov.ctypes.data))
    if rc != 0:
        raise RuntimeError("nf_rng_map_round rc=%d" % rc)
    return out, ov


def rng_sample_native(n, k, seed, stream, offset):
    pool = np.zeros(max(1, n), dtype=np.int32)
    out = np.zeros(max(1, k), dtype=np.int32)
    ng = int(_rng_native().nf_rng_sample_no_replace(
        n, k, int(seed), int(stream), int(offset),
        pool.ctypes.data, out.ctypes.data))
    if ng < 0:
        raise RuntimeError("nf_rng_sample_no_replace rc=%d" % ng)
    return out[:ng].copy()


def rng_permutation_native(n, seed, stream, offset):
    pool = np.zeros(max(1, n), dtype=np.int32)
    out = np.zeros(max(1, n), dtype=np.int32)
    ng = int(_rng_native().nf_rng_permutation(
        n, int(seed), int(stream), int(offset),
        pool.ctypes.data, out.ctypes.data))
    if ng < 0:
        raise RuntimeError("nf_rng_permutation rc=%d" % ng)
    return out[:ng].copy()


def rng_compat_runif_native(n, seed, lo, hi):
    out = np.zeros(max(0, n), dtype=np.float64)
    rc = int(_rng_native().nf_rng_compat_runif(
        out.ctypes.data, n, int(seed), float(lo), float(hi)))
    if rc != 0:
        raise RuntimeError("nf_rng_compat_runif rc=%d" % rc)
    return out


def rng_compat_sample_native(n, m, seed):
    out = np.zeros(max(1, m), dtype=np.int32)
    rc = int(_rng_native().nf_rng_compat_sample(
        int(n), out.ctypes.data, m, int(seed)))
    if rc != 0:
        raise RuntimeError("nf_rng_compat_sample rc=%d" % rc)
    return out[:m].copy()


# ---------------- TEXT length/contains (additive) ----------------
# Same wire as pattern_encode_buffers: data bytes + offs int32.
# Length = Unicode code points (NOT bytes); contains = UTF-8
# substring, case-sensitive, empty needle matches all rows.
# Fallback owns correctness (np.char parity); native is speed only.
# Missing symbols (old DLL) -> None lane, never a cascade.

_TEXT_OK = None
_TEXT_KEY = None


def _req_text():
    """TEXT kernels (additive ABI): lib or None (fallback owns it).

    Never disables the proven paths: missing symbols only disable the
    TEXT native lane. Memoized by loaded-object generation (see _abi_key).
    """
    global _TEXT_OK, _TEXT_KEY
    _probe()
    key = _abi_key()
    if _TEXT_KEY == key:
        return _lib if _TEXT_OK else None
    _TEXT_KEY = key
    _TEXT_OK = False
    if _lib is None:
        return None
    try:
        _lib.nf_text_length.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        _lib.nf_text_length.restype = ctypes.c_int32
        _lib.nf_text_contains.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        _lib.nf_text_contains.restype = ctypes.c_int32
        _lib.nf_text_startswith.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        _lib.nf_text_startswith.restype = ctypes.c_int32
        _lib.nf_text_endswith.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        _lib.nf_text_endswith.restype = ctypes.c_int32
        _lib.nf_text_equals.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        _lib.nf_text_equals.restype = ctypes.c_int32
        _ = _lib.nf_text_length  # force Windows lookup now
        _TEXT_OK = True
    except (OSError, AttributeError, ValueError):
        _TEXT_OK = False
    return _lib if _TEXT_OK else None


def text_available():
    """True when TEXT native entry points are present (env-aware)."""
    return _req_text() is not None


def build_text_buffers(strs):
    """strings -> (data bytes, offs int32). Same wire as pattern path."""
    return build_pattern_buffers(strs)


def _fb_text_length(strs):
    return np.char.str_len(np.asarray(list(strs), dtype=str)).astype(np.int32)


def _fb_text_contains(strs, substr):
    if substr == "":
        return np.ones(len(list(strs)), dtype=np.uint8)
    return (np.char.find(np.asarray(list(strs), dtype=str), substr) != -1).astype(np.uint8)


def _fb_text_startswith(strs, prefix):
    return np.char.startswith(np.asarray(list(strs), dtype=str), prefix).astype(np.uint8)


def _fb_text_endswith(strs, suffix):
    return np.char.endswith(np.asarray(list(strs), dtype=str), suffix).astype(np.uint8)


def _fb_text_equals(strs, key):
    return (np.asarray(list(strs), dtype=str) == key).astype(np.uint8)


def text_length_buffers(data, offs):
    """Code-point lengths over data/offs buffers (native or reference).

    Returns int32[N]. rc != 0 (malformed/UTF-8) raises RuntimeError:
    caller falls back to the verbatim reference.
    """
    offs = np.ascontiguousarray(offs, dtype=np.int32)
    n = offs.size - 1
    lib = _req_text()
    if lib is None:
        raise RuntimeError("text native unavailable (DLL predates TEXT symbols)")
    dp = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros(max(0, n), dtype=np.int32)
    if n == 0:
        return out
    rc = int(lib.nf_text_length(dp.ctypes.data, len(data), offs.ctypes.data,
                                n, out.ctypes.data))
    if rc != 0:
        raise RuntimeError("nf_text_length rc=%d" % rc)
    return out


def text_contains_buffers(data, offs, substr):
    """Substring hits (u8 0/1) over data/offs buffers (native or raise).

    Empty substr matches all rows. rc != 0 raises RuntimeError.
    """
    return _text_affix_buffers("nf_text_contains", data, offs, substr)


def _text_affix_buffers(sym, data, offs, needle):
    """Shared data/offs -> u8 wrapper for contains/startswith/endswith/equals."""
    offs = np.ascontiguousarray(offs, dtype=np.int32)
    n = offs.size - 1
    p = needle.encode("utf-8") if isinstance(needle, str) else bytes(needle)
    try:
        p.decode("utf-8")
    except UnicodeDecodeError:
        raise RuntimeError("%s rc=-2 (non-UTF8 needle)" % sym)
    lib = _req_text()
    if lib is None:
        raise RuntimeError("text native unavailable (DLL predates TEXT symbols)")
    dp = np.frombuffer(data, dtype=np.uint8)
    pp = np.frombuffer(p, dtype=np.uint8)
    out = np.zeros(max(0, n), dtype=np.uint8)
    if n == 0:
        return out
    rc = int(getattr(lib, sym)(dp.ctypes.data, len(data), offs.ctypes.data,
                               n, pp.ctypes.data, len(p), out.ctypes.data))
    if rc != 0:
        raise RuntimeError("%s rc=%d" % (sym, rc))
    return out


def text_startswith_buffers(data, offs, prefix):
    """Prefix hits (u8 0/1) over data/offs buffers (native or raise)."""
    return _text_affix_buffers("nf_text_startswith", data, offs, prefix)


def text_endswith_buffers(data, offs, suffix):
    """Suffix hits (u8 0/1) over data/offs buffers (native or raise)."""
    return _text_affix_buffers("nf_text_endswith", data, offs, suffix)


def text_equals_buffers(data, offs, key):
    """Equality hits (u8 0/1) over data/offs buffers (native or raise)."""
    return _text_affix_buffers("nf_text_equals", data, offs, key)


# ---------------- TEXT dictionary dedup (additive) ----------------
# nf_unique_dict_utf8: Arrow buffer consumption, no Python str objects.
# Missing symbol (old DLL) -> None: caller runs the proven Python dedup.

_DEDUP_OK = None
_DEDUP_KEY = None


def _req_text_dedup():
    """Text dedup entry point: lib or None (fallback owns it).

    Same optional-ABI discipline as _req_text/_req_sort: missing symbols
    disable only the native dedup path, never the proven TEXT/groupby
    paths (no whole-backend unavailable cascade).
    """
    global _DEDUP_OK, _DEDUP_KEY
    _probe()
    key = _abi_key()
    if _DEDUP_KEY == key:
        return _lib if _DEDUP_OK else None
    _DEDUP_KEY = key
    _DEDUP_OK = False
    if _lib is None:
        return None
    try:
        _lib.nf_unique_dict_utf8.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,  # data, data_len
            ctypes.c_void_p, ctypes.c_size_t,  # offs, offs_len
            ctypes.c_void_p, ctypes.c_size_t,  # valid, n
            ctypes.c_void_p, ctypes.c_void_p,  # codes_out, uniq_data_out
            ctypes.c_void_p, ctypes.c_void_p]  # uniq_offs_out, ng_out
        _lib.nf_unique_dict_utf8.restype = ctypes.c_int32
        _ = _lib.nf_unique_dict_utf8  # force Windows lookup now
        _DEDUP_OK = True
    except (OSError, AttributeError, ValueError):
        _DEDUP_OK = False
    return _lib if _DEDUP_OK else None


def text_dedup_available():
    """True when text dedup native entry points are present (env-aware)."""
    return _req_text_dedup() is not None


# Segmented P1 / adjacency P2 native-alongside (additive, explicit-call only).
# Wraps nf_segment_count/nf_segment_reduce_f32/nf_segment_reduce_i32 and
# nf_adjacency_slice/nf_adjacency_gather: the frozen P1/P2 semantics of
# Relational/Segmented (variant-A bounds reduce; count->u32; i32 sum
# saturates; f32 sum IEEE-propagates; min/max of an empty group errors;
# UINT32_MAX reserved; V/E/k=0 safe). Segmented.py does NOT call this
# yet (this step only places Rust beside Python); parity scripts call
# these wrappers explicitly with (result, backend). NumPy stays the
# proven oracle (fallback owns every absence/failure).
_SEG_OK = None
_SEG_KEY = None
_ADJ_OK = None
_ADJ_KEY = None

_SEG_MAX_N = 4194240
_SEG_OPS = {"sum": 0, "min": 2, "max": 3}
_ADJ_INF = np.int64(4294967295)


def _req_segment():
    """Segmented P1 entry points (additive ABI): lib or None (fallback owns it).

    Never disables the proven paths: missing symbols only disable the
    segmented candidate lane. Memoized by loaded-object generation (see _abi_key).
    """
    global _SEG_OK, _SEG_KEY
    _probe()
    key = _abi_key()
    if _SEG_KEY == key:
        return _lib if _SEG_OK else None
    _SEG_KEY = key
    _SEG_OK = False
    if _lib is None:
        return None
    try:
        v = ctypes.c_void_p
        z = ctypes.c_size_t
        u = ctypes.c_uint32
        _lib.nf_segment_count.argtypes = [v, z, z, v]
        _lib.nf_segment_count.restype = ctypes.c_int32
        _lib.nf_segment_reduce_f32.argtypes = [v, z, v, z, u, v]
        _lib.nf_segment_reduce_f32.restype = ctypes.c_int32
        _lib.nf_segment_reduce_i32.argtypes = [v, z, v, z, u, v]
        _lib.nf_segment_reduce_i32.restype = ctypes.c_int32
        _ = _lib.nf_segment_count  # force Windows lookup now
        _SEG_OK = True
    except (OSError, AttributeError, ValueError):
        _SEG_OK = False
    return _lib if _SEG_OK else None


def segment_available():
    """True when segmented P1 native entry points are present (env-aware)."""
    return _req_segment() is not None


def _req_adjacency():
    """Adjacency P2 entry points (additive ABI): lib or None (fallback owns it).

    Never disables the proven paths: missing symbols only disable the
    adjacency candidate lane. Memoized by loaded-object generation (see _abi_key).
    """
    global _ADJ_OK, _ADJ_KEY
    _probe()
    key = _abi_key()
    if _ADJ_KEY == key:
        return _lib if _ADJ_OK else None
    _ADJ_KEY = key
    _ADJ_OK = False
    if _lib is None:
        return None
    try:
        v = ctypes.c_void_p
        z = ctypes.c_size_t
        _lib.nf_adjacency_slice.argtypes = [v, z, v, z, v, z, v, v]
        _lib.nf_adjacency_slice.restype = ctypes.c_int32
        _lib.nf_adjacency_gather.argtypes = [v, z, v, v, z, v, z]
        _lib.nf_adjacency_gather.restype = ctypes.c_int32
        _ = _lib.nf_adjacency_slice  # force Windows lookup now
        _ADJ_OK = True
    except (OSError, AttributeError, ValueError):
        _ADJ_OK = False
    return _lib if _ADJ_OK else None


def adjacency_available():
    """True when adjacency P2 native entry points are present (env-aware)."""
    return _req_adjacency() is not None


def _seg_rc(rc, what):
    if rc == 0:
        return
    if rc == -1:
        raise RuntimeError("segmented candidate: native null-state on %s" % what)
    raise ValueError("segmented candidate: %s rejected (rc=%d)" % (what, rc))


def _seg_values_lanes(values):
    """values -> contiguous int32/float32 (same lane split as Segmented P1)."""
    a = np.ascontiguousarray(np.asarray(values))
    if a.dtype == np.dtype(np.int32) or a.dtype == np.dtype(np.float32):
        if a.ndim != 1:
            raise ValueError("segmented candidate: values ndim != 1")
        return a
    if a.dtype == np.dtype(np.int64):
        if a.ndim != 1:
            raise ValueError("segmented candidate: values ndim != 1")
        if a.size and (int(a.min()) < -2147483648 or int(a.max()) > 2147483647):
            raise ValueError("segmented candidate: int64 out of int32 range")
        return np.ascontiguousarray(a.astype(np.int32))
    if a.dtype == np.dtype(np.float64):
        if a.ndim != 1:
            raise ValueError("segmented candidate: values ndim != 1")
        return np.ascontiguousarray(a.astype(np.float32))
    raise ValueError("segmented candidate: dtype %s not f32/i32" % (a.dtype,))


def _seg_bounds_u32(bounds):
    """bounds -> contiguous uint32 (range gate; monotone/terminal owned by Rust)."""
    b = np.asarray(bounds)
    if b.ndim != 1:
        raise ValueError("segmented candidate: bounds ndim != 1")
    if b.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if b.dtype.kind not in "iu":
        raise ValueError("segmented candidate: bounds dtype %s not u32" % (b.dtype,))
    b64 = b.astype(np.int64)
    if bool(np.any((b64 < 0) | (b64 > 4294967295))):
        raise ValueError("segmented candidate: bounds out of u32 range")
    return np.ascontiguousarray(b64.astype(np.uint32))


def _fb_segmented(vals, bou32, n, m, op):
    """Numpy reference for P1 (mirrors Relational/Segmented semantics).

    Bit-exact vs the native P1 lanes: i32 sums via padded reduceat
    (integer arithmetic is order-independent; the zero pad only touches
    windows of trailing empty groups, which are overwritten below);
    f32 sums via sequential reduceat (native order is strided-sequential,
    while reduceat is pairwise: they agree to ~1ulp on short segments but
    can diverge on long ones); min/max/count are
    order-independent (NaN compared by propagation, not payload).
    """
    if n > _SEG_MAX_N:
        raise ValueError("segmented candidate: n=%d > %d" % (n, _SEG_MAX_N))
    b64 = bou32.astype(np.int64)
    if n == 0:
        if m != 0:
            raise ValueError("segmented candidate: n=0 requires M=0")
        if op == "count":
            return np.zeros(0, dtype=np.uint32)
        return np.zeros(0, dtype=np.float32
                        if vals.dtype == np.dtype(np.float32) else np.int32)
    if b64.size == 0:
        raise ValueError("segmented candidate: bounds empty with n>0")
    if int(b64[-1]) != n:
        raise ValueError("segmented candidate: bounds[-1] != n")
    if bool(np.any(np.diff(b64) < 0)):
        raise ValueError("segmented candidate: bounds not monotone")
    if bool(np.any((b64 < 0) | (b64 > n))):
        raise ValueError("segmented candidate: bounds outside [0, n]")
    if op in ("min", "max") and bool(np.any(np.diff(b64) == 0)):
        raise ValueError("segmented candidate: min/max of empty group")
    if op == "count":
        return np.diff(b64).astype(np.uint32)
    empty = np.diff(b64) == 0
    if vals.dtype == np.dtype(np.int32):
        if op == "sum":
            if m == 0:
                return np.empty(0, dtype=np.int32)
            # Zero pad: every probe lane <= n < n+1 is a valid reduceat
            # index (trailing empties included); the pad only widens the
            # final window, which belongs to an empty group whenever the
            # pad is actually covered, and empties are zeroed below.
            vext = np.concatenate([vals.astype(np.int64),
                                   np.zeros(1, dtype=np.int64)])
            acc = np.add.reduceat(vext, b64[:-1])
            acc = np.where(empty, np.int64(0), acc)
            return np.clip(acc, -2147483648, 2147483647).astype(np.int32)
        fn = np.minimum if op == "min" else np.maximum
        return fn.reduceat(vals, b64[:-1]).astype(np.int32)
    if op == "sum":
        if m == 0:
            return np.empty(0, dtype=np.float32)
        # NATIVE-ALL: vectorized reduceat is canonical (exact, C-speed).
        vext = np.concatenate([vals, np.zeros(1, dtype=np.float32)])
        out = np.add.reduceat(vext, b64[:-1])
        return np.where(empty, np.float32(0.0), out).astype(np.float32)
    fn = np.minimum if op == "min" else np.maximum
    return fn.reduceat(vals, b64[:-1]).astype(np.float32)


def segmented_reduce_native(values, bounds, op):
    """P1 candidate: (out, backend) with frozen Segmented P1 semantics.

    op in sum/count/min/max (mean/unknown -> ValueError, same as
    Segmented.py). count -> uint32. backend is "native" only when the
    DLL symbol actually executed, else "numpy" (disabled/missing/failed
    -> proven reference, exact). Segmented.py does not call this yet.
    """
    if op not in ("sum", "count", "min", "max"):
        raise ValueError("segmented candidate: op %r not in sum/count/min/max" % (op,))
    vals = _seg_values_lanes(values)
    n = int(vals.shape[0])
    bou32 = _seg_bounds_u32(bounds)
    if n == 0:
        if bou32.size == 0:
            m = 0
        elif bou32.size == 1 and int(bou32[0]) == 0:
            m = 0
        else:
            raise ValueError("segmented candidate: n=0 requires M=0")
        return _fb_segmented(vals, bou32, 0, m, op), "numpy"
    if bou32.size == 0:
        raise ValueError("segmented candidate: bounds empty with n>0")
    m = int(bou32.size) - 1
    lib = _req_segment()
    if lib is None:
        return _fb_segmented(vals, bou32, n, m, op), "numpy"
    try:
        if op == "count":
            out = np.empty(m, dtype=np.uint32)
            rc = lib.nf_segment_count(bou32.ctypes.data, m, n, out.ctypes.data)
            _seg_rc(rc, "count")
            return out, "native"
        if vals.dtype == np.dtype(np.float32):
            out = np.empty(m, dtype=np.float32)
            rc = lib.nf_segment_reduce_f32(vals.ctypes.data, n, bou32.ctypes.data,
                                           m, _SEG_OPS[op], out.ctypes.data)
        else:
            out = np.empty(m, dtype=np.int32)
            rc = lib.nf_segment_reduce_i32(vals.ctypes.data, n, bou32.ctypes.data,
                                           m, _SEG_OPS[op], out.ctypes.data)
        _seg_rc(rc, "op=%s" % op)
        return out, "native"
    except (OSError, RuntimeError, ValueError, AttributeError, MemoryError):
        raise
    except Exception:
        return _fb_segmented(vals, bou32, n, m, op), "numpy"


def _adj_ids_u32(a, name):
    """CSR id lane -> contiguous uint32 (UINT32_MAX reserved -> ValueError)."""
    x = np.asarray(a)
    if x.ndim != 1:
        raise ValueError("adjacency candidate: %s ndim != 1" % name)
    if x.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if x.dtype.kind not in "iu":
        raise ValueError("adjacency candidate: %s dtype %s not u32" % (name, x.dtype))
    x64 = x.astype(np.int64)
    if bool(np.any(x64 == _ADJ_INF)):
        raise ValueError("adjacency candidate: %s holds UINT32_MAX (INF/INVALID)" % name)
    if bool(np.any((x64 < 0) | (x64 > 4294967294))):
        raise ValueError("adjacency candidate: %s out of u32 range" % name)
    return np.ascontiguousarray(x64.astype(np.uint32))


def _adj_rc(rc, what):
    if rc == 0:
        return
    if rc == -1:
        raise RuntimeError("adjacency candidate: native null-state on %s" % what)
    raise ValueError("adjacency candidate: %s rejected (rc=%d)" % (what, rc))


def _fb_adjacency_slice(ip, ix, q):
    """Numpy reference for P2 slice (mirrors Relational/Segmented semantics)."""
    ip64 = ip.astype(np.int64)
    if ip64.shape[0] == 0:
        v = 0
        if ix.shape[0] != 0:
            raise ValueError("adjacency candidate: indptr empty but indices non-empty")
    else:
        v = int(ip64.shape[0]) - 1
    e = int(ix.shape[0])
    k = int(q.shape[0])
    if k == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint32)
    if v == 0:
        raise ValueError("adjacency candidate: query on empty graph (V=0)")
    if bool(np.any(np.diff(ip64) < 0)):
        raise ValueError("adjacency candidate: indptr not monotone")
    if bool(np.any((ip64 < 0) | (ip64 > e))):
        raise ValueError("adjacency candidate: indptr outside [0, E]")
    if int(ip64[-1]) != e:
        raise ValueError("adjacency candidate: indptr[-1] != E")
    q64 = q.astype(np.int64)
    if bool(np.any((q64 < 0) | (q64 >= v))):
        raise ValueError("adjacency candidate: query vertex out of range")
    return ip64[q64].astype(np.uint32), ip64[q64 + 1].astype(np.uint32)


def adjacency_slice_native(indptr, indices, query):
    """P2 slice candidate: ((begins, ends), backend) with frozen P2 semantics.

    backend is "native" only when the DLL symbol actually executed,
    else "numpy". Segmented.py does not call this yet.
    """
    ip = _adj_ids_u32(indptr, "indptr")
    ix = _adj_ids_u32(indices, "indices")
    q = _adj_ids_u32(query, "query")
    k = int(q.size)
    lib = _req_adjacency()
    if lib is None:
        return _fb_adjacency_slice(ip, ix, q), "numpy"
    try:
        begins = np.empty(k, dtype=np.uint32)
        ends = np.empty(k, dtype=np.uint32)
        rc = lib.nf_adjacency_slice(ip.ctypes.data, ip.size, ix.ctypes.data, ix.size,
                                    q.ctypes.data, k, begins.ctypes.data,
                                    ends.ctypes.data)
        _adj_rc(rc, "slice")
        return (begins, ends), "native"
    except (RuntimeError, ValueError):
        raise
    except Exception:
        return _fb_adjacency_slice(ip, ix, q), "numpy"


def adjacency_gather_native(indices, begins, ends):
    """P2 gather candidate: (flat, backend) over begins/ends pairs.

    Storage order inside each slice, query order across slices.
    backend is "native" only when the DLL symbol actually executed,
    else "numpy". Segmented.py does not call this yet.
    """
    ix = np.ascontiguousarray(np.asarray(indices))
    b = np.ascontiguousarray(np.asarray(begins, dtype=np.int64))
    en = np.ascontiguousarray(np.asarray(ends, dtype=np.int64))
    if b.shape != en.shape:
        raise ValueError("adjacency candidate: begins/ends shape mismatch")
    if b.size == 0:
        return np.zeros(0, dtype=ix.dtype if ix.size else np.uint32), (
            "native" if adjacency_available() else "numpy")
    k = int(b.size)
    total = int((en - b).sum())
    if bool(np.any(en < b)):
        raise ValueError("adjacency candidate: begins > ends")
    if bool(np.any(en > ix.size)):
        raise ValueError("adjacency candidate: ends > E")
    lib = _req_adjacency()
    if lib is None or ix.dtype != np.dtype(np.uint32):
        parts = [np.take(ix, np.arange(int(s), int(t)))
                 for s, t in zip(b.tolist(), en.tolist()) if int(t) > int(s)]
        flat = np.concatenate(parts) if parts else np.zeros(0, dtype=ix.dtype)
        return flat, "numpy"
    b32 = np.ascontiguousarray(b.astype(np.uint32))
    e32 = np.ascontiguousarray(en.astype(np.uint32))
    try:
        out = np.empty(total, dtype=np.uint32)
        rc = lib.nf_adjacency_gather(ix.ctypes.data, ix.size, b32.ctypes.data,
                                     e32.ctypes.data, k, out.ctypes.data, total)
        _adj_rc(rc, "gather")
        return out, "native"
    except (RuntimeError, ValueError):
        raise
    except Exception:
        parts = [np.take(ix, np.arange(int(s), int(t)))
                 for s, t in zip(b.tolist(), en.tolist()) if int(t) > int(s)]
        flat = np.concatenate(parts) if parts else np.zeros(0, dtype=ix.dtype)
        return flat, "numpy"
