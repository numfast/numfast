# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Native GroupedHashMT: proven partitioned pair-hash COUNT DISTINCT on Rust.

STANDALONE module: only numpy/ctypes/threading/time/os imports, no
Builder/Extension imports (same discipline as
`Drivers/CPU/_lib/native_cpu.py`). Research scripts load this file
directly by path; the old Numba lane
(`grouped_mt_hash.py`) stays intact as fallback/reference and is
never imported here (a cold process running this module never
touches JIT at all).

Algorithm: bit-identical port of the proven Numba GroupedHashMT
(partitioned open-addressing over composite (key, value) pairs +
fingerprint/slot routing + MT + reusable caller-owned buffers).
Five additive C ABI kernels in the existing `numfast-native` crate
(caller-owned buffers, no alloc), same return-code convention as
the frozen kernels (0 ok, -1 null, -3 bad geometry). Generic only:
no cardinality constants, no packed sort keys, dtypes preserved
(no int32 narrowing).

Integer generality: the Rust kernels are int64 lanes. Narrower
integer dtypes widen exactly (int32/uint32 -> int64 injective, so
fingerprints and full-key compares match the Numba lane bit for
bit); uint64 lanes past i64::MAX and non-integer dtypes raise
_HashMiss (the caller keeps the proven lane — this module never
raises past _HashMiss).

Structural constants (same as the proven lane, NOT data cards):
P from N only (~256K rows per partition, 16..256 pow2); slot
factor 1.5; DENSE_CAP 2**24; threads from os.cpu_count capped by P
and 16 (overridable per call for the MT ladder — mechanical fan-out
only, algorithm untouched).
"""

# ONE documented exception to the NO-INTERNAL-IMPORTS rule above:
# `_default_native_dll` imports numfast._lib.native_env LAZILY, inside the
# function body, because an eager module-scope import runs while
# numfast/__init__ is still executing and numfast._lib/__init__ eagerly builds
# Series/Table -- a cycle. The rule exists so this file stays loadable BY PATH;
# a lazy import cannot break that, and without it this file's only candidate on
# a non-Windows host is a Windows path.


import ctypes
import math
import os
import threading
import time

import numpy as np


class _HashMiss(Exception):
    """Fast-lane miss: caller must use the proven Numba lane."""


GroupedHashMiss = _HashMiss


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
_probe_env = None


def _probe():
    global _lib, _why, _probe_env
    key = (os.environ.get("NUMFAST_NATIVE_DISABLE"),
           os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT))
    if key == _probe_env and _why != "unprobed":
        return
    _probe_env = key
    if os.environ.get("NUMFAST_NATIVE_DISABLE") == "1":
        _lib, _why = None, "disabled-by-env"
        return
    path = os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT)
    try:
        lib = ctypes.CDLL(path)
        v = ctypes.c_void_p
        z = ctypes.c_size_t
        i = ctypes.c_int64
        lib.nf_ghash_fp_i64.argtypes = [v, v, z, v, i, z]
        lib.nf_ghash_fp_i64.restype = ctypes.c_int32
        lib.nf_ghash_count_i64.argtypes = [v, z, v, z, z, z]
        lib.nf_ghash_count_i64.restype = ctypes.c_int32
        lib.nf_ghash_scatter_i64.argtypes = [v, v, v, z, v, z, z, v, v, z]
        lib.nf_ghash_scatter_i64.restype = ctypes.c_int32
        lib.nf_ghash_pins_i64.argtypes = [v, v, v, v, v, v, v, v, z, z]
        lib.nf_ghash_pins_i64.restype = ctypes.c_int32
        lib.nf_ghash_occ_count.argtypes = [v, v, z, v, z]
        lib.nf_ghash_occ_count.restype = ctypes.c_int32
        lib.nf_ghash_occ_fill.argtypes = [v, v, v, v, v, z, z]
        lib.nf_ghash_occ_fill.restype = ctypes.c_int32
        _ = lib.nf_ghash_fp_i64  # force Windows lookup now
        _ = lib.nf_ghash_count_i64
        _ = lib.nf_ghash_scatter_i64
        _lib, _why = lib, "loaded:" + path
    except (OSError, AttributeError, ValueError) as e:
        _lib, _why = None, "unavailable:%s" % e


def available():
    """True when the native GroupedHashMT kernels are loaded (env-aware)."""
    _probe()
    return _lib is not None


def why():
    _probe()
    return _why


def _req_native():
    _probe()
    if _lib is None:
        raise _HashMiss("native ghash backend %s" % _why)
    return _lib


_DENSE_CAP = 1 << 24
_ROWS_PER_PART = 1 << 18
_SLOTF = 1.5

# Generic transient-buffer pool: same discipline as the proven lane
# (largest-buffer + slice reuse across calls; roles are distinct
# live ranges so no aliasing; accumulate-into buffers zeroed on
# reuse). Pool is module-local so the Numba lane and the native lane
# never share pages.
_POOL = {}
_POOL_LOCK = threading.Lock()


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
        raise _HashMiss("pool alloc %s: %s" % (role, e))
    with _POOL_LOCK:
        old = _POOL.get(key)
        if old is None or old.size < n:
            _POOL[key] = fresh
            return fresh
        out = old[:n]
        if zero:
            out.fill(0)
        return out


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
    return int(min(max(c, 1), p, 16))


def _widen_i64(a):
    """Exact integer widening to int64 (no narrowing, no wrap).

    int64 -> zero-copy view; narrower int/uint -> exact astype;
    uint64 past i64::MAX -> _HashMiss (fallback owns it).
    Non-integer kinds -> _HashMiss.
    """
    a = np.ascontiguousarray(np.asarray(a)).ravel()
    kind = a.dtype.kind
    if kind not in ("i", "u"):
        raise _HashMiss("non-integer dtype %s" % (a.dtype,))
    if a.dtype == np.dtype(np.int64):
        return a
    if kind == "u" and a.dtype.itemsize == 8 and a.size:
        try:
            if int(a.max()) > (1 << 63) - 1:
                raise _HashMiss("uint64 past i64 range")
        except (ValueError, OverflowError, TypeError) as e:
            raise _HashMiss("uint64 range: %s" % e)
    try:
        return np.ascontiguousarray(a.astype(np.int64, copy=False))
    except (ValueError, OverflowError, TypeError) as e:
        raise _HashMiss("widen: %s" % e)


def _adaptive_unique(OK):
    """Sorted (ukeys int64, counts int64). Same rule as the proven lane."""
    n = int(OK.size)
    if n == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    try:
        lo = int(OK.min())
        hi = int(OK.max())
    except (ValueError, TypeError) as e:
        raise _HashMiss("range: %s" % e)
    if -(1 << 63) <= lo and hi <= (1 << 63) - 1 and (hi - lo) <= _DENSE_CAP:
        try:
            bc = np.bincount((OK - lo).astype(np.int64),
                             minlength=(hi - lo) + 1)
        except (MemoryError, ValueError, OverflowError) as e:
            raise _HashMiss("dense alloc: %s" % e)
        nz = np.flatnonzero(bc)
        return (np.ascontiguousarray((nz + lo).astype(np.int64)),
                np.ascontiguousarray(bc[nz].astype(np.int64)))
    uk, cn = np.unique(OK, return_counts=True)
    return (np.ascontiguousarray(uk.astype(np.int64, copy=False)),
            np.ascontiguousarray(cn.astype(np.int64, copy=False)))


def grouped_distinct_hash_native(kc, vc, nthreads=None):
    """(ukeys int64 sorted, nunique int64, hash_ms, scan_ms).

    Native port of the proven lane; raises _HashMiss on any
    condition the native lane cannot cover (backend absent,
    non-integer/uint64-huge dtypes, size mismatch, kernel rc).
    `nthreads` overrides the heuristic fan-out (MT ladder;
    mechanical only, clamped to [1, 64] in the kernels).
    """
    try:
        lib = _req_native()
        K = _widen_i64(kc)
        V = _widen_i64(vc)
        n = int(K.size)
        if n != int(V.size):
            raise _HashMiss("size mismatch")
        if n == 0:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                    0.0, 0.0)
        P = _partitions_for(n)
        if nthreads is None:
            T = _threads_for(P)
        else:
            T = int(max(1, min(int(nthreads), P, 64)))
        pmask = P - 1
        t0 = time.perf_counter()
        FP = _pool_get("FP", n, np.int64, False)
        rc = lib.nf_ghash_fp_i64(K.ctypes.data, V.ctypes.data, n,
                                 FP.ctypes.data, pmask, T)
        if rc != 0:
            raise _HashMiss("fp rc=%d" % rc)
        TC = _pool_get("TC", T * P, np.int64, True)
        rc = lib.nf_ghash_count_i64(FP.ctypes.data, n, TC.ctypes.data, T, P, T)
        if rc != 0:
            raise _HashMiss("count rc=%d" % rc)
        TC = TC.reshape(T, P)
        col = np.cumsum(TC, axis=0)
        counts = col[-1]
        starts = np.empty(P + 1, dtype=np.int64)
        starts[0] = 0
        starts[1:] = np.cumsum(counts)
        tpos = np.empty((T, P), dtype=np.int64)
        tpos[0] = starts[:-1]
        if T > 1:
            tpos[1:] = starts[:-1] + col[:-1]
        PK = _pool_get("PK", n, np.int64, False)
        PV = _pool_get("PV", n, np.int64, False)
        rc = lib.nf_ghash_scatter_i64(K.ctypes.data, V.ctypes.data,
                                      FP.ctypes.data, n, tpos.ctypes.data,
                                      T, P, PK.ctypes.data, PV.ctypes.data, T)
        if rc != 0:
            raise _HashMiss("scatter rc=%d" % rc)
        mp = np.array([_next_pow2(max(64, int(math.ceil(int(c) * _SLOTF))))
                       for c in counts], dtype=np.int64)
        toff = np.empty(P + 1, dtype=np.int64)
        toff[0] = 0
        toff[1:] = np.cumsum(mp)
        m = int(toff[-1])
        TK = _pool_get("TK", m, np.int64, False)
        TV = _pool_get("TV", m, np.int64, False)
        used = _pool_get("USED", m, np.uint8, True)
        tm = np.ascontiguousarray(mp - 1)
        rc = lib.nf_ghash_pins_i64(PK.ctypes.data, PV.ctypes.data,
                                   starts.ctypes.data, toff.ctypes.data,
                                   tm.ctypes.data,
                                   TK.ctypes.data, TV.ctypes.data,
                                   used.ctypes.data, P, T)
        if rc != 0:
            raise _HashMiss("pins rc=%d" % rc)
        hash_ms = (time.perf_counter() - t0) * 1000
        if os.environ.get("NF_MT_DEBUG"):
            print("[GroupedHashMT-native] P=%d T=%d n=%d M=%d hash=%.1f" % (
                P, T, n, m, hash_ms), flush=True)
        t1 = time.perf_counter()
        CNT = _pool_get("CNT", P, np.int64, False)
        rc = lib.nf_ghash_occ_count(used.ctypes.data, toff.ctypes.data,
                                    P, CNT.ctypes.data, T)
        if rc != 0:
            raise _HashMiss("occ_count rc=%d" % rc)
        ostart = np.empty(P + 1, dtype=np.int64)
        ostart[0] = 0
        ostart[1:] = np.cumsum(CNT)
        nocc = int(ostart[-1])
        OK = _pool_get("OK", nocc if nocc > 0 else 1, np.int64, False)[:nocc]
        rc = lib.nf_ghash_occ_fill(TK.ctypes.data, used.ctypes.data,
                                   toff.ctypes.data, ostart.ctypes.data,
                                   OK.ctypes.data, P, T)
        if rc != 0:
            raise _HashMiss("occ_fill rc=%d" % rc)
        ukeys, nunique = _adaptive_unique(OK)
        scan_ms = (time.perf_counter() - t1) * 1000
        return ukeys, nunique, hash_ms, scan_ms
    except _HashMiss:
        raise
    except Exception as e:
        raise _HashMiss("native lane: %s" % type(e).__name__)
