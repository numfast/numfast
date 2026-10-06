# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Rowwise K-way time-argmin + gather (ctypes, no PyO3) + numpy fallback.

STANDALONE module: only numpy/ctypes stdlib imports, no Builder/Extension
imports (NO INTERNAL IMPORTS rule). Research scripts load this file
directly by path until the primitive is proven; manifest wiring
(alias/mods/setup) happens only after proof.

Contract: identical outputs with backend available or not
(available=False never breaks correctness, only speed). Backend
absence (missing DLL, NUMFAST_NATIVE_DISABLE=1, symbol predates the
DLL) -> numpy fallback. No JIT dependency. No benchmark branches.
Generic lanes only.

Semantics (per row i over lanes k = 0..K, 1 <= K <= 256):
  m[i] = argmin_k(T[k][i]) with strict `<` from lane 0
         (ties keep the smallest index; int32 full-range ordered).
  t_best[i] = T[m[i]][i] (int32 exact copy).
  d_best[i] = D[m[i]][i] (float32 bit-exact copy, NaN/Inf/-0.0 ride as-is).
D NEVER selects the mode.

Inputs: T lanes int32[N] x K, D lanes float32[N] x K (1D, contiguous
buffers at the FFI boundary; non-contiguous views are copied once up
front). K in 1..=256 (else ValueError, outputs untouched).
Outputs: (t_best int32[N], d_best float32[N], m_best uint8[N]).
Caller-owned outputs (fresh arrays; inputs never mutated).
N=0 -> empty outputs without touching the backend.
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
_probe_env = None  # memoized (disable-flag, dll-path)
_kway_ok = None  # memoized K-way entry point (new ABI, optional until rebuilt)


def _probe():
    global _lib, _why, _probe_env, _kway_ok
    key = (os.environ.get("NUMFAST_NATIVE_DISABLE"),
           os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT))
    if key == _probe_env and _why != "unprobed":
        return
    _probe_env = key
    _kway_ok = None  # DLL identity changed -> re-probe entry point
    if os.environ.get("NUMFAST_NATIVE_DISABLE") == "1":
        _lib, _why = None, "disabled-by-env"
        return
    path = os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT)
    try:
        lib = ctypes.CDLL(path)
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


def _req_kway():
    """K-way entry point: lib or None (fallback owns it).

    Optional until the DLL is rebuilt with the K-way kernel: a missing
    symbol disables only the K-way native path, never any other path.
    """
    global _kway_ok
    _probe()
    if _lib is None:
        return None
    if _kway_ok is None:
        try:
            _lib.nf_rowwise_kway_time_argmin_gather.argtypes = (
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p)
            _lib.nf_rowwise_kway_time_argmin_gather.restype = ctypes.c_int32
            # Touch the symbol to force lazy Windows lookup now, not mid-call.
            _ = _lib.nf_rowwise_kway_time_argmin_gather
            _kway_ok = True
        except (OSError, AttributeError, ValueError):
            _kway_ok = False
    return _lib if _kway_ok else None


def kway_available():
    """True when the K-way native entry point is present (env-aware)."""
    return _req_kway() is not None


def _guard_lane(a, dtype, label):
    arr = np.asarray(a)
    if arr.dtype == np.dtype(bool):
        raise TypeError(
            f"rowwise_kway {label} must be {dtype}, got bool. "
            "Fix: pass int32/float32 lanes.")
    if arr.dtype != np.dtype(dtype):
        raise TypeError(
            f"rowwise_kway {label} must be {dtype}, got {arr.dtype}. "
            f"Fix: pass np.ascontiguousarray(a, dtype=np.{dtype}).")
    if arr.ndim != 1:
        raise ValueError(
            f"rowwise_kway {label} must be rank-1, got shape {arr.shape}. "
            "Fix: pass flat columns.")
    return np.ascontiguousarray(arr)


def _fb_kway(t_lanes, d_lanes):
    """NumPy fallback: strict-less argmin cascade over T + fancy gather.

    Strict `<` from lane 0 (ties keep the smallest index). No Python
    loops over rows, no sorting. D lanes are pure payload (never compared).
    """
    T = np.stack(t_lanes).astype(np.int32, copy=False)  # (K, N)
    D = np.stack(d_lanes).astype(np.float32, copy=False)  # (K, N)
    n = T.shape[1]
    m = np.zeros(n, dtype=np.uint8)
    best = T[0].copy()
    for k in range(1, T.shape[0]):
        take = T[k] < best
        m = np.where(take, np.uint8(k), m)
        best = np.where(take, T[k], best)
    idx = np.arange(n)
    t_best = np.ascontiguousarray(best)
    d_best = np.ascontiguousarray(D[m.astype(np.int64), idx])
    return (t_best, d_best, np.ascontiguousarray(m))


def rowwise_kway_time_argmin_gather(t_lanes, d_lanes):
    """Generic K-way rowwise MIN over int32 lanes + float32 gather.

    Inputs: sequences of K int32[N] T lanes and K float32[N] D lanes
    with 1 <= K <= 256 (else ValueError); all lanes share one length N.
    1D lanes; non-contiguous views are copied once up front.
    Outputs: (t_best int32[N], d_best float32[N], m_best uint8[N]) with
    m = argmin over the T lanes (ties keep the smallest index),
    t_best = T[m] exact, d_best = D[m] bit-exact.
    Caller-owned outputs (fresh arrays; inputs never mutated). N=0 ->
    empty outputs without touching the backend.
    """
    if not isinstance(t_lanes, (list, tuple)) or not isinstance(d_lanes, (list, tuple)):
        raise TypeError(
            "rowwise_kway lanes must be list/tuple of arrays. "
            "Fix: pass ([t0..], [d0..]).")
    k = len(t_lanes)
    if k != len(d_lanes):
        raise ValueError(
            f"rowwise_kway T/D lane count mismatch {k} != {len(d_lanes)}. "
            "Fix: pass equal counts.")
    if k < 1 or k > 256:
        raise ValueError(
            f"rowwise_kway K={k} outside 1..=256. "
            "Fix: pass 1..256 lanes.")
    t = [_guard_lane(a, "int32", f"T{i}") for i, a in enumerate(t_lanes)]
    d = [_guard_lane(a, "float32", f"D{i}") for i, a in enumerate(d_lanes)]
    sizes = {a.size for a in t} | {a.size for a in d}
    if len(sizes) != 1:
        raise ValueError(
            f"rowwise_kway lane size mismatch {sorted(sizes)}. "
            "Fix: pass lanes of one shared length N.")
    n = sizes.pop()
    if n == 0:
        return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.uint8))
    lib = _req_kway()
    if lib is None:
        return _fb_kway(t, d)
    t_ptrs = (ctypes.c_void_p * k)(*[a.ctypes.data for a in t])
    d_ptrs = (ctypes.c_void_p * k)(*[a.ctypes.data for a in d])
    t_best = np.empty(n, dtype=np.int32)
    d_best = np.empty(n, dtype=np.float32)
    m_best = np.empty(n, dtype=np.uint8)
    rc = lib.nf_rowwise_kway_time_argmin_gather(
        t_ptrs, d_ptrs, k, n,
        t_best.ctypes.data, d_best.ctypes.data, m_best.ctypes.data)
    assert rc == 0, rc
    return t_best, d_best, m_best
