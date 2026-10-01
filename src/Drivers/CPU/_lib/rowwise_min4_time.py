# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Rowwise MIN4 time-argmin + optional gather (ctypes, no PyO3) + numpy fallback.

STANDALONE module: only numpy/ctypes stdlib imports, no Builder/Extension
imports (NO INTERNAL IMPORTS rule). Research scripts load this file
directly by path until the primitive is proven; manifest wiring
(alias/mods/setup) happens only after proof.

Contract: identical outputs with backend available or not
(available=False never breaks correctness, only speed). Backend
absence (missing DLL, NUMFAST_NATIVE_DISABLE=1, symbol predates the
DLL) -> numpy fallback. No JIT dependency. No benchmark branches.
No query/transport/Dispatcher/RoadGraph vocabulary: generic lanes only.

Semantics (per row i over lanes k = 0..3):
  m[i] = argmin_k(T[k][i]) with strict `<` from lane 0
         (ties keep the smallest index; int32 full-range ordered).
  t_best[i] = T[m[i]][i] (int32 exact copy).
  d_best[i] = D[m[i]][i] (float32 bit-exact copy, NaN/Inf/-0.0 ride as-is)
              when D lanes are given; zeros when D lanes are omitted.
D NEVER selects the mode (regression vs the old D-argmin semantics).

Inputs: T0-T3 int32[N] (1D, contiguous buffers at the FFI boundary;
non-contiguous views are copied once up front). D0-D3 float32[N]
OPTIONAL as a group: all four given or all four None (mixed -> ValueError).
Outputs: (t_best int32[N], d_best float32[N], m_best uint8[N]).
Caller-owned outputs (fresh arrays; inputs never mutated).
N=0 -> empty outputs without touching the backend.
"""

import ctypes
import os

import numpy as np

_DLL_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "..", "..", "numfast-native", "target",
                            "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
_DLL_DEFAULT = os.path.abspath(_DLL_DEFAULT)

_lib = None
_why = "unprobed"
_probe_env = None  # memoized (disable-flag, dll-path)
_timerowmin_ok = None  # memoized time-rowmin entry point (new ABI, optional until rebuilt)


def _probe():
    global _lib, _why, _probe_env, _timerowmin_ok
    key = (os.environ.get("NUMFAST_NATIVE_DISABLE"),
           os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT))
    if key == _probe_env and _why != "unprobed":
        return
    _probe_env = key
    _timerowmin_ok = None  # DLL identity changed -> re-probe entry point
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


def _req_timerowmin():
    """Time-rowmin entry point (new ABI): lib or None (fallback owns it).

    Optional until the DLL is rebuilt with the time kernel: a missing
    symbol disables only the time-rowmin native path, never any other path.
    """
    global _timerowmin_ok
    _probe()
    if _lib is None:
        return None
    if _timerowmin_ok is None:
        try:
            _lib.nf_rowwise_min4_time_argmin_gather.argtypes = (
                [ctypes.c_void_p] * 8 + [ctypes.c_size_t]
                + [ctypes.c_void_p] * 3)
            _lib.nf_rowwise_min4_time_argmin_gather.restype = ctypes.c_int32
            # Touch the symbol to force lazy Windows lookup now, not mid-call.
            _ = _lib.nf_rowwise_min4_time_argmin_gather
            _timerowmin_ok = True
        except (OSError, AttributeError, ValueError):
            _timerowmin_ok = False
    return _lib if _timerowmin_ok else None


def time_rowmin_available():
    """True when the canonical K-way native entry point is present (env-aware)."""
    try:
        return bool(_kway_mod().kway_available())
    except Exception:
        return _req_timerowmin() is not None


def _guard_lane(a, dtype, label):
    arr = np.asarray(a)
    if arr.dtype == np.dtype(bool):
        raise TypeError(
            f"rowwise_min4_time {label} must be {dtype}, got bool. "
            "Fix: pass int32/float32 lanes.")
    if arr.dtype != np.dtype(dtype):
        raise TypeError(
            f"rowwise_min4_time {label} must be {dtype}, got {arr.dtype}. "
            f"Fix: pass np.ascontiguousarray(a, dtype=np.{dtype}).")
    if arr.ndim != 1:
        raise ValueError(
            f"rowwise_min4_time {label} must be rank-1, got shape {arr.shape}. "
            "Fix: pass flat columns.")
    return np.ascontiguousarray(arr)


def _kway_mod():
    """Sibling canonical K-way module (same dir, file-load safe)."""
    import importlib.util as _ilu
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rowwise_kway.py")
    spec = _ilu.spec_from_file_location("nf_rowwise_kway", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fb_time_rowmin(t_lanes, d_lanes_or_none):
    """NumPy fallback via the canonical K-way fallback (thin wrapper).

    Strict `<` from lane 0 (ties keep the smallest index). D lanes are
    pure payload (never compared). Time-only (D omitted) reuses the same
    T-scan with zero payload, then returns zeros for d_best.
    """
    kw = _kway_mod()
    if d_lanes_or_none is None:
        zeros = [np.zeros(a.size, dtype=np.float32) for a in t_lanes]
        tb, _, mb = kw._fb_kway(t_lanes, zeros)
        return (tb, np.zeros(tb.size, dtype=np.float32), mb)
    return kw._fb_kway(t_lanes, d_lanes_or_none)


def rowwise_min4_time_argmin_gather(t0, t1, t2, t3,
                                    d0=None, d1=None, d2=None, d3=None):
    """Generic rowwise MIN4 over int32 time lanes + optional float32 gather.

    Thin compat wrapper over the canonical K-way kernel (K = 4).
    Inputs: T0-T3 int32[N], optional D0-D3 float32[N] (all four given or
    all four None; mixed -> ValueError). 1D lanes; non-contiguous views
    are copied once up front.
    Outputs: (t_best int32[N], d_best float32[N], m_best uint8[N]) with
    m = argmin over the T lanes (ties keep the smallest index),
    t_best = T[m] exact, d_best = D[m] bit-exact (zeros when D omitted).
    Caller-owned outputs (fresh arrays; inputs never mutated). N=0 ->
    empty outputs without touching the backend.
    """
    t_lanes = [_guard_lane(a, "int32", f"T{i}")
               for i, a in enumerate((t0, t1, t2, t3))]
    given = [a is not None for a in (d0, d1, d2, d3)]
    if any(given) and not all(given):
        raise ValueError(
            "rowwise_min4_time D lanes must be all given or all None. "
            "Fix: pass d0..d3 together or omit all four.")
    d_lanes = None
    if all(given):
        d_lanes = [_guard_lane(a, "float32", f"D{i}")
                   for i, a in enumerate((d0, d1, d2, d3))]
    sizes = {a.size for a in t_lanes}
    if d_lanes is not None:
        sizes |= {a.size for a in d_lanes}
    if len(sizes) != 1:
        raise ValueError(
            f"rowwise_min4_time lane size mismatch {sorted(sizes)}. "
            "Fix: pass lanes of one shared length N.")
    n = sizes.pop()
    if n == 0:
        return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.uint8))
    kw = _kway_mod()
    if d_lanes is None:
        zeros = [np.zeros(n, dtype=np.float32) for _ in range(4)]
        tb, _, mb = kw.rowwise_kway_time_argmin_gather(t_lanes, zeros)
        return (tb, np.zeros(n, dtype=np.float32), mb)
    return kw.rowwise_kway_time_argmin_gather(t_lanes, d_lanes)
