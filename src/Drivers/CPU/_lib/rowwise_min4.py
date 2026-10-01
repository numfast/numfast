# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Rowwise MIN4 + argmin + gather (ctypes, no PyO3) + numpy fallback.

STANDALONE module: only numpy/ctypes stdlib imports, no Builder/Extension
imports (NO INTERNAL IMPORTS rule). Research scripts load this file
directly by path until the primitive is proven; manifest wiring
(alias/mods/setup) happens only after proof.

Contract: identical outputs with backend available or not
(available=False never breaks correctness, only speed). Backend
absence (missing DLL, NUMFAST_NATIVE_DISABLE=1, symbol predates the
DLL) -> numpy fallback. No JIT dependency. No benchmark branches.
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
_rowmin_ok = None  # memoized rowmin entry point (new ABI, optional until rebuilt)


def _probe():
    global _lib, _why, _probe_env, _rowmin_ok
    key = (os.environ.get("NUMFAST_NATIVE_DISABLE"),
           os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT))
    if key == _probe_env and _why != "unprobed":
        return
    _probe_env = key
    _rowmin_ok = None  # DLL identity changed -> re-probe rowmin entry point
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


def _req_rowmin():
    """Rowmin entry point (new ABI): lib or None (fallback owns it).

    Optional until the DLL is rebuilt with the rowmin kernel: a missing
    symbol disables only the rowmin native path, never any other path.
    """
    global _rowmin_ok
    _probe()
    if _lib is None:
        return None
    if _rowmin_ok is None:
        try:
            _lib.nf_rowwise_min4_argmin_gather.argtypes = (
                [ctypes.c_void_p] * 8 + [ctypes.c_size_t]
                + [ctypes.c_void_p] * 3)
            _lib.nf_rowwise_min4_argmin_gather.restype = ctypes.c_int32
            # Touch the symbol to force lazy Windows lookup now, not mid-call.
            _ = _lib.nf_rowwise_min4_argmin_gather
            _rowmin_ok = True
        except (OSError, AttributeError, ValueError):
            _rowmin_ok = False
    return _lib if _rowmin_ok else None


def rowmin_available():
    """True when the rowmin native entry point is present (env-aware)."""
    return _req_rowmin() is not None


def _guard_lane(a, dtype, label):
    arr = np.asarray(a)
    if arr.dtype == np.dtype(bool):
        raise TypeError(
            f"rowwise_min4 {label} must be {dtype}, got bool. "
            "Fix: pass int32/float32 lanes.")
    if arr.dtype != np.dtype(dtype):
        raise TypeError(
            f"rowwise_min4 {label} must be {dtype}, got {arr.dtype}. "
            f"Fix: pass np.ascontiguousarray(a, dtype=np.{dtype}).")
    if arr.ndim != 1:
        raise ValueError(
            f"rowwise_min4 {label} must be rank-1, got shape {arr.shape}. "
            "Fix: pass flat columns.")
    return np.ascontiguousarray(arr)


def _fb_rowmin(t_lanes, d_lanes):
    """NumPy fallback: strict-less argmin cascade + fancy gather.

    Strict `<` from lane 0 (ties keep the smallest index; a NaN lane
    never wins by `<`, exactly the native kernel order). No Python
    loops, no sorting.
    """
    T = np.stack(t_lanes).astype(np.int32, copy=False)  # (4, N)
    D = np.stack(d_lanes).astype(np.float32, copy=False)  # (4, N)
    n = T.shape[1]
    m = np.zeros(n, dtype=np.uint8)
    best = D[0].copy()
    for k in (1, 2, 3):
        take = D[k] < best
        m = np.where(take, np.uint8(k), m)
        best = np.where(take, D[k], best)
    idx = np.arange(n)
    return (np.ascontiguousarray(T[m, idx]),
            np.ascontiguousarray(best),
            np.ascontiguousarray(m))


def rowwise_min4_argmin_gather(t0, t1, t2, t3, d0, d1, d2, d3):
    """Generic rowwise MIN4 + argmin + gather.

    Inputs: T0-T3 int32[N], D0-D3 float32[N] (1D, contiguous buffers at
    the FFI boundary; non-contiguous views are copied once up front).
    Outputs: (t_best int32[N], d_best float32[N], m_best uint8[N]) with
    m = argmin over the D lanes (ties keep the smallest index),
    t_best = T[m] exact, d_best = D[m] bit-exact. Caller-owned outputs
    (fresh arrays; inputs never mutated). N=0 -> empty outputs without
    touching the backend.
    """
    t_lanes = [_guard_lane(a, "int32", f"T{i}")
               for i, a in enumerate((t0, t1, t2, t3))]
    d_lanes = [_guard_lane(a, "float32", f"D{i}")
               for i, a in enumerate((d0, d1, d2, d3))]
    sizes = {a.size for a in t_lanes} | {a.size for a in d_lanes}
    if len(sizes) != 1:
        raise ValueError(
            f"rowwise_min4 lane size mismatch {sorted(sizes)}. "
            "Fix: pass 8 lanes of one shared length N.")
    n = sizes.pop()
    if n == 0:
        return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.uint8))
    lib = _req_rowmin()
    if lib is None:
        return _fb_rowmin(t_lanes, d_lanes)
    t_best = np.empty(n, dtype=np.int32)
    d_best = np.empty(n, dtype=np.float32)
    m_best = np.empty(n, dtype=np.uint8)
    rc = lib.nf_rowwise_min4_argmin_gather(
        t_lanes[0].ctypes.data, t_lanes[1].ctypes.data,
        t_lanes[2].ctypes.data, t_lanes[3].ctypes.data,
        d_lanes[0].ctypes.data, d_lanes[1].ctypes.data,
        d_lanes[2].ctypes.data, d_lanes[3].ctypes.data, n,
        t_best.ctypes.data, d_best.ctypes.data, m_best.ctypes.data)
    assert rc == 0, rc
    return t_best, d_best, m_best
