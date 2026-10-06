# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Generic batch-assignment pair insert over typeless 64-bit lanes.

Native-first (ctypes `nf_pair_insert_i64`), else bit-exact numpy/Python
fallback (same mul-shift probe + linear probing, input order,
last-write-wins). Caller-owned outputs (fresh arrays; inputs never
mutated). Generic lanes only: callers may view lanes as u64 or i64
(bit-identical); no domain vocabulary. No GPU path (native CPU research
surface, same WASM gate as Sssp: no wasm export).
"""
import ctypes
import os

import numpy as np

_MIX = 0x9E3779B97F4A7C15
_MASK64 = 0xFFFFFFFFFFFFFFFF


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
_pi_ok = None


def _probe():
    global _lib, _why, _probe_env, _pi_ok
    key = (os.environ.get("NUMFAST_NATIVE_DISABLE"),
           os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT))
    if key == _probe_env and _why != "unprobed":
        return
    _probe_env = key
    _pi_ok = None
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
    _probe()
    return _lib is not None


def why():
    _probe()
    return _why


def _req():
    global _pi_ok
    _probe()
    if _lib is None:
        return None
    if _pi_ok is None:
        try:
            _lib.nf_pair_insert_i64.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_size_t]
            _lib.nf_pair_insert_i64.restype = ctypes.c_int64
            _ = _lib.nf_pair_insert_i64
            _pi_ok = True
        except (OSError, AttributeError, ValueError):
            _pi_ok = False
    return _lib if _pi_ok else None


def pair_insert_available():
    return _req() is not None


def _as_i64(a, name):
    x = np.ascontiguousarray(a)
    if x.ndim != 1:
        raise ValueError(f"pair_insert {name} ndim != 1")
    if x.size == 0:
        return np.zeros(0, dtype=np.int64)
    if x.dtype == np.dtype(np.int64):
        return x
    if x.dtype == np.dtype(np.uint64):
        return x.view(np.int64)
    raise ValueError(f"pair_insert {name} dtype {x.dtype} not i64/u64")


def _fb_pair_insert(keys, vals, cap):
    n = int(keys.shape[0])
    tkeys = np.zeros(cap, dtype=np.int64)
    tvals = np.zeros(cap, dtype=np.int64)
    used = np.zeros(cap, dtype=np.uint8)
    mask = cap - 1
    ng = 0
    for i in range(n):
        k = int(keys[i])
        v = int(vals[i])
        ku = k & _MASK64
        slot = (((ku * _MIX) & _MASK64) >> 32) & mask
        for _ in range(cap):
            if int(used[slot]) == 0:
                tkeys[slot] = np.int64(k)
                tvals[slot] = np.int64(v)
                used[slot] = np.uint8(1)
                ng += 1
                break
            if int(tkeys[slot]) == k:
                tvals[slot] = np.int64(v)
                break
            slot = (slot + 1) & mask
    return tkeys, tvals, used, int(ng)


def pair_insert(keys, vals, cap):
    """Insert `keys[i] -> vals[i]` pairs into a fresh open-addressing table.

    Returns `(tkeys, tvals, used, ng)`: `tkeys/tvals[0..cap]` i64,
    `used[0..cap]` u8 0/1, `ng` occupied slots. Duplicate key =
    last-write-wins. `cap` power of two, `cap >= 1`.
    """
    cap = int(cap)
    if cap < 1 or (cap & (cap - 1)) != 0:
        raise ValueError(f"pair_insert cap={cap} not pow2 >= 1")
    k = _as_i64(keys, "keys")
    v = _as_i64(vals, "vals")
    if k.shape[0] != v.shape[0]:
        raise ValueError("pair_insert keys/vals len mismatch")
    lib = _req()
    if lib is None:
        return _fb_pair_insert(k, v, cap)
    n = int(k.shape[0])
    tkeys = np.zeros(cap, dtype=np.int64)
    tvals = np.zeros(cap, dtype=np.int64)
    used = np.zeros(cap, dtype=np.uint8)
    rc = lib.nf_pair_insert_i64(k.ctypes.data, v.ctypes.data, n,
                                tkeys.ctypes.data, tvals.ctypes.data,
                                used.ctypes.data, cap)
    assert rc >= 0, rc
    return tkeys, tvals, used, int(rc)
