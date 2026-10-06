# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical SSSP over CSR u32 lanes (native-first, ctypes, no PyO3).

Generic graph/weights/sources only. No domain vocabulary. No GPU.

Contract: indptr[V+1] u32 monotone + indices[E] u32 + weights[E] u32
+ source -> dist[V] u32 (UINT32_MAX = unreachable). INF edges
(UINT32_MAX weights) skipped, never relaxed. Tie-break by vertex id
(heap order (d, u), same as heapq). Caller-owned outputs (fresh
arrays; inputs never mutated).

Native when the DLL exposes nf_sssp_csr/nf_sssp_batch, else bit-exact
heapq fallback (same order, same saturating add, same INF guard).
"""
import ctypes
import heapq
import os

import numpy as np

_INF = 4294967295


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
_sssp_ok = None


def _probe():
    global _lib, _why, _probe_env, _sssp_ok
    key = (os.environ.get("NUMFAST_NATIVE_DISABLE"),
           os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT))
    if key == _probe_env and _why != "unprobed":
        return
    _probe_env = key
    _sssp_ok = None
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


def _req_sssp():
    global _sssp_ok
    _probe()
    if _lib is None:
        return None
    if _sssp_ok is None:
        try:
            _lib.nf_sssp_csr.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_uint32, ctypes.c_void_p]
            _lib.nf_sssp_csr.restype = ctypes.c_int32
            _lib.nf_sssp_csr_pred.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
            _lib.nf_sssp_csr_pred.restype = ctypes.c_int32
            _lib.nf_sssp_batch.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_size_t]
            _lib.nf_sssp_batch.restype = ctypes.c_int32
            _ = _lib.nf_sssp_csr
            _sssp_ok = True
        except (OSError, AttributeError, ValueError):
            _sssp_ok = False
    return _lib if _sssp_ok else None


def sssp_available():
    return _req_sssp() is not None


def _as_u32(a, name):
    x = np.ascontiguousarray(a)
    if x.ndim != 1:
        raise ValueError(f"sssp {name} ndim != 1")
    if x.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if x.dtype == np.dtype(np.uint32):
        return x
    if x.dtype.kind not in "iu":
        raise ValueError(f"sssp {name} dtype {x.dtype} not u32")
    x64 = x.astype(np.int64)
    if bool(np.any((x64 < 0) | (x64 > 4294967294))) and name != "weights":
        raise ValueError(f"sssp {name} out of u32 range")
    if name != "weights" and bool(np.any(x64 == _INF)):
        raise ValueError(f"sssp {name} holds UINT32_MAX (INF/INVALID)")
    return x64.astype(np.uint32)


def _check_csr(ip, ix, w):
    v = int(ip.shape[0]) - 1
    e = int(ix.shape[0])
    if ip.shape[0] == 0 or v < 1:
        raise ValueError("sssp empty graph (V=0)")
    if w.shape[0] != e:
        raise ValueError(f"sssp weights len {w.shape[0]} != E={e}")
    if bool(np.any(np.diff(ip.astype(np.int64)) < 0)):
        raise ValueError("sssp indptr not monotone")
    if int(ip[-1]) != e:
        raise ValueError(f"sssp indptr[-1]={int(ip[-1])} != E={e}")
    if bool(np.any(ix.astype(np.int64) >= v)):
        raise ValueError("sssp index out of range")
    return v, e


def _fb_sssp(ip, ix, w, source):
    v = int(ip.shape[0]) - 1
    dist = np.full(v, _INF, dtype=np.uint32)
    dist[int(source)] = np.uint32(0)
    pq = [(0, int(source))]
    while pq:
        d, u = heapq.heappop(pq)
        if d != int(dist[u]):
            continue
        for e in range(int(ip[u]), int(ip[u + 1])):
            ww = int(w[e])
            if ww == _INF:
                continue
            vv = int(ix[e])
            nd = d + ww
            if nd >= _INF:
                nd = _INF
            if nd < int(dist[vv]):
                dist[vv] = np.uint32(nd)
                heapq.heappush(pq, (nd, vv))
    return dist


def _fb_pred(ip, ix, w, source):
    v = int(ip.shape[0]) - 1
    dist = np.full(v, _INF, dtype=np.uint32)
    pred = np.full(v, -1, dtype=np.int32)
    dist[int(source)] = np.uint32(0)
    pq = [(0, int(source))]
    while pq:
        d, u = heapq.heappop(pq)
        if d != int(dist[u]):
            continue
        for e in range(int(ip[u]), int(ip[u + 1])):
            ww = int(w[e])
            if ww == _INF:
                continue
            vv = int(ix[e])
            nd = d + ww
            if nd >= _INF:
                nd = _INF
            if nd < int(dist[vv]):
                dist[vv] = np.uint32(nd)
                pred[vv] = np.int32(u)
                heapq.heappush(pq, (nd, vv))
    return dist, pred


def sssp_csr(indptr, indices, weights, source):
    """CSR + weights + source -> dist[V] u32 (INF = unreachable)."""
    ip = _as_u32(indptr, "indptr")
    ix = _as_u32(indices, "indices")
    w = np.ascontiguousarray(weights, dtype=np.uint32)
    v, e = _check_csr(ip, ix, w)
    s = int(np.asarray(source).ravel()[0]) if np.asarray(source).size else -1
    if s < 0 or s >= v:
        raise ValueError(f"sssp source {s} out of [0, {v})")
    lib = _req_sssp()
    if lib is None:
        return _fb_sssp(ip, ix, w, s)
    out = np.empty(v, dtype=np.uint32)
    rc = lib.nf_sssp_csr(ip.ctypes.data, ip.size,
                         ix.ctypes.data, w.ctypes.data, e,
                         s, out.ctypes.data)
    assert rc == 0, rc
    return out


def sssp_csr_pred(indptr, indices, weights, source):
    """CSR + weights + source -> (dist[V] u32, pred[V] i32)."""
    ip = _as_u32(indptr, "indptr")
    ix = _as_u32(indices, "indices")
    w = np.ascontiguousarray(weights, dtype=np.uint32)
    v, e = _check_csr(ip, ix, w)
    s = int(np.asarray(source).ravel()[0]) if np.asarray(source).size else -1
    if s < 0 or s >= v:
        raise ValueError(f"sssp source {s} out of [0, {v})")
    lib = _req_sssp()
    if lib is None:
        return _fb_pred(ip, ix, w, s)
    out = np.empty(v, dtype=np.uint32)
    pred = np.empty(v, dtype=np.int32)
    rc = lib.nf_sssp_csr_pred(ip.ctypes.data, ip.size,
                              ix.ctypes.data, w.ctypes.data, e,
                              s, out.ctypes.data, pred.ctypes.data)
    assert rc == 0, rc
    return out, pred


def sssp_batch(indptr, indices, weights, sources, nthreads=0):
    """CSR + weights + sources[K] -> dist matrix (K, V) u32 row-major."""
    ip = _as_u32(indptr, "indptr")
    ix = _as_u32(indices, "indices")
    w = np.ascontiguousarray(weights, dtype=np.uint32)
    v, e = _check_csr(ip, ix, w)
    src = _as_u32(sources, "sources")
    k = int(src.shape[0])
    if k == 0:
        return np.zeros((0, v), dtype=np.uint32)
    if bool(np.any(src.astype(np.int64) >= v)):
        raise ValueError("sssp source out of range")
    lib = _req_sssp()
    if lib is None:
        rows = [_fb_sssp(ip, ix, w, int(s)) for s in src]
        return np.ascontiguousarray(np.stack(rows))
    out = np.empty((k, v), dtype=np.uint32)
    rc = lib.nf_sssp_batch(ip.ctypes.data, ip.size,
                           ix.ctypes.data, w.ctypes.data, e,
                           src.ctypes.data, k,
                           out.ctypes.data, int(nthreads))
    assert rc == 0, rc
    return out
