# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Multi-source Dijkstra router over CSR u32 lanes with target-set early exit.

Native-first via ctypes to ``nf_router_route`` (numfast-native/src/router.rs),
same probe/dispatch shape as the Sssp extension. Bit-exact heapq fallback
when the native symbol is unavailable.

Frozen semantics (parity with the Rust kernel):
- all valid sources start at dist 0; min-heap of ``(dist, vertex)`` so the
  tie-break is by vertex id, exactly as CPython ``heapq`` on ``(d, u)``;
- stale-pop skip (``d != dist[u]``);
- early exit: the first *popped* (settled) vertex that is in the target set
  wins, its dist is returned and the search stops;
- relax ``nd = d + w`` with saturating add, update iff ``nd < dist[v]``;
- sources out of ``[0, V)`` are ignored (Rust parity);
- no target popped -> ``(INF, -1, popped)``.

Lane convention: same u32 lanes as Sssp. ``UINT32_MAX`` (4294967295) is the
reserved INF marker: edges carrying it are skipped, never relaxed, in both
the native and the fallback path.

No GPU, no heuristics, no domain vocabulary. Generic graph/weights/sources/
targets only. Python owns orchestration, data and the ABI.
"""
import ctypes
import heapq
import os

import numpy as np

_INF = 4294967295
_UNREACHABLE = 9223372036854775807  # INT64_MAX, matches the Rust return
# Rust `router_route_csr` initialises `dist` to `i64::MAX / 4` and uses
# `saturating_add`, so any relax landing at or above this never wins.
_DIST_INF = 9223372036854775807 // 4
# The Rust kernel takes i64 weights and only skips `w < 0`; the reserved
# UINT32_MAX edge marker must be excluded at flatten time (documented in
# router.rs). Mapping it to i64::MAX makes the relax a no-op
# (`saturating_add` -> i64::MAX > any `dist[v]`), which is observationally
# identical to skipping the edge, without rebuilding the CSR.
_W_INF = 9223372036854775807

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", "..", ".."))
_DLL_DEFAULT = os.path.join(_ROOT, "numfast-native", "target",
                            "x86_64-pc-windows-gnu", "release",
                            "numfast_native.dll")

_lib = None
_why = "unprobed"
_probe_env = None
_router_ok = None


def _probe():
    global _lib, _why, _probe_env, _router_ok
    key = (os.environ.get("NUMFAST_NATIVE_DISABLE"),
           os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT))
    if key == _probe_env and _why != "unprobed":
        return
    _probe_env = key
    _router_ok = None
    if os.environ.get("NUMFAST_NATIVE_DISABLE") == "1":
        _lib, _why = None, "disabled-by-env"
        return
    path = os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT)
    try:
        lib = ctypes.CDLL(path)
        _lib, _why = lib, "loaded:" + path
    except (OSError, AttributeError) as e:
        _lib, _why = None, "unavailable:%s" % e


def why():
    _probe()
    return _why


def _req_router():
    global _router_ok
    _probe()
    if _lib is None:
        return None
    if _router_ok is None:
        try:
            _lib.nf_router_route.argtypes = [
                ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            _lib.nf_router_route.restype = ctypes.c_int32
            _ = _lib.nf_router_route
            _router_ok = True
        except (OSError, AttributeError, ValueError):
            _router_ok = False
    return _lib if _router_ok else None


def router_available():
    return _req_router() is not None


def _as_u32(a, name):
    x = np.ascontiguousarray(a)
    if x.ndim != 1:
        raise ValueError(f"router {name} ndim != 1")
    if x.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if x.dtype == np.dtype(np.uint32):
        return x
    if x.dtype.kind not in "iu":
        raise ValueError(f"router {name} dtype {x.dtype} not u32")
    x64 = x.astype(np.int64)
    if bool(np.any((x64 < 0) | (x64 > _INF))) and name != "weights":
        raise ValueError(f"router {name} out of u32 range")
    if name != "weights" and bool(np.any(x64 == _INF)):
        raise ValueError(f"router {name} holds UINT32_MAX (INF/INVALID)")
    return x64.astype(np.uint32)


def _check_csr(ip, ix, w):
    v = int(ip.shape[0]) - 1
    e = int(ix.shape[0])
    if ip.shape[0] == 0 or v < 1:
        raise ValueError("router empty graph (V=0)")
    if w.shape[0] != e:
        raise ValueError(f"router weights len {w.shape[0]} != E={e}")
    if bool(np.any(np.diff(ip.astype(np.int64)) < 0)):
        raise ValueError("router indptr not monotone")
    if int(ip[-1]) != e:
        raise ValueError(f"router indptr[-1]={int(ip[-1])} != E={e}")
    if bool(np.any(ix.astype(np.int64) >= v)):
        raise ValueError("router index out of range")
    return v, e


def _fb_route(ip, ix, w, sources, is_target):
    """heapq fallback, bit-exact with router_route_csr."""
    v = int(ip.shape[0]) - 1
    inf = _DIST_INF
    dist = [inf] * v
    pq = []
    for s in sources:
        if s < 0 or s >= v:
            continue
        if dist[s] > 0:
            dist[s] = 0
            heapq.heappush(pq, (0, s))
    popped = 0
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        popped += 1
        if is_target[u] != 0:
            return d, u, popped
        for e in range(int(ip[u]), int(ip[u + 1])):
            ww = int(w[e])
            if ww == _INF:
                continue
            vv = int(ix[e])
            nd = d + ww
            if nd > inf:
                nd = inf
            if nd < dist[vv]:
                dist[vv] = nd
                heapq.heappush(pq, (nd, vv))
    return _UNREACHABLE, -1, popped


def router_route(indptr, indices, weights, sources, targets):
    """CSR + weights + sources + targets -> (dist, target, visited).

    ``dist`` is the settled distance of the first popped target vertex, or
    ``9223372036854775807`` (INT64_MAX) if no target is reachable.
    ``target`` is that vertex id, or ``-1`` if unreachable.
    ``visited`` counts the settled (non-stale) pops performed.
    """
    ip = _as_u32(indptr, "indptr")
    ix = _as_u32(indices, "indices")
    w = np.ascontiguousarray(weights, dtype=np.uint32)
    v, e = _check_csr(ip, ix, w)
    src = _as_u32(sources, "sources")
    if src.size == 0:
        raise ValueError("router sources empty")
    tgt = np.ascontiguousarray(targets).ravel()
    if tgt.dtype == np.dtype(np.int64) or tgt.dtype == np.dtype(np.uint32):
        pass
    elif tgt.dtype.kind not in "iu":
        raise ValueError(f"router targets dtype {tgt.dtype} not integer")
    tgt64 = tgt.astype(np.int64)
    if tgt.size and bool(np.any((tgt64 < 0) | (tgt64 >= v))):
        raise ValueError("router target out of range")
    is_target = np.zeros(v, dtype=np.uint8)
    is_target[tgt64.astype(np.intp)] = 1

    lib = _req_router()
    if lib is None:
        return _fb_route(ip, ix, w, src.astype(np.int64).tolist(),
                         is_target.tolist())

    ip64 = np.ascontiguousarray(ip.astype(np.int64))
    ix32 = np.ascontiguousarray(ix.astype(np.int32))
    w64 = np.ascontiguousarray(w.astype(np.int64))
    if bool(np.any(w == _INF)):
        w64[w64 == _INF] = _W_INF
    src32 = np.ascontiguousarray(src.astype(np.int32))
    out_dist = np.empty(1, dtype=np.int64)
    out_tgt = np.empty(1, dtype=np.int32)
    out_vis = np.empty(1, dtype=np.int64)
    rc = lib.nf_router_route(v, ip64.ctypes.data, e,
                             ix32.ctypes.data, w64.ctypes.data,
                             src32.ctypes.data, int(src.size),
                             is_target.ctypes.data,
                             out_dist.ctypes.data, out_tgt.ctypes.data,
                             out_vis.ctypes.data)
    assert rc == 0 or rc == 1, rc
    return int(out_dist[0]), int(out_tgt[0]), int(out_vis[0])