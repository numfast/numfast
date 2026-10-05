# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Reusable prepared-router surface: hoist the per-call prologue.

``router.py`` pays, on every call, for work that depends only on the CSR
lanes and never on the query: validation of indptr/indices/weights, the
u32->i64 widening of indptr/weights, the u32->i32 narrowing of indices,
the UINT32_MAX -> INT64_MAX weight remap, and the ``is_target`` buffer
allocation. Measured on the real portal graph (V=241348, E=938249) that
prologue is ~10.5-10.8 ms per call and ~15.6 ms of marginal cost once
cold-allocation effects are included, against a ~44-49 ms full Dijkstra.

``RouterPlan`` performs that work once at construction and keeps the
prepared lanes resident, then answers queries through the same
``nf_router_route`` symbol with the same argument order and the same
frozen semantics. ``route_many`` amortises the remaining per-query cost
(FFI call + heap) over a batch of queries against one prepared graph.

``router_route`` in ``router.py`` is untouched and stays bit-identical;
this module is an additional surface, not a replacement. Frozen semantics
(settled-pop target, stale-pop skip, saturating add, tie-break by vertex
id, sources outside [0, V) ignored, UINT32_MAX edges never relaxed) are
identical because the same native kernel decides them.
"""
import numpy as np

from . import router as _rt

_UNREACHABLE = 9223372036854775807  # INT64_MAX, matches the Rust return


class RouterPlan:
    """Prepared CSR lanes + resident target mask for repeated routing.

    Construction validates and widens once; ``route``/``route_many`` do no
    lane work. Inputs are never mutated.
    """

    __slots__ = ("_v", "_e", "_ip64", "_ix32", "_w64", "_is_target",
                 "_out_dist", "_out_tgt", "_out_vis", "_inf_edges")

    def __init__(self, indptr, indices, weights):
        ip = _rt._as_u32(indptr, "indptr")
        ix = _rt._as_u32(indices, "indices")
        w = np.ascontiguousarray(weights, dtype=np.uint32)
        v, e = _rt._check_csr(ip, ix, w)

        # hoisted widening: done once, resident for the lifetime of the plan
        ip64 = np.ascontiguousarray(ip.astype(np.int64))
        ix32 = np.ascontiguousarray(ix.astype(np.int32))
        w64 = np.ascontiguousarray(w.astype(np.int64))
        inf_edges = int(np.count_nonzero(w == np.uint32(_rt._INF)))
        if inf_edges:
            w64[w64 == _rt._INF] = _rt._W_INF

        self._v = v
        self._e = e
        self._ip64 = ip64
        self._ix32 = ix32
        self._w64 = w64
        self._inf_edges = inf_edges
        self._is_target = np.zeros(v, dtype=np.uint8)
        self._out_dist = np.empty(1, dtype=np.int64)
        self._out_tgt = np.empty(1, dtype=np.int32)
        self._out_vis = np.empty(1, dtype=np.int64)

    # ---- introspection -------------------------------------------------
    @property
    def n_vertices(self):
        return self._v

    @property
    def n_edges(self):
        return self._e

    @property
    def n_inf_edges(self):
        """UINT32_MAX edges remapped to INT64_MAX at prepare time."""
        return self._inf_edges

    # ---- queries -------------------------------------------------------
    def _check_sources(self, sources):
        src = _rt._as_u32(sources, "sources")
        if src.size == 0:
            raise ValueError("router sources empty")
        return src

    def _check_targets(self, targets):
        tgt = np.ascontiguousarray(targets).ravel()
        if tgt.dtype.kind not in "iu":
            raise ValueError(f"router targets dtype {tgt.dtype} not integer")
        tgt64 = tgt.astype(np.int64)
        if tgt.size and bool(np.any((tgt64 < 0) | (tgt64 >= self._v))):
            raise ValueError("router target out of range")
        return tgt64

    def _mask_targets(self, tgt64):
        """Resident u8 mask, reset then set. Same mask the Rust kernel reads."""
        is_t = self._is_target
        is_t[:] = 0
        if tgt64.size:
            is_t[tgt64.astype(np.intp)] = 1
        return is_t

    def route(self, sources, targets):
        """Same contract as ``router_route``, prepared lanes.

        Returns ``(dist, target, visited)``.
        """
        src = self._check_sources(sources)
        tgt64 = self._check_targets(targets)
        is_t = self._mask_targets(tgt64)
        lib = _rt._req_router()
        if lib is None:
            # bit-exact heapq fallback on the same prepared lanes
            return _rt._fb_route(self._ip64.astype(np.uint32),
                                 self._ix32.astype(np.uint32),
                                 np.ascontiguousarray(
                                     self._w64.astype(np.uint32)),
                                 src.astype(np.int64).tolist(), is_t.tolist())
        src32 = np.ascontiguousarray(src.astype(np.int32))
        rc = lib.nf_router_route(
            self._v, self._ip64.ctypes.data, self._e,
            self._ix32.ctypes.data, self._w64.ctypes.data,
            src32.ctypes.data, int(src.size),
            is_t.ctypes.data,
            self._out_dist.ctypes.data, self._out_tgt.ctypes.data,
            self._out_vis.ctypes.data)
        assert rc == 0 or rc == 1, rc
        return (int(self._out_dist[0]), int(self._out_tgt[0]),
                int(self._out_vis[0]))

    def route_many(self, pairs):
        """Route K (source, target) queries against the prepared lanes.

        ``pairs`` is an iterable of ``(source, target)``; sources may be a
        1-element sequence per query. One prepared graph, K queries, no lane
        work between them. Returns a list of ``(dist, target, visited)`` in
        input order.
        """
        out = []
        for source, target in pairs:
            s = np.asarray(source, dtype=np.uint32).ravel()
            t = np.asarray(target, dtype=np.int64).ravel()
            out.append(self.route(s, t))
        return out


__all__ = ["RouterPlan", "_UNREACHABLE"]