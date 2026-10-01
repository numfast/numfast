# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""CpuJoin production: CPU hash-join (int32 keys) as composition of primitives.

Mechanic (existing CPU-driver op semantics, no IR/Planner change):
- build: sort right keys once (stable argsort) -> U[K] sorted-unique +
  aligned payloads Pu[K]. Unique-checked (oracle contract: dict U[K]).
- probe/lookup: vectorized binary search (np.searchsorted, C-speed, no
  per-row Python loop) sharded over T row-partitions (host MT only;
  numpy releases GIL on mass data).
- gather payload: positional take Pu[pos] at hits (gather op semantics:
  output order = indices order).
- materialize: inner -> filter compact (boolean-mask take, probe order
  preserved); left -> full columns + validity sidecar (miss -> valid 0,
  payload 0; NULL == invalid, never NaN/sentinel).
Dtypes: int32 keys/payloads, int64 chk accumulators, float64 display only.
"""

import time as _time
from concurrent.futures import ThreadPoolExecutor as _Pool

import numpy as _np

_MT_MAX = 64
_POOLS = {}


def _pool(t):
    p = _POOLS.get(t)
    if p is None:
        p = _Pool(max_workers=t)
        _POOLS[t] = p
    return p


def make_pair(seed=42, n=1000, s=10, hit_rate=0.9):
    """Synthetic join pair (seed 42, honest SYNTHETIC: no J*.csv on disk).

    right keys unique 0..S-1 shuffled; x keys hit_rate sampled from right
    (dupes ok) + misses above S; v1/v2 randint(1,101) int32.
    """
    rng = _np.random.default_rng(seed)
    rk = rng.permutation(s).astype(_np.int32)
    rv = rng.integers(1, 101, s).astype(_np.int32)
    n_hit = int(n * hit_rate)
    xk = _np.empty(n, dtype=_np.int32)
    if n_hit:
        xk[:n_hit] = rk[rng.integers(0, max(s, 1), n_hit)]
    xk[n_hit:] = rng.integers(s, s + max(1, s // 2), n - n_hit)
    rng.shuffle(xk)
    xv = rng.integers(1, 101, n).astype(_np.int32)
    return xk, xv, rk, rv


class JoinBuild:
    """Resident right side: U[K] sorted keys + Pu[K] aligned payloads."""

    def __init__(self, right_keys, right_v2):
        rk = _np.ascontiguousarray(_np.asarray(right_keys).ravel(),
                                   dtype=_np.int32)
        rv = _np.ascontiguousarray(_np.asarray(right_v2).ravel(),
                                   dtype=_np.int32)
        if rk.size != rv.size:
            raise ValueError(
                f"CpuJoin build keys/payload {rk.size}!={rv.size}")
        order = _np.argsort(rk.astype(_np.int64), kind="stable")
        u = rk[order]
        if u.size > 1 and bool((u[1:] == u[:-1]).any()):
            raise ValueError("CpuJoin build keys not unique: dupes collapse")
        self.u = _np.ascontiguousarray(u)
        self.pu = _np.ascontiguousarray(rv[order])
        self.k = int(u.size)


def build_timed(right_keys, right_v2):
    s = _time.perf_counter()
    b = JoinBuild(right_keys, right_v2)
    return b, (_time.perf_counter() - s) * 1000


def _shard_bounds(n, t):
    t = max(1, min(int(t), n)) if n else 1
    return _np.linspace(0, n, t + 1).astype(_np.int64), t


def _probe_shard(xk, u, k, a, b):
    """Binary-search probe on rows [a:b): (pos int64, hit bool)."""
    xs = xk[a:b]
    pos = _np.searchsorted(u, xs)
    in_range = pos < k
    hit = _np.zeros(xs.shape, dtype=bool)
    safe = _np.where(in_range, pos, 0)
    if k:
        hit = in_range & (u[safe] == xs)
    return pos, hit


def _probe_mt(xk, u, k, t):
    n = int(xk.size)
    if n == 0:
        return _np.zeros(0, dtype=_np.int64), _np.zeros(0, dtype=bool), 0.0
    bounds, t = _shard_bounds(n, t)
    ex = _pool(t)
    s = _time.perf_counter()
    if t == 1:
        pos, hit = _probe_shard(xk, u, k, 0, n)
    else:
        futs = [ex.submit(_probe_shard, xk, u, k, int(bounds[w]),
                          int(bounds[w + 1])) for w in range(t)]
        parts = [f.result() for f in futs]
        pos = _np.concatenate([p[0] for p in parts])
        hit = _np.concatenate([p[1] for p in parts])
    return pos, hit, (_time.perf_counter() - s) * 1000


def join_inner(x_keys, x_v1, build, threads=16):
    """INNER join, probe-order compacted (keys, v1, v2) + stage ms."""
    t = {"build": 0.0, "probe": 0.0, "gather": 0.0, "materialize": 0.0,
         "e2e": 0.0}
    s0 = _time.perf_counter()
    xk = _np.ascontiguousarray(_np.asarray(x_keys).ravel(), dtype=_np.int32)
    xv = _np.ascontiguousarray(_np.asarray(x_v1).ravel(), dtype=_np.int32)
    if xv.size != xk.size:
        raise ValueError(
            f"CpuJoin probe keys/v1 {xk.size}!={xv.size}")
    u, pu, k = build.u, build.pu, build.k
    pos, hit, pms = _probe_mt(xk, u, k, threads)
    t["probe"] = pms
    s = _time.perf_counter()
    v2hits = pu[pos[hit]]  # gather op: positional take at hits
    t["gather"] = (_time.perf_counter() - s) * 1000
    s = _time.perf_counter()
    ok = xk[hit]  # filter op: compact, probe order preserved
    o1 = xv[hit]
    o2 = _np.ascontiguousarray(v2hits)
    t["materialize"] = (_time.perf_counter() - s) * 1000
    t["e2e"] = (_time.perf_counter() - s0) * 1000
    return (ok, o1, o2), t


def join_left(x_keys, x_v1, build, threads=16):
    """LEFT join: full (keys, v1, v2, valid); miss -> v2=0, valid=0 (NULL)."""
    t = {"build": 0.0, "probe": 0.0, "gather": 0.0, "materialize": 0.0,
         "e2e": 0.0}
    s0 = _time.perf_counter()
    xk = _np.ascontiguousarray(_np.asarray(x_keys).ravel(), dtype=_np.int32)
    xv = _np.ascontiguousarray(_np.asarray(x_v1).ravel(), dtype=_np.int32)
    if xv.size != xk.size:
        raise ValueError(
            f"CpuJoin probe keys/v1 {xk.size}!={xv.size}")
    u, pu, k = build.u, build.pu, build.k
    pos, hit, pms = _probe_mt(xk, u, k, threads)
    t["probe"] = pms
    s = _time.perf_counter()
    v2g = pu[_np.where(hit, pos, 0)]  # gather op over full domain
    t["gather"] = (_time.perf_counter() - s) * 1000
    s = _time.perf_counter()
    o2 = _np.where(hit, v2g, _np.int32(0))
    valid = _np.ascontiguousarray(hit)
    t["materialize"] = (_time.perf_counter() - s) * 1000
    t["e2e"] = (_time.perf_counter() - s0) * 1000
    return (xk, xv, _np.ascontiguousarray(o2), valid), t


def chk_inner(o1, o2):
    """Exact chk: sum(v1)+sum(v2) int64 accumulators."""
    a = _np.asarray(o1)
    b = _np.asarray(o2)
    return int(a.astype(_np.int64).sum()) if a.size else 0, \
        int(b.astype(_np.int64).sum()) if b.size else 0


def chk_left(o1, o2, valid):
    a = _np.asarray(o1)
    b = _np.asarray(o2)
    v = _np.asarray(valid, dtype=bool)
    return int(a.astype(_np.int64).sum()) if a.size else 0, \
        int(b[v].astype(_np.int64).sum()) if bool(v.any()) else 0
