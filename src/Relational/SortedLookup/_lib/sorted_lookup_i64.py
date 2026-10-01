# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Generic native sorted-searchsorted int64 lookup/isin lane.

Generic only (no query branches, no per-query code, no benchmark branches):
- sorted-searchsorted over int64 keys: single stable argsort(build) +
  vectorized searchsorted(probe). No hash table, no JIT dep.
- lookup_positions(build, probe) -> (positions int64, hit bool):
  positions[i] = index in build of probe[i], -1 on miss.
- isin(probe, build) -> hit bool (membership, probe order).
- lookup(build, probe) -> (positions, hit) alias of lookup_positions.
- Safe fallback: any unexpected failure -> numpy isin path (correctness
  holds, speed drops). Empty inputs handled explicitly.
- Dtypes: int64 keys (int32/uint widened caller-side); int64 positions;
  bool hits. Contiguous caller-owned buffers; inputs never mutated;
  outputs freshly allocated.
"""

import numpy as _np


def available():
    return True


def why():
    return "sorted-searchsorted-i64:numpy"


def _as_i64(arr, label):
    a = _np.asarray(arr)
    if a.dtype.kind not in "iu":
        raise ValueError(f"SortedLookup {label} needs int keys, got {a.dtype}")
    if a.dtype == _np.dtype(_np.uint64) and bool((a > _np.int64(2 ** 63 - 1)).any()):
        raise ValueError(f"SortedLookup {label} overflows int64")
    return _np.ascontiguousarray(a.ravel(), dtype=_np.int64)


def lookup_positions(build_keys, probe_keys):
    """Sorted-searchsorted positions + hit mask (generic int64)."""
    try:
        b = _as_i64(build_keys, "build keys")
        p = _as_i64(probe_keys, "probe keys")
    except ValueError:
        raise
    except Exception:
        pb = _np.asarray(probe_keys).ravel()
        bb = _np.asarray(build_keys).ravel()
        hit = _np.isin(pb, bb)
        pos = _np.full(int(pb.size), -1, dtype=_np.int64)
        return _np.ascontiguousarray(pos), _np.ascontiguousarray(hit)
    n = int(p.size)
    s = int(b.size)
    if n == 0:
        return _np.zeros(0, dtype=_np.int64), _np.zeros(0, dtype=bool)
    if s == 0:
        return _np.full(n, -1, dtype=_np.int64), _np.zeros(n, dtype=bool)
    try:
        sord = _np.argsort(b, kind="stable")
        sb = b[sord]
        pos = _np.searchsorted(sb, p)
        in_range = pos < s
        safe = _np.where(in_range, pos, 0)
        hit = in_range & (sb[safe] == p)
        out = _np.where(hit, sord[_np.where(hit, pos, 0)], -1).astype(_np.int64)
        return _np.ascontiguousarray(out), _np.ascontiguousarray(hit)
    except Exception:
        hit = _np.isin(p, b)
        pos = _np.full(n, -1, dtype=_np.int64)
        return _np.ascontiguousarray(pos), _np.ascontiguousarray(hit)


def isin_i64(probe_keys, build_keys):
    """Membership lane: probe[i] in build (generic int64)."""
    try:
        b = _as_i64(build_keys, "build keys")
        p = _as_i64(probe_keys, "probe keys")
    except ValueError:
        raise
    except Exception:
        return _np.ascontiguousarray(_np.isin(_np.asarray(probe_keys).ravel(), _np.asarray(build_keys).ravel()))
    n = int(p.size)
    s = int(b.size)
    if n == 0:
        return _np.zeros(0, dtype=bool)
    if s == 0:
        return _np.zeros(n, dtype=bool)
    try:
        sord = _np.argsort(b, kind="stable")
        sb = b[sord]
        pos = _np.searchsorted(sb, p)
        in_range = pos < s
        safe = _np.where(in_range, pos, 0)
        hit = in_range & (sb[safe] == p)
        return _np.ascontiguousarray(hit)
    except Exception:
        return _np.ascontiguousarray(_np.isin(p, b))


def lookup_i64(build_keys, probe_keys):
    return lookup_positions(build_keys, probe_keys)
