# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""P3 cost candidate: thin wrapper over Rust generic costing core (native-first).

NOT WIRED into any existing path (this step only places it alongside):
Segmented.py / adjacency.py / native_cpu.py do not call this module.

The boundary (strictly): a general compute mechanism, no domain semantics.
distance_mm u32 (traffic-independent); travel_time_ms resolution (u64 intermediate,
INF guard, speed==0 -> INF); traffic_k u16 (K_SCALE=1000); CostTable/interning
(vec -> cost_id dedup); directed rows kept separate (not collapsed); parallel
rows carry no MIN. Profile strings live only here (the Python adapter); the Rust
side sees integer codes only.

Canonical: the Rust cost_travel_batch / cost_intern_rows. Python exists only for
compatibility: call the native path when it is present, otherwise a bit-exact
fallback (integer arithmetic, the same INF / K-quart guard).
"""

import ctypes
import os

import numpy as np

_INF = np.int64(4294967295)
K_SCALE = 1000
DEPARTURE_NOMINAL = 255
N_BUCKETS = 24
PROFILE_ID = {"pedestrian": 0, "scooter": 1, "bicycle": 2, "automobile": 3, "truck": 255}

_DLL_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "..", "..", "numfast-native", "target",
                            "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
_DLL_DEFAULT = os.path.abspath(_DLL_DEFAULT)

_lib = None
_why = "unprobed"
_cost_ok = None


def _probe():
    global _lib, _why, _cost_ok
    if os.environ.get("NUMFAST_NATIVE_DISABLE") == "1":
        _lib, _why, _cost_ok = None, "disabled-by-env", False
        return
    path = os.environ.get("NUMFAST_NATIVE_DLL", _DLL_DEFAULT)
    try:
        lib = ctypes.CDLL(path)
        v = ctypes.c_void_p
        z = ctypes.c_size_t
        lib.nf_cost_travel_batch.argtypes = [v, v, v, z, v]
        lib.nf_cost_travel_batch.restype = ctypes.c_int32
        lib.nf_cost_intern.argtypes = [v, z, z, v, v, z]
        lib.nf_cost_intern.restype = ctypes.c_int64
        _ = lib.nf_cost_travel_batch
        _lib, _why, _cost_ok = lib, "loaded:" + path, True
    except (OSError, AttributeError, ValueError) as e:
        _lib, _why, _cost_ok = None, "unavailable:%s" % e, False


def cost_available():
    """True when P3 cost native entry points are present (env-aware)."""
    _probe()
    return bool(_cost_ok)


def _fb_travel(dist, speed, k):
    dist = np.ascontiguousarray(dist, dtype=np.uint32).astype(np.int64)
    speed = np.ascontiguousarray(speed, dtype=np.uint32).astype(np.int64)
    k = np.ascontiguousarray(k, dtype=np.uint16).astype(np.int64)
    out = np.empty(dist.size, dtype=np.uint32)
    for i in range(dist.size):
        d, s, kk = int(dist[i]), int(speed[i]), int(k[i])
        if kk == 0:
            raise ValueError("cost candidate: K == 0")
        if d == 4294967295 or s == 4294967295 or s == 0:
            out[i] = np.uint32(4294967295)
            continue
        v = (d * kk + s // 2) // s
        if v == 0 and d > 0:
            v = 1
        out[i] = np.uint32(4294967295 if v >= 4294967295 else v)
    return out


def cost_travel_native(dist, speed, k):
    """P3 candidate: (out u32 ms, backend) with frozen cost semantics."""
    d = np.ascontiguousarray(np.asarray(dist, dtype=np.uint32))
    s = np.ascontiguousarray(np.asarray(speed, dtype=np.uint32))
    kk = np.ascontiguousarray(np.asarray(k, dtype=np.uint16))
    if not (d.size == s.size == kk.size):
        raise ValueError("cost candidate: lane size mismatch")
    n = int(d.size)
    if n == 0:
        return np.zeros(0, dtype=np.uint32), "numpy"
    _probe()
    if _lib is None:
        return _fb_travel(d, s, kk), "numpy"
    try:
        out = np.empty(n, dtype=np.uint32)
        rc = int(_lib.nf_cost_travel_batch(d.ctypes.data, s.ctypes.data,
                                           kk.ctypes.data, n, out.ctypes.data))
        if rc != 0:
            raise ValueError("cost candidate: travel rejected (rc=%d)" % rc)
        return out, "native"
    except (RuntimeError, ValueError):
        raise
    except Exception:
        return _fb_travel(d, s, kk), "numpy"


def _fb_intern(vecs, n, width):
    ids = np.empty(n, dtype=np.uint32)
    seen = {}
    uniq_rows = []
    for r in range(n):
        key = tuple(int(x) for x in vecs[r * width:(r + 1) * width])
        if key not in seen:
            seen[key] = len(uniq_rows)
            uniq_rows.append(key)
        ids[r] = np.uint32(seen[key])
    flat = np.array([x for row in uniq_rows for x in row], dtype=np.uint32)
    return ids, flat, len(uniq_rows)


def cost_intern_native(vecs, width):
    """P3 candidate: ((ids, uniq_flat, ng), backend) row interning."""
    v = np.ascontiguousarray(np.asarray(vecs, dtype=np.uint32))
    width = int(width)
    if width <= 0:
        raise ValueError("cost candidate: width must be > 0")
    if v.size == 0:
        return (np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint32), 0), "numpy"
    if v.size % width != 0:
        raise ValueError("cost candidate: vecs size not a multiple of width")
    n = v.size // width
    _probe()
    if _lib is None:
        (ids, flat, ng) = _fb_intern(v, n, width)
        return (ids, flat, ng), "numpy"
    try:
        ids = np.empty(n, dtype=np.uint32)
        uniq = np.empty(n * width, dtype=np.uint32)
        ng = int(_lib.nf_cost_intern(v.ctypes.data, n, width,
                                     ids.ctypes.data, uniq.ctypes.data, n * width))
        if ng < 0:
            raise ValueError("cost candidate: intern rejected (rc=%d)" % ng)
        return (ids, uniq[:ng * width].copy(), ng), "native"
    except (RuntimeError, ValueError):
        raise
    except Exception:
        (ids, flat, ng) = _fb_intern(v, n, width)
        return (ids, flat, ng), "numpy"


def bucket_of(departure_time):
    """Adapter-only strings-free bucket: None/255 -> 255, 0..23, epoch % 24."""
    if departure_time is None:
        return DEPARTURE_NOMINAL
    d = int(departure_time)
    if d == DEPARTURE_NOMINAL:
        return DEPARTURE_NOMINAL
    if d < 0:
        raise ValueError("departure_time hour out of 0..23|255")
    if d > 24:
        d = d % N_BUCKETS
    if not 0 <= d <= 23:
        raise ValueError("departure_time hour out of 0..23|255")
    return int(d)


def profile_id(name):
    """Adapter-only name -> integer code (Rust sees only integers)."""
    if name == "truck":
        raise NotImplementedError("truck profile extensible, model not provided")
    try:
        return int(PROFILE_ID[str(name)])
    except KeyError:
        raise ValueError("unknown profile %r" % (name,))
