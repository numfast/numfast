# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Join int64 native lane: Rust hash build+probe over int64 keys, NumPy fallback.

Additive (existing join.py / native.py int32 lane untouched):
- Rust (production DLL, capability-probed symbols):
  nf_join_build_i64 (right int64 keys -> open-addressing table,
  unique-checked) + nf_join_probe_i64 (left int64 keys -> right-row
  pos int64 + hit u8, MT, input order).
- Python maps the raw right-row pos through a rank array to
  SORTED-UNIQUE positions, so native and fallback share one contract:
  positions index the sorted-unique build order (miss -> -1) + hit
  bool mask + k. That is the ir_lookup contract, and join
  materialize (gather pu[pos] at hits) is bit-exact on both paths.
- Fallback (here, local sorted-searchsorted oracle copy): stable
  argsort build -> U[K] sorted-unique + aligned payloads Pu[K]
  (unique-checked) -> vectorized searchsorted probe. Used when the
  DLL/symbols are absent or NUMFAST_NATIVE_DISABLE=1.
Dtypes: int64/int32 keys widened caller-side (uint64 past i64::MAX
rejected, float/bool/text rejected); int32 payloads; int64 pos/chk
accumulators; u8/bool hits. Contiguous caller-owned buffers; inputs
never mutated; outputs freshly allocated. No JIT dep, no benchmark
branches, generic only (no per-query code).
"""

import ctypes as _ct
import os as _os
from pathlib import Path as _Path

import numpy as _np


def _fork_root():
    # Same layout defect as Runtime/Planner/_lib/calibrate.py:_fork_root: a
    # fixed parents[4] is the repo root in the checkout and site-packages in the
    # wheel, where this file is vendored to numfast/_ext/Join/_lib/ (one level
    # shallower). The fork root is the nearest ancestor holding full.toml --
    # parents[4] in the checkout, parents[3] in the wheel, where numfast/
    # full.toml is installed alongside the Extensions. Walk the ancestors: a
    # fixed index is right at one vendoring depth and wrong at the other.
    here = _Path(__file__).resolve()
    for p in (*here.parents, _Path.cwd()):
        try:
            if (p / "full.toml").exists():
                return p
        except OSError:
            continue
    return here.parents[4]


_FORK = _fork_root()

#: Cargo target triple the checkout build tree sits under.
_TARGET_TRIPLE = "x86_64-pc-windows-gnu"


def _candidates(name):
    """Every place `name` legitimately lives, best first.

    A source checkout keeps the cargo build tree at
    <fork>/numfast-native/target/<triple>/release/ AND a staged copy at
    <fork>/src/numfast/_native/. A wheel ships ONLY the built binary at
    <fork>/_native/ (setup.py:_vendor_native_binary copies exactly what
    src/numfast/_native/ holds) and no build tree at all, so the previous
    _FORK/"src"/"numfast"/"_native" probe -- the only other candidate --
    resolved in a checkout and nowhere else.

    The order is the STAGED copy first, then the build tree, because that is
    the precedence this file already had and changing it would change which
    binary a checkout loads. native.py:_candidates lists the build tree first
    for the same reason (it only ever had that one).
    """
    return (str(_FORK / "src" / "numfast" / "_native" / name),
            str(_FORK / "numfast-native" / "target" / _TARGET_TRIPLE
                / "release" / name),
            str(_FORK / "_native" / name))


def _probe(name):
    """(chosen, probed_paths). Never raises and never truncates: when nothing
    is found it returns the FIRST candidate together with every path probed, so
    the failure can name them."""
    seen, paths = set(), []
    for p in _candidates(name):
        if p in seen:
            continue
        seen.add(p)
        paths.append(p)
        if _os.path.exists(p):
            return p, paths
    return paths[0], paths


def _why_missing(path, name, probed):
    """Named failure text listing the probed paths. The list is added only when
    the DEFAULT was in play: with NUMFAST_NATIVE_DLL set the caller already
    named the file."""
    if not _os.path.exists(path) and probed:
        return ("; %s not found. Probed in order: %s. Fix: build "
                "numfast-native and copy the binary into src/numfast/_native/, "
                "or set NUMFAST_NATIVE_DLL." % (name, "; ".join(probed)))
    return ""


_GNU_DEFAULT, _GNU_PROBED = _probe("numfast_native.dll")


def _resolve_default():
    """Production DLL path: canonical env first, else whichever layout holds
    the binary (checkout build tree, checkout staged copy, packaged _native/)."""
    return _os.environ.get("NUMFAST_NATIVE_DLL") or _GNU_DEFAULT


_DLL_PATH = _resolve_default()

_dll = None
_loaded_path = None
_why = "unprobed"


def _lib():
    global _dll, _loaded_path, _why
    if _dll is not None and _loaded_path == _DLL_PATH \
            and _why != "unprobed":
        return _dll
    try:
        lib = _ct.CDLL(_DLL_PATH)
        b = lib.nf_join_build_i64
        b.argtypes = [_ct.c_void_p, _ct.c_size_t, _ct.c_void_p,
                      _ct.c_void_p, _ct.c_void_p, _ct.c_size_t]
        b.restype = _ct.c_int32
        p = lib.nf_join_probe_i64
        p.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p,
                      _ct.c_size_t, _ct.c_void_p, _ct.c_size_t,
                      _ct.c_void_p, _ct.c_void_p, _ct.c_size_t]
        p.restype = _ct.c_int32
        _ = lib.nf_join_build_i64  # force Windows lookup now
        _ = lib.nf_join_probe_i64
        _dll, _loaded_path = lib, _DLL_PATH
        _why = "loaded:" + _DLL_PATH
    except (OSError, AttributeError) as e:
        _dll, _loaded_path = None, _DLL_PATH
        _why = "unavailable:%s%s" % (
            e, _why_missing(_DLL_PATH, "numfast_native.dll", _GNU_PROBED))
        raise RuntimeError("join-i64 native backend %s" % _why) from e
    return _dll


def available():
    """True when the Rust int64 build+probe pair is loaded."""
    if _os.environ.get("NUMFAST_NATIVE_DISABLE") == "1":
        return False
    try:
        _lib()
    except (OSError, RuntimeError):
        return False
    return _dll is not None


def why():
    """Backend state string (loaded path or unavailable reason)."""
    if _os.environ.get("NUMFAST_NATIVE_DISABLE") == "1":
        return "disabled-by-env"
    try:
        _lib()
    except (OSError, RuntimeError):
        pass
    return _why


def cap_for(s):
    """Table capacity: pow2, >= 2*s (load <= 0.5)."""
    cap = 1
    while cap < max(2 * int(s), 1):
        cap *= 2
    return cap


def _as_i64_keys(arr, label):
    a = _np.asarray(arr)
    if a.dtype.kind not in "iu":
        raise ValueError(
            f"I64Join {label} needs int keys, got {a.dtype}")
    if a.dtype == _np.dtype(_np.uint64) and bool((a > _np.int64(2 ** 63 - 1)).any()):
        raise ValueError(
            f"I64Join {label} overflows int64 (uint64 past i64::MAX)")
    return _np.ascontiguousarray(a.ravel(), dtype=_np.int64)


def _as_i32_payload(arr, label):
    a = _np.asarray(arr)
    if a.dtype.kind not in "iu":
        raise ValueError(
            f"I64Join {label} needs int payload, got {a.dtype}")
    v = _np.ascontiguousarray(a.ravel(), dtype=_np.int64)
    if bool(((v < -2 ** 31) | (v > 2 ** 31 - 1)).any()):
        raise ValueError(
            f"I64Join {label} overflows int32")
    return _np.ascontiguousarray(v, dtype=_np.int32)


class NativeJoinBuildI64:
    """Resident right side: Rust hash table + sorted-order images.

    t_keys (int64) / t_pos (int32) / t_occ (u8) lanes + cap: native
    table. u (int64 sorted keys) + pu (int32 aligned payloads) +
    rank (int64 right-pos -> sorted-pos) + k: shared sorted images
    (native probe maps raw right-pos through rank, so probe output
    indexes the sorted-unique order on every path).
    """

    def __init__(self, right_keys, right_v2):
        lib = _lib()
        rk = _as_i64_keys(right_keys, "build keys")
        rv = _as_i32_payload(right_v2, "build payload")
        if rk.size != rv.size:
            raise ValueError(
                f"I64Join build keys/payload {rk.size}!={rv.size}")
        s = int(rk.size)
        cap = cap_for(s)
        self.t_keys = _np.zeros(cap, dtype=_np.int64)
        self.t_pos = _np.zeros(cap, dtype=_np.int32)
        self.t_occ = _np.zeros(cap, dtype=_np.uint8)
        rc = lib.nf_join_build_i64(
            rk.ctypes.data, s, self.t_keys.ctypes.data,
            self.t_pos.ctypes.data, self.t_occ.ctypes.data, cap)
        if rc == -2:
            raise ValueError("I64Join build keys not unique: dupes collapse")
        if rc != 0:
            raise RuntimeError(f"nf_join_build_i64 rc={rc}")
        order = _np.argsort(rk, kind="stable")
        self.u = _np.ascontiguousarray(rk[order])
        self.pu = _np.ascontiguousarray(rv[order])
        self.rank = _np.empty(s, dtype=_np.int64)
        self.rank[order] = _np.arange(s, dtype=_np.int64)
        self.cap = cap
        self.k = s
        self._join_backend = "native-i64"


class NumpyJoinBuildI64:
    """Fallback right side: U[K] sorted keys + Pu[K] payloads."""

    def __init__(self, right_keys, right_v2):
        rk = _as_i64_keys(right_keys, "build keys")
        rv = _as_i32_payload(right_v2, "build payload")
        if rk.size != rv.size:
            raise ValueError(
                f"I64Join build keys/payload {rk.size}!={rv.size}")
        order = _np.argsort(rk, kind="stable")
        u = rk[order]
        if u.size > 1 and bool((u[1:] == u[:-1]).any()):
            raise ValueError("I64Join build keys not unique: dupes collapse")
        self.u = _np.ascontiguousarray(u)
        self.pu = _np.ascontiguousarray(rv[order])
        self.k = int(u.size)
        self._join_backend = "numpy"


def build(right_keys, right_v2):
    """Build the resident right side (native when present, else NumPy)."""
    rk = _as_i64_keys(right_keys, "build keys")
    rv = _as_i32_payload(right_v2, "build payload")
    if rk.size != rv.size:
        raise ValueError(
            f"I64Join build keys/payload {rk.size}!={rv.size}")
    if not available():
        return NumpyJoinBuildI64(rk, rv)
    try:
        return NativeJoinBuildI64(rk, rv)
    except ValueError:
        raise
    except (OSError, RuntimeError):
        return NumpyJoinBuildI64(rk, rv)


def _probe_sorted(build_obj, x_keys, threads=16):
    """Probe -> (pos_sorted int64, hit bool) on every path."""
    xk = _as_i64_keys(x_keys, "probe keys")
    n = int(xk.size)
    k = int(build_obj.k)
    if n == 0:
        return (_np.zeros(0, dtype=_np.int64),
                _np.zeros(0, dtype=bool))
    if getattr(build_obj, "_join_backend", None) == "native-i64":
        lib = _lib()
        pos_raw = _np.zeros(n, dtype=_np.int64)
        hit8 = _np.zeros(n, dtype=_np.uint8)
        rc = lib.nf_join_probe_i64(
            build_obj.t_keys.ctypes.data, build_obj.t_pos.ctypes.data,
            build_obj.t_occ.ctypes.data, build_obj.cap, xk.ctypes.data, n,
            pos_raw.ctypes.data, hit8.ctypes.data, int(threads))
        if rc != 0:
            raise RuntimeError(f"nf_join_probe_i64 rc={rc}")
        hit = _np.ascontiguousarray(hit8.astype(bool))
        pos = _np.where(hit, build_obj.rank[pos_raw], -1).astype(_np.int64)
        return (_np.ascontiguousarray(pos),
                _np.ascontiguousarray(hit))
    u = build_obj.u
    pos = _np.searchsorted(u, xk)
    in_range = pos < k
    hit = _np.zeros(n, dtype=bool)
    if k:
        hit = in_range & (u[_np.where(in_range, pos, 0)] == xk)
    return (_np.ascontiguousarray(_np.where(hit, pos, -1).astype(_np.int64)),
            _np.ascontiguousarray(hit))


def lookup_positions(build_obj, x_keys, threads=16):
    """ir_lookup contract: (positions int64, hit bool, k)."""
    pos, hit = _probe_sorted(build_obj, x_keys, threads)
    return pos, hit, int(build_obj.k)


def join_inner(x_keys, x_v1, build_obj, threads=16):
    """INNER join, probe-order compacted (keys int64, v1, v2 int32)."""
    xv = _as_i32_payload(x_v1, "probe v1")
    xk = _as_i64_keys(x_keys, "probe keys")
    if xv.size != xk.size:
        raise ValueError(
            f"I64Join probe keys/v1 {xk.size}!={xv.size}")
    pos, hit = _probe_sorted(build_obj, xk, threads)
    v2hits = build_obj.pu[pos[hit]]
    return (_np.ascontiguousarray(xk[hit]),
            _np.ascontiguousarray(xv[hit]),
            _np.ascontiguousarray(v2hits))


def join_left(x_keys, x_v1, build_obj, threads=16):
    """LEFT join: full (keys, v1, v2, valid); miss -> v2=0, valid=0."""
    xv = _as_i32_payload(x_v1, "probe v1")
    xk = _as_i64_keys(x_keys, "probe keys")
    if xv.size != xk.size:
        raise ValueError(
            f"I64Join probe keys/v1 {xk.size}!={xv.size}")
    pos, hit = _probe_sorted(build_obj, xk, threads)
    v2g = build_obj.pu[_np.where(hit, pos, 0)]
    o2 = _np.where(hit, v2g, _np.int32(0))
    return (xk, xv, _np.ascontiguousarray(o2),
            _np.ascontiguousarray(hit))


def chk_inner(o1, o2):
    """Exact chk: sum(v1)+sum(v2) int64 accumulators."""
    a = _np.asarray(o1)
    b = _np.asarray(o2)
    return int(a.astype(_np.int64).sum()) if a.size else 0, \
        int(b.astype(_np.int64).sum()) if b.size else 0


def chk_left(o1, o2, valid):
    """Exact chk for LEFT (v2 summed over valid rows only)."""
    a = _np.asarray(o1)
    b = _np.asarray(o2)
    v = _np.asarray(valid, dtype=bool)
    return int(a.astype(_np.int64).sum()) if a.size else 0, \
        int(b[v].astype(_np.int64).sum()) if bool(v.any()) else 0
