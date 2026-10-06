# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Join native production: Rust hash build+probe/gather/fused, NumPy fallback.

Additive extension point (existing join.py untouched):
- Rust (production DLL numfast_native.dll, capability-probed symbols):
  nf_join_build (right keys -> open-addressing table, unique-checked)
  + nf_join_probe (left keys -> pos int64 + hit u8, MT, input order)
  + nf_join_gather_i32 (M2 positional take, MT)
  + nf_join_fused_inner_i32 / nf_join_fused_left_i32 (M3, primary).
- Dispatch (production prod_build/prod_inner/prod_left):
  fused inner/left (primary) -> probe+gather (native gather when
  present, else NumPy take) -> proven NumPy join.py (last fallback,
  available() False: correctness holds, speed drops).
  Capability-probe of symbols, never DLL version/filename.
- NumPy (here, local copy of gather/materialize semantics):
  positional take Pu[pos] at hits + inner compact / left full + chk.
Dtypes: int32 keys/payloads, int64 pos/chk accumulators, u8/bool hits.
"""

import ctypes as _ct
import os as _os
import time as _time
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

# The cargo build tree's target triple is DISCOVERED, never hardcoded.
# "x86_64-pc-windows-gnu" was a Windows-only default: on a Linux host this file
# had no reachable candidate at all, and every candidate it could name pointed
# at a build tree this machine does not have. The directory is the authority;
# which triples are present is the build host's business, not this file's.


def _candidates(name):
    """Every place `name` legitimately lives, best first.

    A source checkout keeps the cargo build tree at
    <fork>/numfast-native/target/<triple>/release/. A wheel ships ONLY the
    built binary at <fork>/_native/ (setup.py:_vendor_native_binary copies
    exactly what src/numfast/_native/ holds) and no build tree at all, so this
    file's single fixed layout missed the library in the installed package.
    All three layouts are probed and the probe list is reported on failure
    rather than swallowed. native_i64.py:_candidates lists the same three in a
    different order -- the precedence each file already had, so that neither
    changes which binary a checkout loads.
    """
def _candidates(name):
    """Every place `name` legitimately lives, in THIS file's established order.

    ORDER IS LOAD-BEARING and is unchanged. The STAGED copy comes before the
    build tree here and after it in native.py, because that is the precedence
    each file already had: changing it would change which binary a checkout
    loads. Only the SET widened -- the triples are enumerated instead of named,
    and both the bare and `lib`-prefixed spelling is offered, because cargo
    writes `libnumfast_native.so` on a Unix host and the staged copy is
    whatever an operator copied.

    The names and the triples come from numfast._lib.native_env, imported
    INSIDE this function. This module is imported by the numfast boot while
    numfast/__init__ is still executing and numfast/_lib/__init__ eagerly
    builds Series/Table, so a module-scope import from here would be a cycle.
    The fallback keeps the single hardcoded Windows candidate working, which is
    what this function resolved to before.
    """
    try:
        from numfast._lib.native_env import build_tree_release_dirs, platform_names
        names = platform_names(_Path(name).stem) or (name,)
        out = [str(d / n) for d in build_tree_release_dirs(_FORK) for n in names]
        out += [str(_FORK / "src" / "numfast" / "_native" / n) for n in names]
        out += [str(_FORK / "_native" / n) for n in names]
        if out:
            return tuple(out)
    except Exception:
        pass
    return (str(_FORK / "src" / "numfast" / "_native" / name),
            str(_FORK / "numfast-native" / "target" / "x86_64-pc-windows-gnu"
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
    """Named failure text listing the probed paths.

    The list is added only when the DEFAULT was in play: with
    NUMFAST_NATIVE_DLL set the caller already named the file, and listing the
    defaults next to it would point at the wrong thing.
    """
    if not _os.path.exists(path) and probed:
        return ("; %s not found. Probed in order: %s. Fix: build "
                "numfast-native and copy the binary into src/numfast/_native/, "
                "or set NUMFAST_NATIVE_DLL." % (name, "; ".join(probed)))
    return ""


_PROD_DLL_DEFAULT, _PROD_DLL_PROBED = _probe("numfast_native.dll")
#: The M3 DLL is a checkout-only artefact: nothing in src/numfast/_native/ holds
#: it, so no wheel carries it and this stays unresolved there by packaging, not
#: by path. Kept a module constant; it is not what _lib() loads.
_M3_DLL_DEFAULT, _M3_DLL_PROBED = _probe("numfast_native_join_m3.dll")


def _resolve_default():
    """Production DLL path: canonical env first, legacy test-DLL envs kept
    for bench BEFORE/AFTER scripts (capability probe decides, not name)."""
    return (_os.environ.get("NUMFAST_NATIVE_DLL")
            or _os.environ.get("NUMFAST_JOIN_M3_DLL")
            or _os.environ.get("NUMFAST_JOIN_M1_DLL")
            or _PROD_DLL_DEFAULT)


_DLL_DEFAULT = _PROD_DLL_DEFAULT
_DLL_PATH = _resolve_default()
_M3_DLL_PATH = _os.environ.get("NUMFAST_JOIN_M3_DLL", _M3_DLL_DEFAULT)

_dll = None
_loaded_path = None
_why = "unprobed"
_gather_ok = None
_fused_ok = None


def _lib():
    global _dll, _loaded_path, _why, _gather_ok, _fused_ok
    if _dll is not None and _loaded_path == _DLL_PATH \
            and _why != "unprobed":
        return _dll
    _gather_ok, _fused_ok = None, None
    try:
        lib = _ct.CDLL(_DLL_PATH)
        b = lib.nf_join_build
        b.argtypes = [_ct.c_void_p, _ct.c_size_t, _ct.c_void_p,
                      _ct.c_void_p, _ct.c_void_p, _ct.c_size_t]
        b.restype = _ct.c_int32
        p = lib.nf_join_probe
        p.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p,
                      _ct.c_size_t, _ct.c_void_p, _ct.c_size_t,
                      _ct.c_void_p, _ct.c_void_p, _ct.c_size_t]
        p.restype = _ct.c_int32
        _ = lib.nf_join_build  # force Windows lookup now, not mid-query
        _dll, _loaded_path = lib, _DLL_PATH
        _why = "loaded:" + _DLL_PATH
    except (OSError, AttributeError) as e:
        _dll, _loaded_path = None, _DLL_PATH
        _why = "unavailable:%s%s" % (
            e, _why_missing(_DLL_PATH, "numfast_native.dll", _PROD_DLL_PROBED))
        raise RuntimeError("join native backend %s" % _why) from e
    try:
        g = lib.nf_join_gather_i32
    except AttributeError:
        g = None
    if g is not None:
        try:
            g.argtypes = [_ct.c_void_p, _ct.c_size_t, _ct.c_void_p,
                          _ct.c_size_t, _ct.c_void_p, _ct.c_size_t]
            g.restype = _ct.c_int32
            _ = lib.nf_join_gather_i32
            _gather_ok = True
        except (OSError, AttributeError, ValueError):
            _gather_ok = False
    else:
        _gather_ok = False
    try:
        fi = lib.nf_join_fused_inner_i32
        fl = lib.nf_join_fused_left_i32
    except AttributeError:
        fi, fl = None, None
    if fi is not None and fl is not None:
        try:
            fi.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p,
                           _ct.c_size_t, _ct.c_void_p, _ct.c_void_p,
                           _ct.c_void_p, _ct.c_size_t, _ct.c_size_t,
                           _ct.c_void_p, _ct.c_void_p, _ct.c_void_p,
                           _ct.c_size_t]
            fi.restype = _ct.c_int64
            fl.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p,
                           _ct.c_size_t, _ct.c_void_p, _ct.c_void_p,
                           _ct.c_void_p, _ct.c_size_t, _ct.c_size_t,
                           _ct.c_void_p, _ct.c_void_p, _ct.c_void_p,
                           _ct.c_void_p, _ct.c_size_t]
            fl.restype = _ct.c_int32
            _ = lib.nf_join_fused_inner_i32
            _fused_ok = True
        except (OSError, AttributeError, ValueError):
            _fused_ok = False
    else:
        _fused_ok = False
    return _dll


def available():
    """True when Rust build+probe backend is loaded (env-aware).

    NUMFAST_NATIVE_DISABLE=1 or missing DLL/symbols -> False: caller
    runs the proven NumPy path (correctness holds, speed drops).
    """
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


def gather_available():
    """True when native positional gather is present (M2 lane)."""
    if not available():
        return False
    return bool(_gather_ok)


def fused_available():
    """True when fused inner+left pair is present (M3 primary lane)."""
    if not available():
        return False
    return bool(_fused_ok)


def cap_for(s):
    """Table capacity: pow2, >= 2*s (load <= 0.5)."""
    cap = 1
    while cap < max(2 * int(s), 1):
        cap *= 2
    return cap


class NativeJoinBuild:
    """Resident right side: Rust hash table (keys/pos/occ lanes)."""

    def __init__(self, right_keys, right_v2):
        lib = _lib()
        rk = _np.ascontiguousarray(_np.asarray(right_keys).ravel(),
                                   dtype=_np.int32)
        rv = _np.ascontiguousarray(_np.asarray(right_v2).ravel(),
                                   dtype=_np.int32)
        if rk.size != rv.size:
            raise ValueError(
                f"NativeJoin build keys/payload {rk.size}!={rv.size}")
        s = int(rk.size)
        cap = cap_for(s)
        self.t_keys = _np.zeros(cap, dtype=_np.int32)
        self.t_pos = _np.zeros(cap, dtype=_np.int32)
        self.t_occ = _np.zeros(cap, dtype=_np.uint8)
        self.pu = _np.ascontiguousarray(rv)
        self.cap = cap
        self.k = s
        rc = lib.nf_join_build(
            rk.ctypes.data, s, self.t_keys.ctypes.data,
            self.t_pos.ctypes.data, self.t_occ.ctypes.data, cap)
        if rc == -2:
            raise ValueError("NativeJoin build keys not unique: "
                             "dupes collapse")
        if rc != 0:
            raise RuntimeError(f"nf_join_build rc={rc}")


def build_timed(right_keys, right_v2):
    s = _time.perf_counter()
    b = NativeJoinBuild(right_keys, right_v2)
    return b, (_time.perf_counter() - s) * 1000


def probe_timed(x_keys, build, threads=16):
    """Rust MT probe -> (pos int64, hit bool) + ms (FFI call timed)."""
    lib = _lib()
    xk = _np.ascontiguousarray(_np.asarray(x_keys).ravel(), dtype=_np.int32)
    n = int(xk.size)
    pos = _np.zeros(n, dtype=_np.int64)
    hit8 = _np.zeros(n, dtype=_np.uint8)
    s = _time.perf_counter()
    rc = lib.nf_join_probe(
        build.t_keys.ctypes.data, build.t_pos.ctypes.data,
        build.t_occ.ctypes.data, build.cap, xk.ctypes.data, n,
        pos.ctypes.data, hit8.ctypes.data, int(threads))
    ms = (_time.perf_counter() - s) * 1000
    if rc != 0:
        raise RuntimeError(f"nf_join_probe rc={rc}")
    return pos, _np.ascontiguousarray(hit8.astype(bool)), ms, xk


def join_inner(x_keys, x_v1, build, threads=16):
    """INNER join, probe-order compacted (keys, v1, v2) + stage ms."""
    t = {"build": 0.0, "probe": 0.0, "gather": 0.0, "materialize": 0.0,
         "e2e": 0.0}
    s0 = _time.perf_counter()
    xv = _np.ascontiguousarray(_np.asarray(x_v1).ravel(), dtype=_np.int32)
    xk = _np.ascontiguousarray(_np.asarray(x_keys).ravel(), dtype=_np.int32)
    if xv.size != xk.size:
        raise ValueError(
            f"NativeJoin probe keys/v1 {xk.size}!={xv.size}")
    pos, hit, pms, xk = probe_timed(xk, build, threads)
    t["probe"] = pms
    s = _time.perf_counter()
    v2hits = build.pu[pos[hit]]  # gather op: positional take at hits
    t["gather"] = (_time.perf_counter() - s) * 1000
    s = _time.perf_counter()
    ok = xk[hit]  # filter op: compact, probe order preserved
    o1 = xv[hit]
    o2 = _np.ascontiguousarray(v2hits)
    t["materialize"] = (_time.perf_counter() - s) * 1000
    t["e2e"] = (_time.perf_counter() - s0) * 1000
    return (ok, o1, o2), t


def join_left(x_keys, x_v1, build, threads=16):
    """LEFT join: full (keys, v1, v2, valid); miss -> v2=0, valid=0."""
    t = {"build": 0.0, "probe": 0.0, "gather": 0.0, "materialize": 0.0,
         "e2e": 0.0}
    s0 = _time.perf_counter()
    xv = _np.ascontiguousarray(_np.asarray(x_v1).ravel(), dtype=_np.int32)
    xk = _np.ascontiguousarray(_np.asarray(x_keys).ravel(), dtype=_np.int32)
    if xv.size != xk.size:
        raise ValueError(
            f"NativeJoin probe keys/v1 {xk.size}!={xv.size}")
    pos, hit, pms, xk = probe_timed(xk, build, threads)
    t["probe"] = pms
    s = _time.perf_counter()
    v2g = build.pu[_np.where(hit, pos, 0)]  # gather op over full domain
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


def gather_timed(payload, pos, threads=16):
    """M2: Rust MT positional gather payload[pos] -> (out int32, ms).

    Caller-owned contiguous buffers; pos in [0, s). Index prep
    (boolean-take / where) stays caller-side inside the gather stage
    (same boundary as the NumPy take it replaces).
    """
    lib = _lib()
    g = getattr(lib, "nf_join_gather_i32", None)
    if g is None:
        raise RuntimeError("nf_join_gather_i32 absent (need join_m2 DLL)")
    pu = _np.ascontiguousarray(_np.asarray(payload).ravel(),
                               dtype=_np.int32)
    ps = _np.ascontiguousarray(_np.asarray(pos).ravel(),
                               dtype=_np.int64)
    m = int(ps.size)
    out = _np.zeros(m, dtype=_np.int32)
    s = _time.perf_counter()
    rc = g(pu.ctypes.data, int(pu.size), ps.ctypes.data, m,
           out.ctypes.data, int(threads))
    ms = (_time.perf_counter() - s) * 1000
    if rc != 0:
        raise RuntimeError(f"nf_join_gather_i32 rc={rc}")
    return out, ms


def join_inner_g(x_keys, x_v1, build, threads=16):
    """M2 INNER: Rust probe + Rust gather, NumPy materialize (unchanged)."""
    t = {"build": 0.0, "probe": 0.0, "gather": 0.0, "materialize": 0.0,
         "e2e": 0.0}
    s0 = _time.perf_counter()
    xv = _np.ascontiguousarray(_np.asarray(x_v1).ravel(), dtype=_np.int32)
    xk = _np.ascontiguousarray(_np.asarray(x_keys).ravel(), dtype=_np.int32)
    if xv.size != xk.size:
        raise ValueError(
            f"NativeJoin probe keys/v1 {xk.size}!={xv.size}")
    pos, hit, pms, xk = probe_timed(xk, build, threads)
    t["probe"] = pms
    s = _time.perf_counter()
    v2hits, _ = gather_timed(build.pu, pos[hit], threads)
    t["gather"] = (_time.perf_counter() - s) * 1000
    s = _time.perf_counter()
    ok = xk[hit]  # filter op: compact, probe order preserved
    o1 = xv[hit]
    o2 = _np.ascontiguousarray(v2hits)
    t["materialize"] = (_time.perf_counter() - s) * 1000
    t["e2e"] = (_time.perf_counter() - s0) * 1000
    return (ok, o1, o2), t


def join_left_g(x_keys, x_v1, build, threads=16):
    """M2 LEFT: Rust probe + Rust gather, NumPy materialize (unchanged)."""
    t = {"build": 0.0, "probe": 0.0, "gather": 0.0, "materialize": 0.0,
         "e2e": 0.0}
    s0 = _time.perf_counter()
    xv = _np.ascontiguousarray(_np.asarray(x_v1).ravel(), dtype=_np.int32)
    xk = _np.ascontiguousarray(_np.asarray(x_keys).ravel(), dtype=_np.int32)
    if xv.size != xk.size:
        raise ValueError(
            f"NativeJoin probe keys/v1 {xk.size}!={xv.size}")
    pos, hit, pms, xk = probe_timed(xk, build, threads)
    t["probe"] = pms
    s = _time.perf_counter()
    v2g, _ = gather_timed(build.pu, _np.where(hit, pos, 0), threads)
    t["gather"] = (_time.perf_counter() - s) * 1000
    s = _time.perf_counter()
    o2 = _np.where(hit, v2g, _np.int32(0))
    valid = _np.ascontiguousarray(hit)
    t["materialize"] = (_time.perf_counter() - s) * 1000
    t["e2e"] = (_time.perf_counter() - s0) * 1000
    return (xk, xv, _np.ascontiguousarray(o2), valid), t


def fused_inner(x_keys, x_v1, build, threads=16):
    """M3 INNER: Rust fused probe->compact (keys, v1, v2) in one call.

    No pos/pos[hit]/fancy-indexing temps, no separate materialize.
    Caller allocates n-lane outputs; kernel returns compact hits m.
    """
    t = {"build": 0.0, "fused": 0.0, "e2e": 0.0}
    s0 = _time.perf_counter()
    lib = _lib()
    f = getattr(lib, "nf_join_fused_inner_i32", None)
    if f is None:
        raise RuntimeError("nf_join_fused_inner_i32 absent (need join_m3 DLL)")
    xv = _np.ascontiguousarray(_np.asarray(x_v1).ravel(), dtype=_np.int32)
    xk = _np.ascontiguousarray(_np.asarray(x_keys).ravel(), dtype=_np.int32)
    if xv.size != xk.size:
        raise ValueError(
            f"NativeJoin probe keys/v1 {xk.size}!={xv.size}")
    n = int(xk.size)
    ok = _np.zeros(n, dtype=_np.int32)
    o1 = _np.zeros(n, dtype=_np.int32)
    o2 = _np.zeros(n, dtype=_np.int32)
    s = _time.perf_counter()
    rc = f(build.t_keys.ctypes.data, build.t_pos.ctypes.data,
           build.t_occ.ctypes.data, build.cap, xk.ctypes.data,
           xv.ctypes.data, build.pu.ctypes.data, int(build.pu.size), n,
           ok.ctypes.data, o1.ctypes.data, o2.ctypes.data, int(threads))
    t["fused"] = (_time.perf_counter() - s) * 1000
    if rc < 0:
        raise RuntimeError(f"nf_join_fused_inner_i32 rc={rc}")
    m = int(rc)
    t["e2e"] = (_time.perf_counter() - s0) * 1000
    return (ok[:m], o1[:m], o2[:m]), t


def fused_left(x_keys, x_v1, build, threads=16):
    """M3 LEFT: Rust fused probe->full output + valid NULL sidecar.

    Miss -> v2 = 0, valid = 0 (same NULL contract as join_left).
    """
    t = {"build": 0.0, "fused": 0.0, "e2e": 0.0}
    s0 = _time.perf_counter()
    lib = _lib()
    f = getattr(lib, "nf_join_fused_left_i32", None)
    if f is None:
        raise RuntimeError("nf_join_fused_left_i32 absent (need join_m3 DLL)")
    xv = _np.ascontiguousarray(_np.asarray(x_v1).ravel(), dtype=_np.int32)
    xk = _np.ascontiguousarray(_np.asarray(x_keys).ravel(), dtype=_np.int32)
    if xv.size != xk.size:
        raise ValueError(
            f"NativeJoin probe keys/v1 {xk.size}!={xv.size}")
    n = int(xk.size)
    ok = _np.zeros(n, dtype=_np.int32)
    o1 = _np.zeros(n, dtype=_np.int32)
    o2 = _np.zeros(n, dtype=_np.int32)
    valid = _np.zeros(n, dtype=_np.uint8)
    s = _time.perf_counter()
    rc = f(build.t_keys.ctypes.data, build.t_pos.ctypes.data,
           build.t_occ.ctypes.data, build.cap, xk.ctypes.data,
           xv.ctypes.data, build.pu.ctypes.data, int(build.pu.size), n,
           ok.ctypes.data, o1.ctypes.data, o2.ctypes.data,
           valid.ctypes.data, int(threads))
    t["fused"] = (_time.perf_counter() - s) * 1000
    if rc != 0:
        raise RuntimeError(f"nf_join_fused_left_i32 rc={rc}")
    t["e2e"] = (_time.perf_counter() - s0) * 1000
    return (ok, o1, o2, _np.ascontiguousarray(valid.astype(bool))), t


# ---------------- production dispatch (proven wiring only) ----------------
# Chain: fused (primary) -> probe+gather (native gather when present,
# else NumPy take) -> proven NumPy join.py (last fallback). No new
# algorithms: every lane reuses the M1/M2/M3 functions above or join.py.
# Error contract == join.py (CpuJoin texts); native ValueErrors are
# mapped, never surfaced with a different prefix.

def _oj():
    """Proven NumPy fallback (same Extension, lazy: no import-time cost)."""
    try:
        from .join import JoinBuild as _B
        from .join import build_timed as _bt
        from .join import join_inner as _ji
        from .join import join_left as _jl
    except ImportError:  # Builder mount (file-location, no sys.path)
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_nf_join_fallback",
            str(_Path(__file__).resolve().parent / "join.py"))
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _B = _mod.JoinBuild
        _bt = _mod.build_timed
        _ji = _mod.join_inner
        _jl = _mod.join_left
    return _B, _bt, _ji, _jl


def _cpu_build_err(a, b):
    return ValueError(f"CpuJoin build keys/payload {a}!={b}")


def _cpu_dupe_err():
    return ValueError("CpuJoin build keys not unique: dupes collapse")


def _cpu_probe_err(a, b):
    return ValueError(f"CpuJoin probe keys/v1 {a}!={b}")


def prod_build(right_keys, right_v2):
    """Production build: native table when available, else proven NumPy.

    Returns (build, ms); build carries _join_backend ("native"/"numpy").
    Contract errors (size/dupe) raise the exact join.py ValueErrors.
    """
    rk = _np.ascontiguousarray(_np.asarray(right_keys).ravel(),
                               dtype=_np.int32)
    rv = _np.ascontiguousarray(_np.asarray(right_v2).ravel(),
                               dtype=_np.int32)
    if rk.size != rv.size:
        raise _cpu_build_err(rk.size, rv.size)
    if not available():
        _, bt, _, _ = _oj()
        b, ms = bt(rk, rv)
        b._join_backend = "numpy"
        return b, ms
    try:
        b, ms = build_timed(rk, rv)
    except ValueError as e:
        m = str(e)
        if "not unique" in m:
            raise _cpu_dupe_err() from None
        if "keys/payload" in m:
            raise _cpu_build_err(rk.size, rv.size) from None
        raise
    b._join_backend = "native"
    return b, ms


def _is_numpy_build(build):
    return getattr(build, "_join_backend", None) == "numpy" \
        or not hasattr(build, "t_keys")


def prod_inner(x_keys, x_v1, build, threads=16):
    """Production INNER: fused -> probe+gather -> proven NumPy."""
    xv = _np.ascontiguousarray(_np.asarray(x_v1).ravel(), dtype=_np.int32)
    xk = _np.ascontiguousarray(_np.asarray(x_keys).ravel(), dtype=_np.int32)
    if xv.size != xk.size:
        raise _cpu_probe_err(xk.size, xv.size)
    if _is_numpy_build(build):
        _, _, ji, _ = _oj()
        (ok, o1, o2), t = ji(xk, xv, build, threads=threads)
        t["backend"] = "numpy"
        return (ok, o1, o2), t
    if fused_available():
        try:
            (ok, o1, o2), t = fused_inner(xk, xv, build, threads=threads)
        except ValueError:
            raise _cpu_probe_err(xk.size, xv.size) from None
        t["backend"] = "fused"
        return (ok, o1, o2), t
    if gather_available():
        try:
            (ok, o1, o2), t = join_inner_g(xk, xv, build, threads=threads)
        except ValueError:
            raise _cpu_probe_err(xk.size, xv.size) from None
        t["backend"] = "probe+gather-native"
        return (ok, o1, o2), t
    if available():
        try:
            (ok, o1, o2), t = join_inner(xk, xv, build, threads=threads)
        except ValueError:
            raise _cpu_probe_err(xk.size, xv.size) from None
        t["backend"] = "probe+gather-numpy"
        return (ok, o1, o2), t
    _, _, ji, _ = _oj()
    (ok, o1, o2), t = ji(xk, xv, _oj_rebuild(build), threads=threads)
    t["backend"] = "numpy"
    return (ok, o1, o2), t


def prod_left(x_keys, x_v1, build, threads=16):
    """Production LEFT: fused -> probe+gather -> proven NumPy."""
    xv = _np.ascontiguousarray(_np.asarray(x_v1).ravel(), dtype=_np.int32)
    xk = _np.ascontiguousarray(_np.asarray(x_keys).ravel(), dtype=_np.int32)
    if xv.size != xk.size:
        raise _cpu_probe_err(xk.size, xv.size)
    if _is_numpy_build(build):
        _, _, _, jl = _oj()
        (ok, o1, o2, valid), t = jl(xk, xv, build, threads=threads)
        t["backend"] = "numpy"
        return (ok, o1, o2, valid), t
    if fused_available():
        try:
            (ok, o1, o2, valid), t = fused_left(xk, xv, build,
                                               threads=threads)
        except ValueError:
            raise _cpu_probe_err(xk.size, xv.size) from None
        t["backend"] = "fused"
        return (ok, o1, o2, valid), t
    if gather_available():
        try:
            (ok, o1, o2, valid), t = join_left_g(xk, xv, build,
                                                threads=threads)
        except ValueError:
            raise _cpu_probe_err(xk.size, xv.size) from None
        t["backend"] = "probe+gather-native"
        return (ok, o1, o2, valid), t
    if available():
        try:
            (ok, o1, o2, valid), t = join_left(xk, xv, build,
                                              threads=threads)
        except ValueError:
            raise _cpu_probe_err(xk.size, xv.size) from None
        t["backend"] = "probe+gather-numpy"
        return (ok, o1, o2, valid), t
    _, _, _, jl = _oj()
    (ok, o1, o2, valid), t = jl(xk, xv, _oj_rebuild(build),
                                threads=threads)
    t["backend"] = "numpy"
    return (ok, o1, o2, valid), t


def _oj_rebuild(build):
    """Native build -> equivalent NumPy build (last-resort fallback).

    Only reached when the DLL vanished mid-run (build ok, probe gone):
    rebuilds sorted-unique state from the resident payload is impossible
    without right keys, so this path raises loudly instead of silently
    degrading. Kept as an explicit error branch (never silent).
    """
    raise RuntimeError(
        "join native backend lost mid-run (build ok, probe gone)")
