# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""import numfast as nf — packaged boundary.

Boot: kernel assembles from full.toml resolved package-relatively
(wheel: numfast/full.toml + numfast/_ext/*; checkout: repo full.toml +
src/*). Native DLL resolves package-relatively (numfast/_native/*);
NUMFAST_NATIVE_DLL env overrides, NUMFAST_NATIVE_DISABLE=1 forces the
numpy fallback. No hardcoded dev paths, no out-of-wheel builder imports.

Typical:
    import numfast as nf
    s = nf.from_numpy(arr)          # 1D -> Series
    t = nf.from_numpy(mat)          # 2D -> Table
    out = nf.to_numpy((s * 2).filter(s > 0))

Internal contract: numfast._* (including _lib, _builder, adapters
internals and any other underscore modules) is NOT part of the public
API and carries no backward-compat promise; the physical import of
such modules is NOT blocked (introspection still works), but new code
must use only the public nf.* names listed in __all__.
"""

_FALLBACK_VERSION = "0.2.1"


def _resolve_version():
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("numfast")
        except PackageNotFoundError:
            return _FALLBACK_VERSION
    except Exception:
        return _FALLBACK_VERSION


__version__ = _resolve_version()

from ._lib.series import Series
from ._lib.table import Table


def _pkg_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent


_PKG = _pkg_dir()
_KERNEL = None


def _package_root():
    from pathlib import Path
    full = _PKG / "full.toml"
    if full.exists():
        return _PKG
    repo = _PKG.parents[1]  # checkout: <root>/src/numfast -> <root>
    if (repo / "full.toml").exists():
        return repo
    raise FileNotFoundError(
        "numfast boot: full.toml not found next to the package nor at the "
        "checkout root. Fix: reinstall the numfast wheel."
    )


def _ensure_native_env():
    from ._lib.native_env import ensure_native_env as _impl
    _impl(_PKG)


def get_kernel(fresh=False):
    """Process kernel singleton (fresh=True rebuilds, e.g. for tests)."""
    global _KERNEL
    if fresh or _KERNEL is None:
        _ensure_native_env()
        from ._builder import build as _build
        root = _package_root()
        kernel = _build(str(root))
        if not fresh:
            _KERNEL = kernel
            return kernel
        return kernel
    return _KERNEL


def native_info():
    """Native backend resolution facts (no private imports cross the boundary).

    Returns {'disabled': bool, 'dll': str|None, 'dll_exists': bool}.
    Whether the engine actually engages the DLL is proven by the
    native enabled/disabled parity tests, not by this probe.
    """
    import os
    get_kernel()
    if os.environ.get("NUMFAST_NATIVE_DISABLE") == "1":
        return {"disabled": True, "dll": None, "dll_exists": False}
    dll = os.environ.get("NUMFAST_NATIVE_DLL")
    if dll is None:
        for cand in (_PKG / "_native" / "numfast_native.dll",
                     _PKG / "_native" / "numfast_native.so"):
            if cand.exists():
                dll = str(cand)
                break
    if dll is None:
        return {"disabled": False, "dll": None, "dll_exists": False}
    from pathlib import Path
    return {"disabled": False, "dll": dll,
            "dll_exists": bool(dll and Path(dll).exists())}


def from_numpy(arr, name="v", names=None, validity=None):
    from .adapters.numpy import from_numpy as _f
    return _f(get_kernel(), arr, name=name, names=names, validity=validity)


def to_numpy(obj):
    from .adapters.numpy import to_numpy as _t
    return _t(obj)


def from_pandas(df, names=None):
    from .adapters.pandas import from_pandas as _f
    return _f(get_kernel(), df, names=names)


def to_pandas(obj):
    from .adapters.pandas import to_pandas as _t
    return _t(obj)


def from_arrow(table, names=None):
    from .adapters.arrow import from_arrow as _f
    return _f(get_kernel(), table, names=names)


def to_arrow(obj):
    from .adapters.arrow import to_arrow as _t
    return _t(obj)


def rng_seed(seed):
    from ._lib.rng_api import check_seed as _c
    return _c(seed)


def rng_fill_i32(n, seed, stream=0, offset=0, lo=0, hi=100, name="rng",
                 backend="cpu"):
    from ._lib.rng_api import fill_i32 as _f
    return _f(get_kernel(), n, seed, stream, offset, lo, hi, name, backend)


def rng_fill_f64(n, seed, stream=0, offset=0, lo=0.0, hi=1.0, name="rng",
                 backend="cpu"):
    from ._lib.rng_api import fill_f64 as _f
    return _f(get_kernel(), n, seed, stream, offset, lo, hi, name, backend)


def rng_sample(n, k, seed, stream=0, offset=0, name="sample", backend="cpu"):
    from ._lib.rng_api import sample as _s
    return _s(get_kernel(), n, k, seed, stream, offset, name, backend)


def rng_permutation(n, seed, stream=0, offset=0, name="perm", backend="cpu"):
    from ._lib.rng_api import permutation as _p
    return _p(get_kernel(), n, seed, stream, offset, name, backend)


def rng_compat(n, seed, kind="runif", lo=0.0, hi=1.0, m=None, name="compat"):
    from ._lib.rng_api import compat as _c
    return _c(get_kernel(), n, seed, kind, lo, hi, m, name)


def map_round(series, ndigits=0, name=None):
    from ._lib.rng_api import map_round as _m
    return _m(get_kernel(), series, ndigits, name)


def unique(series, name=None):
    from ._lib.rng_api import unique as _u
    return _u(get_kernel(), series, name)


def shift(series, periods, name=None):
    from ._lib.rng_api import shift as _s
    return _s(get_kernel(), series, periods, name)


def cumsum(series, name=None):
    from ._lib.rng_api import cumsum as _c
    return _c(get_kernel(), series, name)


def rolling_mean(series, window, min_periods=None, name=None):
    from ._lib.rng_api import rolling_mean as _r
    return _r(get_kernel(), series, window, min_periods, name)


def returns(series, name=None):
    from ._lib.rng_api import returns as _r
    return _r(get_kernel(), series, name)


def lookup(build, probe, name=None):
    from ._lib.rng_api import lookup as _l
    return _l(get_kernel(), build, probe, name)


def app():
    """v0 consumer facade entry point (Extension Semantic/TableExpr).

    One kernel-alias hop, no private import: ``app()`` is the single context
    object that hands out column references, capability facts, the lazy
    nfs-stream reader and the lazy query chain.
    """
    return get_kernel().alias["tableexpr_app"]()


__all__ = [
    "__version__",
    "Series",
    "Table",
    "get_kernel",
    "native_info",
    "app",
    "from_numpy",
    "to_numpy",
    "from_pandas",
    "to_pandas",
    "from_arrow",
    "to_arrow",
    "rng_seed",
    "rng_fill_i32",
    "rng_fill_f64",
    "rng_sample",
    "rng_permutation",
    "rng_compat",
    "map_round",
    "unique",
    "shift",
    "cumsum",
    "rolling_mean",
    "returns",
    "lookup",
]


def __dir__():
    return sorted(__all__)
