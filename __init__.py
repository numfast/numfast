"""NumFast -- GPU-first numerical computing framework.

Usage:
    import numfast as nf

    # Creation
    x = nf.zeros(5)
    r = nf.random.uniform(low=0.0, high=1.0, shape=10, seed=42)

    # Series
    s = nf.series([1.0, 2.0, 3.0])

    # Math (lazy expression nodes) + top-level aliases
    y = nf.sin(nf.series([0, 1.57]))

    # Stats / ops
    nf.total([1, 2, 3])
    nf.scan(data)

    # Tabular
    nf.topk(data, k=5)
    nf.groupby(keys, values)

    # Diagnostics
    nf.device_info()
    nf.set_backend("cpu")   # "cpu" | "gpu" | "auto"

Public API is a strict whitelist. Internal machinery (Runtime.compile,
drivers, kernel registration) is available only via ``nf._internal``.
"""

import sys
import pathlib
import types
import tomllib
import importlib.util

__version__ = "1.0.0a1"

# === Path setup ===
_base_dir = pathlib.Path(__file__).resolve().parent

# Add src/ and group directories to sys.path (kernel + backend modules).
_src = _base_dir / 'src'
for _sub in ('', 'core', 'math'):
    _p = str(_src / _sub) if _sub else str(_src)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Add APP Builder to path when present (optional -- static fallback exists).
_ab_candidates = [
    _base_dir.parent / 'app-builder',
    _base_dir / 'app-builder',
]
for _c in _ab_candidates:
    if _c.is_dir() and (_c / 'builder').is_dir():
        _ab_path = str(_c.resolve())
        if _ab_path not in sys.path:
            sys.path.insert(0, _ab_path)
        break

# === Lazy kernel build -- deferred until first attribute access ===
_kernel = None
_kernel_built = False
_build_error = None
_context = None
_ns_cache = {}
_internal_shim = None


def _purge_lib_modules():
    """Drop bare '_lib' modules so the next extension's '_lib' resolves fresh."""
    for key in list(sys.modules):
        if key == '_lib' or key.startswith('_lib.'):
            del sys.modules[key]


class _StaticKernel:
    """Minimal alias layer -- fallback when app-builder is unavailable (D3).

    Provides only what numfast's extension loading loop needs:
    alias registration and metadata dict. Kernels are registered
    statically: Compute.setup() finds 'register_kernel' in alias
    (registered by Runtime extension) and registers every compute
    primitive directly -- same path as Compute.register_all.
    """

    def __init__(self, name="numfast", singleton=True):
        self.name = name
        self.singleton = singleton
        self.alias = {}
        self.metadata = {}

    def register(self, name, func, owner=None):
        self.alias[name] = func


def _load_extension(base, ext_cfg, kernel):
    """Load one AppBuilder extension and register it on the kernel."""
    ext_path = (base / ext_cfg["path"]).resolve()
    ext_name = ext_path.name

    ext_manifest_path = ext_path / f"{ext_name}.toml"
    if not ext_manifest_path.exists():
        return
    py_path = ext_path / f"{ext_name}.py"
    if not py_path.exists():
        return
    em = tomllib.loads(ext_manifest_path.read_text(encoding="utf-8"))

    # Load entry <Name>.py AS a package (submodule_search_locations set),
    # so both import styles work:
    #   from Ext._lib.mod import ...   (qualified, via sys.modules[ext_name])
    #   from _lib.mod import ...       (bare, via ext_path on sys.path)
    spec = importlib.util.spec_from_file_location(
        ext_name, str(py_path),
        submodule_search_locations=[str(ext_path)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[ext_name] = mod

    # Bare '_lib' imports need the extension dir itself on sys.path.
    sys.path.insert(0, str(ext_path))
    _purge_lib_modules()
    try:
        spec.loader.exec_module(mod)
        # <Ext>.py is both the extension entry and a potential submodule
        # ('from Ext.Ext import x' is used by some extension internals).
        # Pre-bind the self-submodule name to THIS module so later
        # 'Ext.Ext' imports never shadow entry attributes with a
        # freshly-loaded copy of the same file.
        sys.modules[f"{ext_name}.{ext_name}"] = mod
    finally:
        sys.path.remove(str(ext_path))
        _purge_lib_modules()

    # Register aliases by matching mod names to module attributes.
    for alias_name, mod_name in zip(em.get("alias", []), em.get("mods", [])):
        func = getattr(mod, mod_name, None)
        if func is not None:
            kernel.register(alias_name, func, owner=ext_name)

    # Call setup(kernel) if module has it.
    if hasattr(mod, "setup"):
        mod.setup(kernel)

    # Merge toml metadata into kernel.
    if "metadata" in em:
        kernel.metadata.setdefault(ext_name, {})
        kernel.metadata[ext_name].update(em["metadata"])


def _ensure_built():
    global _kernel, _kernel_built, _build_error
    if _kernel_built:
        return
    _kernel_built = True

    _self_module = sys.modules.get(__name__)
    base = _base_dir.resolve()

    try:
        # D3 fallback: use Builder Kernel when app-builder is present,
        # otherwise static registration via minimal alias layer.
        try:
            from builder import MAIN
            Kernel = MAIN["Kernel"]
            mode = "builder"
        except Exception:
            Kernel = _StaticKernel
            mode = "static"

        manifest_path = base / "full.toml"
        app = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        kernel = Kernel(
            name=app["kernel"]["name"],
            singleton=app["kernel"].get("singleton", True),
        )
        kernel.metadata.setdefault("_build", {})
        kernel.metadata["_build"]["mode"] = mode

        for ext_cfg in app.get("extensions", []):
            _load_extension(base, ext_cfg, kernel)

        _kernel = kernel

    except Exception as _e:
        import traceback
        traceback.print_exc()
        _build_error = _e
        _kernel = None

    if _self_module is not None and __name__ not in sys.modules:
        sys.modules[__name__] = _self_module


def _get_context():
    global _context
    if _context is None:
        _ensure_built()
        from _kernel import Context
        _context = Context(kernel=_kernel)
    return _context


# === Public API whitelist ===

# Resolved through kernel.alias (registered by extensions).
_ALIAS_EXPORTS = frozenset({
    # series
    "series",
    # math top-level aliases
    "sin", "cos", "tan", "exp", "log", "sqrt", "neg", "abs", "square",
    # stats
    "total", "minimum", "maximum", "mean", "var", "std", "count",
    # ops
    "scan", "sort", "histogram", "matmul", "fft",
    # tabular
    "groupby",
})

# Resolved from internal modules (extensions that expose no aliases,
# bound here at composition root). Format: public -> (module, attr).
_BINDING_EXPORTS = {
    # creation
    "zeros": ("Creation._lib.api", "zeros"),
    "ones": ("Creation._lib.api", "ones"),
    "full": ("Creation._lib.api", "full"),
    "arange": ("Creation._lib.api", "arange"),
    "linspace": ("Creation._lib.api", "linspace"),
    "index": ("Creation._lib.api", "index"),
    "tile": ("Creation._lib.api", "tile"),
    "repeat": ("Creation._lib.api", "repeat"),
    # series class
    "Series": ("Series._lib.numeric_series", "NumericSeries"),
    # tabular
    "topk": ("TopK._lib.topk", "topk"),
    # diagnostics
    "device_info": ("_core.backend", "device_info"),
}

_RANDOM_EXPORTS = {
    "uniform": ("Creation._lib.api", "random_uniform"),
    "normal": ("Creation._lib.api", "random_normal"),
    "integers": ("Creation._lib.api", "random_integers"),
}

_MATH_EXPORTS = {
    "sin": "sin", "cos": "cos", "tan": "tan", "exp": "exp",
    "log": "log", "sqrt": "sqrt", "neg": "neg", "abs": "abs", "square": "square",
}


class _LazyNamespace:
    """Lazy namespace: attributes resolve to aliases or internal bindings."""

    __slots__ = ("_map",)

    def __init__(self, mapping):
        object.__setattr__(self, "_map", mapping)

    def __getattr__(self, name):
        mapping = object.__getattribute__(self, "_map")
        if name not in mapping:
            raise AttributeError(f"namespace has no attribute '{name}'")
        target = mapping[name]
        if isinstance(target, tuple):
            mod = importlib.import_module(target[0])
            return getattr(mod, target[1])
        _ensure_built()
        return getattr(_get_context(), target)

    def __dir__(self):
        return sorted(object.__getattribute__(self, "_map"))


def _resolve_binding(name):
    mod = importlib.import_module(_BINDING_EXPORTS[name][0])
    return getattr(mod, _BINDING_EXPORTS[name][1])


def _set_backend(name=None):
    """Set active backend: "cpu" | "gpu" | "numpy" | "wgpu" | "auto"/None."""
    ctx = _get_context()
    return ctx.set_driver(name)


class _ProfileSession:
    """Context-manager wrapper around ExecutionProfiler (spec 2.6).

    with nf.profile(): ...   -> profiler bound to the block
    Attributes of the inner ExecutionProfiler are delegated.
    """

    def __init__(self, *args, **kwargs):
        from Runtime._lib.profiler import ExecutionProfiler
        self._profiler = ExecutionProfiler(*args, **kwargs)

    def __enter__(self):
        return self._profiler

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, name):
        return getattr(self._profiler, name)


def _profile(*args, **kwargs):
    """Return an ExecutionProfiler session usable as a context manager.

    Thin binding to Runtime._lib.profiler.ExecutionProfiler.
    """
    return _ProfileSession(*args, **kwargs)


class _InternalShim:
    """nf._internal -- escape hatch for non-public machinery.

    nf._internal.context  -> runtime Context
    nf._internal.kernel   -> built kernel (or None)
    nf._internal.<name>   -> anything exposed on the Context
                             (compile, execute, register_kernel, drivers...)
    """

    def __getattr__(self, name):
        if name == "context":
            return _get_context()
        if name == "kernel":
            _ensure_built()
            return _kernel
        if name.startswith("_"):
            raise AttributeError(name)
        _ensure_built()
        return getattr(_get_context(), name)


def _get_internal():
    global _internal_shim
    if _internal_shim is None:
        _internal_shim = _InternalShim()
    return _internal_shim


__all__ = [
    # creation
    "zeros", "ones", "full", "arange", "linspace", "index", "tile", "repeat",
    # random
    "random",
    # series
    "Series", "series",
    # math namespace + aliases
    "math", "sin", "cos", "tan", "exp", "log", "sqrt", "neg", "abs", "square",
    # stats
    "total", "minimum", "maximum", "mean", "var", "std", "count",
    # ops
    "scan", "sort", "histogram", "matmul", "fft",
    # tabular
    "topk", "groupby",
    # diagnostics
    "device_info", "set_backend", "profile",
]


class _NumFastModule(types.ModuleType):
    """Module subclass with whitelisted lazy attribute resolution."""

    def __getattr__(self, name):
        if name.startswith('_'):
            if name == '_build_error':
                return _build_error
            if name == '_internal':
                return _get_internal()
            raise AttributeError(name)
        if name == 'random':
            ns = _ns_cache.get('random')
            if ns is None:
                ns = _LazyNamespace(_RANDOM_EXPORTS)
                _ns_cache['random'] = ns
            return ns
        if name == 'math':
            ns = _ns_cache.get('math')
            if ns is None:
                ns = _LazyNamespace(_MATH_EXPORTS)
                _ns_cache['math'] = ns
            return ns
        if name == 'set_backend':
            return _set_backend
        if name == 'profile':
            return _profile
        if name in _BINDING_EXPORTS:
            _ensure_built()
            return _resolve_binding(name)
        if name in _ALIAS_EXPORTS:
            _ensure_built()
            return getattr(_get_context(), name)
        raise AttributeError(f"module 'numfast' has no attribute '{name}'")

    def __dir__(self):
        s1 = set(object.__dir__(self))
        s1.update(__all__)
        s1.add('_internal')
        return sorted(s1)


sys.modules[__name__].__class__ = _NumFastModule
