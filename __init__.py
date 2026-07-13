"""NumFast -- GPU-first numerical computing framework.

Usage:
    import numfast as nf

    # Statistics
    print(nf.total([1, 2, 3]))

    # Math ops (lazy expression nodes)
    y = nf.sin(nf.series([0, 1.57]))

    # Compute ops
    result = nf.scan(data)

    # Pipeline
    jobs = [{"op": "SMA", "inputs": ["Close"], "params": {"period": 20}}]
    tasks = nf.compile(jobs)
    nf.execute(tasks, source_data)
"""

import sys
import pathlib
import types

# === Path setup ===
_base_dir = pathlib.Path(__file__).resolve().parent

# Add src/ and group directories to sys.path
_src = _base_dir / 'src'
for _sub in ['', 'core', 'math']:
    _p = str(_src / _sub) if _sub else str(_src)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Add APP Builder to path
_ab_candidates = [
    _base_dir.parent / 'app-builder',
    _base_dir / 'app-builder',
]
_ab_path = None
for _c in _ab_candidates:
    if _c.is_dir() and (_c / 'builder').is_dir():
        _ab_path = str(_c.resolve())
        break
if _ab_path and _ab_path not in sys.path:
    sys.path.insert(0, _ab_path)

# === Lazy kernel build — deferred until first attribute access ===
_kernel = None
_kernel_built = False
_build_error = None
_context = None


# (flat extension list removed -- all extensions have __init__.py)


def _ensure_built():
    global _kernel, _kernel_built, _build_error
    if _kernel_built:
        return
    _kernel_built = True

    _self_module = sys.modules.get(__name__)

    # Use APP Builder Kernel class
    from builder import MAIN
    Kernel = MAIN["Kernel"]

    import tomllib
    import importlib.util
    from pathlib import Path

    base = _base_dir.resolve()
    manifest_path = base / "full.toml"

    try:
        app = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        _kernel = Kernel(
            name=app["kernel"]["name"],
            singleton=app["kernel"].get("singleton", True),
        )

        for ext_cfg in app.get("extensions", []):
            ext_path = (base / ext_cfg["path"]).resolve()
            ext_name = ext_path.name

            # Read extension manifest
            ext_manifest_path = ext_path / f"{ext_name}.toml"
            if not ext_manifest_path.exists():
                continue
            em = tomllib.loads(ext_manifest_path.read_text(encoding="utf-8"))

            # Import extension module via importlib
            py_path = ext_path / f"{ext_name}.py"
            if not py_path.exists():
                continue

            spec = importlib.util.spec_from_file_location(ext_name, str(py_path))
            mod = importlib.util.module_from_spec(spec)
            # Add ext_path so module-level relative imports (from _lib.xxx) resolve
            sys.path.insert(0, str(ext_path))
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.path.pop(0)

            # Clear _lib from sys.modules to prevent cross-extension collision
            for _key in list(sys.modules):
                if _key == '_lib' or _key.startswith('_lib.'):
                    del sys.modules[_key]

            # Register aliases by matching mod names to module attributes
            for alias_name, mod_name in zip(em.get("alias", []), em.get("mods", [])):
                func = getattr(mod, mod_name, None)
                if func is not None:
                    _kernel.register(alias_name, func, owner=ext_name)

            # Call setup(kernel) if module has it
            if hasattr(mod, "setup"):
                mod.setup(_kernel)

            # Merge toml metadata into kernel
            if "metadata" in em:
                _kernel.metadata.setdefault(ext_name, {})
                _kernel.metadata[ext_name].update(em["metadata"])

        # Note: _lib modules are cleaned inline above (after each extension load).
        # Flat extension modules (Stats, Series, Mods) have __init__.py --
        # not namespace packages, so no cleanup needed.

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


class _NumFastModule(types.ModuleType):
    """Module subclass that delegates attribute access to Context."""

    def __getattr__(self, name):
        if name == '__version__':
            return "0.1.0"
        if name == '_build_error':
            return _build_error
        if _kernel_built:
            try:
                return getattr(_get_context(), name)
            except AttributeError:
                raise AttributeError(f"module 'numfast' has no attribute '{name}'")
        # Before build: only known Context names trigger build.
        # Pytest probes many module attrs during setup; block all others.
        if name.startswith('_') or name.startswith('pytest'):
            raise AttributeError(name)
        # Conservatively block common pytest introspection attrs.
        if name in ('setUpModule', 'setup_module', 'tearDownModule', 'teardown_module',
                     '__builtins__', '__cached__', '__doc__', '__file__', '__loader__',
                     '__name__', '__package__', '__path__', '__spec__'):
            raise AttributeError(name)
        return getattr(_get_context(), name)

    def __dir__(self):
        s1 = set(object.__dir__(self))
        if _kernel_built:
            ctx = _get_context()
            if ctx is not None:
                s2 = set(dir(ctx))
            else:
                s2 = set()
        else:
            s2 = set()
        return sorted(s1 | s2)


sys.modules[__name__].__class__ = _NumFastModule
