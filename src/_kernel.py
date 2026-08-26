"""NumFast kernel -- thin proxy around Builder Kernel.

ALL user-facing functions are mods registered by extensions.
This Context just delegates to kernel.alias.

Usage:
    import numfast as nf
    nf.total([1,2,3])
    nf.sin(nf.series([0, 1.57]))
"""


class Context:
    """Thin proxy around Builder Kernel.

    All functionality comes from kernel.alias (mods registered by extensions).
    No hardcoded imports. If an alias doesn't exist, it's an error.
    """

    def __init__(self, kernel=None):
        self._kernel = kernel

    def set_driver(self, name=None):
        """Set active backend driver.

        Controls which backend executes compute operations.

        Args:
            name: backend name:
                - "cpu"  — force NumPy CPU backend
                - "gpu"  — force WebGPU backend (raises if unavailable)
                - "auto" or None — auto-detect best available backend
                - "numpy", "wgpu" — explicit backend names

        Raises:
            ImportError: if requested backend is not available
        """
        from _core.backend import set_active
        if name is None or name == "auto":
            set_active(None)
        elif name == "cpu":
            set_active("numpy")
        elif name == "gpu":
            set_active("wgpu")
        else:
            set_active(name)

    def get_active_driver(self) -> str:
        """Return name of active driver: 'cpu' or 'gpu'."""
        from _core.backend import get_active_name
        name = get_active_name()
        if name in ("wgpu", "cupy"):
            return "gpu"
        return "cpu"

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        k = object.__getattribute__(self, '_kernel')
        if k is not None:
            try:
                return k.alias[name]
            except KeyError:
                pass
            try:
                return getattr(k, name)
            except AttributeError:
                pass
        raise AttributeError(f"Context has no attribute '{name}'")

    def __dir__(self):
        k = object.__getattribute__(self, '_kernel')
        base = set(object.__dir__(self))
        if k is not None and hasattr(k, 'alias'):
            base.update(k.alias.keys())
        return sorted(base)
