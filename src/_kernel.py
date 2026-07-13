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
