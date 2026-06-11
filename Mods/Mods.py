from Mods._lib.mods_lib import _Registry

_registry = _Registry()

def register(name: str | None = None):
    def decorator(func):
        key = name or func.__name__
        _registry.add(key, func)
        return func
    return decorator

def get_registered() -> dict:
    return dict(_registry.items())
