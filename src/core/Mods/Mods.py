# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        from Mods._lib.mods_lib import _Registry
        _registry = _Registry()
    return _registry


def register(name: str | None = None):
    if callable(name):
        _get_registry().add(name.__name__, name)
        return name
    def decorator(func):
        key = name or func.__name__
        _get_registry().add(key, func)
        return func
    return decorator


def get_registered() -> dict:
    return dict(_get_registry().items())
