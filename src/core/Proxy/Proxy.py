# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

_loader = None


def _get_loader():
    global _loader
    if _loader is None:
        from Proxy._lib.proxy_lib import _LazyLoader
        _loader = _LazyLoader()
    return _loader


def load(module_name: str):
    return _get_loader().load(module_name)


def info() -> dict:
    return {
        "loaded_modules": list(_get_loader()._cache.keys()),
    }
