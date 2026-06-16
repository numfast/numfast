# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from Proxy._lib.proxy_lib import _LazyLoader

_loader = _LazyLoader()

def load(module_name: str):
    return _loader.load(module_name)

def info() -> dict:
    return {
        "loaded_modules": list(_loader._cache.keys()),
    }
