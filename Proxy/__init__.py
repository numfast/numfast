"""Proxy — lazy module loader for NumFast.

Usage:
    from Proxy import load, info

    mod = load("module.name")
    status = info()
"""

from .Proxy import load, info

__all__ = ["load", "info"]
