# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import importlib
import os

_DRIVERS: dict[str, object] = {}
_ACTIVE: object = None
_DEFAULT = os.environ.get("NUMFAST_BACKEND", None)


_PRIORITY = ["cupy", "wgpu", "numpy"]


def _detect():
    for name in _PRIORITY:
        try:
            mod = importlib.import_module(f"._{name}", __package__)
            if mod.AVAILABLE:
                _DRIVERS[name] = mod
        except Exception:
            pass


def set_active(name: str | None = None):
    if not _DRIVERS:
        _detect()
    if name is None:
        name = _DEFAULT if _DEFAULT in _DRIVERS else next(p for p in _PRIORITY if p in _DRIVERS)
    if name not in _DRIVERS:
        raise ImportError(f"Backend '{name}' not available on this platform")
    global _ACTIVE
    _ACTIVE = _DRIVERS[name]


def get_active_name() -> str | None:
    if _ACTIVE is None:
        set_active()
    for name, mod in _DRIVERS.items():
        if mod is _ACTIVE:
            return name
    return None


def get_xp():
    if _ACTIVE is None:
        set_active()
    return _ACTIVE.get_xp()


def get_device():
    if _ACTIVE is None:
        set_active()
    return _ACTIVE.get_device()


def device_info() -> dict:
    if _ACTIVE is None:
        set_active()
    return _ACTIVE.device_info()
