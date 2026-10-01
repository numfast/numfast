# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Kernel: flat alias namespace shared by all Extensions."""


class Kernel:
    """alias -> function; owner sys-aliases _<Owner>_<alias>; variables autoinvoke."""

    def __init__(self, name="App", singleton=True):
        self.alias: dict = {}
        self._owner: dict = {}
        self.metadata: dict = {"_kernel": {"name": name, "singleton": singleton}}
        self.variables: set = set()

    def register(self, alias_name, func, owner=""):
        self.alias[alias_name] = func
        if owner:
            self._owner[alias_name] = owner
            sys_alias = f"_{owner}_{alias_name}"
            self.alias[sys_alias] = func
            self._owner[sys_alias] = owner

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        fn = self.alias.get(name)
        if fn is None:
            raise AttributeError(f"alias '{name}' not found")
        if name in self.variables:
            return fn()
        return fn

    def __repr__(self):
        fn = self.alias.get("_default")
        if fn:
            try:
                return str(fn())
            except Exception:
                pass
        return f"Kernel({self.metadata['_kernel']['name']})"
