# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Packaged NumFast builder: Kernel + manifests + guarded loader + assembly.

Fresh minimal implementation for the wheel boundary (no import from any
dev-tree builder, no hardcoded paths). Same contracts:
flat alias namespace with _<Owner>_<alias> sys aliases, Kahn topo order,
AST import-guard (cross-Extension private imports rejected loud),
exec() with isolated sys.path (own _lib visible), setup(kernel) metadata.
"""

from ._kernel import Kernel
from ._manifest import load_app_manifest, load_extension_manifest
from ._order import resolve_order
from ._loader import load_extension
from ._api import build

__all__ = [
    "Kernel",
    "load_app_manifest",
    "load_extension_manifest",
    "resolve_order",
    "load_extension",
    "build",
]
