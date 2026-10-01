# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Manifests: full.toml (app) + {Name}.toml (extension). stdlib tomllib only."""

import tomllib
from pathlib import Path


def load_app_manifest(path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"full.toml not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    extends = data.get("extends")
    base = None
    if extends:
        base_path = path.parent / extends
        if not base_path.exists():
            raise FileNotFoundError(f"Base manifest not found: {base_path}")
        base = load_app_manifest(base_path)
    k_data = data.get("kernel", {})
    base_kernel = base["kernel"] if base else {}
    raw_exts = data.get("extensions", [])
    if base and not raw_exts:
        extensions = list(base["extensions"])
    elif base:
        exclude = {e["name"] for e in raw_exts if e.get("exclude")}
        ext_map = {e["name"]: dict(e) for e in base["extensions"]
                   if e["name"] not in exclude}
        for e in raw_exts:
            if e.get("exclude"):
                continue
            ext_map[e["name"]] = dict(e)
        extensions = list(ext_map.values())
    else:
        extensions = [dict(e) for e in raw_exts]
    child_meta = data.get("extensions_meta", {})
    base_meta = base.get("extensions_meta", {}) if base else {}
    return {
        "kernel": {
            "name": k_data.get("name", base_kernel.get("name", "App")),
            "singleton": k_data.get("singleton", base_kernel.get("singleton", True)),
        },
        "extensions": extensions,
        "extensions_meta": {**base_meta, **child_meta},
        "tests": data.get("tests", {}),
    }


def load_extension_manifest(path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Extension manifest not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        "name": data.get("name", path.parent.name),
        "version": data.get("version", "0.1.0"),
        "alias": data.get("alias", []),
        "mods": data.get("mods", []),
        "depends": data.get("depends", []),
        "variables": data.get("variables", []),
        "metadata": data.get("metadata", {}),
    }
