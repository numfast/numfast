# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Loader: AST import-guard + exec() with isolated sys.path + alias registration.

Reuse between Extensions is ONLY via depends + kernel.alias at runtime;
absolute Python imports of another Extension's privates are rejected loud.
Own _lib / relative-inside-own-folder / stdlib / third-party are allowed.
"""

import ast
import importlib.util
import sys
from pathlib import Path

from ._kernel import Kernel
from ._manifest import load_extension_manifest


def _cross_msg(ext_name, target, stmt):
    return (
        f"ImportError: entry in extension '{ext_name}' has forbidden '{stmt}'. "
        f"Private import from another extension '{target}'. "
        "Cross-Extension Python imports are FORBIDDEN. "
        f"Correct reuse via Builder: depends = [\"{target}\"] in {ext_name}.toml "
        "plus call at runtime via kernel alias (kernel.alias[\"<alias>\"]), "
        f"never 'from {target}... import'. "
        "Allowed: own _lib (from _lib... / from . ...), stdlib, third-party (numpy)."
    )


def _is_installed(top):
    try:
        return importlib.util.find_spec(top) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _check_top(label, ext_name, own_names, others, *, top, dotted, stmt):
    if not top or top in own_names or top == "_lib":
        return
    parts = dotted.split(".")
    if top in others or (len(parts) > 1 and parts[1] == "_lib"
                         and not _is_installed(top)):
        raise RuntimeError(_cross_msg(ext_name, top, stmt))


def _check_source(source, *, ext_name, own_names, known_extensions, label):
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise RuntimeError(f"SyntaxError: {label}: {e}") from None
    others = set(known_extensions or ()) - set(own_names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                _check_top(label, ext_name, own_names, others,
                           top=(a.name or "").split(".")[0],
                           dotted=a.name or "", stmt=f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                if node.level > 1:
                    raise RuntimeError(
                        f"ImportError: {label} in extension '{ext_name}' escapes "
                        f"its folder ('from ..', level {node.level}). "
                        "Relative imports must stay inside own Extension. "
                        "Cross-Extension reuse — only via depends + kernel alias."
                    )
                continue
            mod = node.module or ""
            _check_top(label, ext_name, own_names, others,
                       top=mod.split(".")[0], dotted=mod,
                       stmt=f"from {mod} import ...")


def load_extension(kernel, ext_dir, override_metadata=None, known_extensions=None):
    ext_dir = Path(ext_dir).resolve()
    ext_name = ext_dir.name
    manifest_path = ext_dir / f"{ext_name}.toml"
    em = load_extension_manifest(str(manifest_path))
    meta_section = em["name"]
    if meta_section not in kernel.metadata:
        kernel.metadata[meta_section] = {}
    kernel.metadata[meta_section].update(em["metadata"])
    if override_metadata:
        kernel.metadata[meta_section].update(override_metadata)
    if "_versions" not in kernel.metadata:
        kernel.metadata["_versions"] = {}
    kernel.metadata["_versions"][em["name"]] = em["version"]
    if not em["mods"]:
        return
    py_path = ext_dir / f"{ext_name}.py"
    if not py_path.exists():
        raise FileNotFoundError(f"Extension module not found: {py_path}")
    source = py_path.read_text(encoding="utf-8")
    own = {ext_name, em["name"]}
    known = set(known_extensions or ()) | own
    _check_source(source, ext_name=em["name"], own_names=own,
                  known_extensions=known, label=py_path.name)
    lib_dir = ext_dir / "_lib"
    if lib_dir.is_dir():
        for f in sorted(lib_dir.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            _check_source(f.read_text(encoding="utf-8"), ext_name=em["name"],
                          own_names=own, known_extensions=known,
                          label=f"{ext_name}/_lib/{f.relative_to(lib_dir).as_posix()}")
    # Pre-purge _lib*: a stale `import _lib.x` from another context (dev-tree
    # test shims import Join's regular-package _lib directly) would otherwise
    # shadow this Extension's own _lib by name. Post-purge keeps it clean.
    for mod in [m for m in sys.modules if m == "_lib" or m.startswith("_lib.")]:
        del sys.modules[mod]
    added = str(ext_dir)
    # Hide competing regular `_lib` owners left on sys.path by foreign
    # contexts (a permanent ext-dir entry would win over this Extension's
    # own _lib by path order). Restored in finally. Namespace-only _lib
    # dirs merge harmlessly and are left alone.
    hidden = [p for p in sys.path
              if p != added and (Path(p) / "_lib" / "__init__.py").exists()]
    for p in hidden:
        sys.path.remove(p)
    sys.path.insert(0, added)
    try:
        ns: dict = {}
        exec(compile(source, str(py_path), "exec"), ns)  # noqa: S102 - builder assembly by design
    finally:
        try:
            sys.path.remove(added)
        except ValueError:
            pass
        for p in hidden:
            if p not in sys.path:
                sys.path.append(p)
        for mod in [m for m in sys.modules if m == "_lib" or m.startswith("_lib.")]:
            del sys.modules[mod]
    ext_public = ns.get("PUBLIC", {})
    if not ext_public:
        raise RuntimeError(f"{py_path.name} has no PUBLIC dict")
    for alias_name, mod_name in zip(em["alias"], em["mods"]):
        func = ext_public.get(mod_name)
        if func is None:
            continue
        kernel.register(alias_name, func, owner=em["name"])
        if alias_name in em["variables"]:
            kernel.variables.add(alias_name)
    setup_fn = ns.get("setup")
    if callable(setup_fn):
        setup_fn(kernel)


__all__ = ["Kernel", "load_extension"]
