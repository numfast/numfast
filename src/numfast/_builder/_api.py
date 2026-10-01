# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Assembly: full.toml -> ordered exec() of Extensions -> Kernel.

Extension dirs resolve package-relatively: <base>/<path> first (repo
checkout: base=repo root, paths like src/Core), then the packaged
fallback <pkg>/../_ext/<DirName> (wheel: base=<pkg>, vendored copies).
No hardcoded filesystem roots anywhere in this path.
"""

from pathlib import Path

from ._kernel import Kernel
from ._manifest import load_app_manifest, load_extension_manifest
from ._order import resolve_order
from ._loader import load_extension


def _resolve_ext_dir(base: Path, rel: str):
    direct = (base / rel).resolve()
    if direct.is_dir() and (direct / f"{direct.name}.toml").exists():
        return direct
    # Packaged fallback: wheel vendors sources under numfast/_ext/<Name>.
    pkg = Path(__file__).resolve().parent.parent
    cand = (pkg / "_ext" / Path(rel).name).resolve()
    if cand.is_dir():
        return cand
    return direct


def build(app_dir):
    base = Path(app_dir).resolve()
    manifest_path = base / "full.toml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"full.toml not found in {base}")
    app = load_app_manifest(str(manifest_path))
    kernel = Kernel(name=app["kernel"]["name"],
                    singleton=app["kernel"]["singleton"])
    ext_refs = []
    for ext_cfg in app["extensions"]:
        name = ext_cfg.get("name", "")
        rel = ext_cfg.get("path", "")
        if not rel:
            continue
        ext_dir = _resolve_ext_dir(base, rel)
        depends = []
        manifest = ext_dir / f"{ext_dir.name}.toml"
        if manifest.exists():
            depends = load_extension_manifest(str(manifest)).get("depends", [])
        ext_refs.append({"name": name, "path": str(ext_dir),
                         "depends": depends,
                         "metadata": ext_cfg.get("metadata", {})})
    ordered = resolve_order(ext_refs)
    known = set()
    for ref in ext_refs:
        known.add(ref["name"])
        known.add(Path(ref["path"]).name)
    for ref in ordered:
        override = {**ref.get("metadata", {}),
                    **app.get("extensions_meta", {}).get(Path(ref["path"]).name, {})}
        load_extension(kernel, ref["path"],
                       override_metadata=override or None,
                       known_extensions=known)
    if app.get("tests"):
        kernel.metadata["_tests"] = app["tests"]
    return kernel


__all__ = ["build"]
