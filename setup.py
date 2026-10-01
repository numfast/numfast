# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Packaging hook: vendor engine Extension sources + full.toml into the wheel.

Repo stays clean (no duplicated engine sources committed): at wheel build
time each [[extensions]] dir from full.toml is copied verbatim into
build_lib/numfast/_ext/<DirName>/, plus full.toml itself. At runtime the
packaged builder resolves extensions package-relatively
(numfast/_ext/...), never via hardcoded dev paths.
Engine sources are NEVER modified here — copy only.
"""

from pathlib import Path
import shutil
import tomllib

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).resolve().parent


class build_py(_build_py):
    def run(self):
        super().run()
        full = ROOT / "full.toml"
        if not full.exists():
            return
        data = tomllib.loads(full.read_text(encoding="utf-8"))
        dest_root = Path(self.build_lib) / "numfast" / "_ext"
        dest_root.mkdir(parents=True, exist_ok=True)
        for ext in data.get("extensions", []):
            rel = ext.get("path", "")
            if not rel:
                continue
            src = (ROOT / rel).resolve()
            if not src.is_dir():
                continue
            # Loader addresses extensions by directory name ({Name}.toml
            # lives next to {Name}.py); keep the leaf name verbatim.
            dest = dest_root / src.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(
                src, dest,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        shutil.copyfile(full, Path(self.build_lib) / "numfast" / "full.toml")


setup(cmdclass={"build_py": build_py})
