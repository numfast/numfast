# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Import-guard probe: cross-Extension private imports must FAIL loud.

Proves the new Builder AST guard runs per Extension (entry + _lib):
depends/alias/setup is the only legal reuse, never Python imports.
"""

import pytest


def _make_ext(parent, name, toml_body, py_body, lib_files=None):
    ext_dir = parent / name
    (ext_dir / "_lib").mkdir(parents=True, exist_ok=True)
    (ext_dir / f"{name}.toml").write_text(toml_body, encoding="utf-8")
    (ext_dir / f"{name}.py").write_text(py_body, encoding="utf-8")
    for rel, body in (lib_files or {}).items():
        full = ext_dir / "_lib" / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(body, encoding="utf-8")
    return ext_dir


@pytest.mark.fast
def test_cross_extension_import_rejected(tmp_path):
    from builder import MAIN

    load_extension = MAIN["load_extension"]
    Kernel = MAIN["Kernel"]
    bad = _make_ext(
        tmp_path, "BadProbe",
        'name = "BadProbe"\nalias = ["bad"]\nmods = ["bad_fn"]\n',
        "from CPU._lib import cpu_execute_impl\n\n"
        "def bad_fn():\n    return 'bad'\n\nPUBLIC = {'bad_fn': bad_fn}\n",
    )
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        load_extension(Kernel(name="Probe"), str(bad), known_extensions={"BadProbe", "CPU"})


@pytest.mark.fast
def test_relative_escape_rejected(tmp_path):
    from builder import MAIN

    load_extension = MAIN["load_extension"]
    Kernel = MAIN["Kernel"]
    bad = _make_ext(
        tmp_path, "EscProbe",
        'name = "EscProbe"\nalias = ["bad"]\nmods = ["bad_fn"]\n',
        "def bad_fn():\n    return 'bad'\n\nPUBLIC = {'bad_fn': bad_fn}\n",
        lib_files={"evil.py": "from .. import something\n"},
    )
    with pytest.raises(RuntimeError, match="escapes"):
        load_extension(Kernel(name="Probe"), str(bad), known_extensions={"EscProbe", "CPU"})
