# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Standalone native discovery: a module loaded BY PATH must find its own binary.

`test_native_env_boot.py` covers discovery when the `numfast` package boots and
`ensure_native_env` pins NUMFAST_NATIVE_DLL. That boot path is the easy one, and
it hides the defect this file exists for: eleven ctypes modules each carried their
own `_DLL_DEFAULT` naming

    <root>/numfast-native/target/x86_64-pc-windows-gnu/release/numfast_native.dll

as a module-level literal. Booted, nobody reads it -- the env var wins. Imported
on its own, which is what the STANDALONE contract in each of those modules says
is supported, it was the ONLY candidate each had, and on Linux it was a Windows
path. Measured on Ubuntu 24.04 before the fix:

    native_cpu._DLL_DEFAULT -> .../target/x86_64-pc-windows-gnu/release/numfast_native.dll
                               (exists=False: the only default, unreachable)
    Join/native._DLL_DEFAULT -> src/numfast/_native/numfast_native.dll
                               (exists=True: a WINDOWS DLL loaded ON LINUX)

The second line is the worse one. A default that points at nothing degrades to
numpy and is merely useless; a default that resolves hands the platform a binary
built for a different one.

Each module is loaded by FILE PATH here, exactly as a research script does, and
never through the booted package, because importing them normally is what makes
this test vacuous.
"""

import importlib.util
import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")

#: Every module that carries its own native default. `native_i64` predates the
#: `_DLL_DEFAULT` spelling and names the resolved path `_GNU_DEFAULT`.
MODULES = [
    ("Drivers/CPU/_lib/native_cpu.py", "_DLL_DEFAULT"),
    ("Drivers/CPU/_lib/rowwise_kway.py", "_DLL_DEFAULT"),
    ("Drivers/CPU/_lib/rowwise_min4.py", "_DLL_DEFAULT"),
    ("Drivers/CPU/_lib/rowwise_min4_time.py", "_DLL_DEFAULT"),
    ("Drivers/GroupedHashMT/_lib/grouped_native_hash.py", "_DLL_DEFAULT"),
    ("Relational/Segmented/_lib/cost.py", "_DLL_DEFAULT"),
    ("Relational/PairInsert/_lib/pair_insert.py", "_DLL_DEFAULT"),
    ("Relational/Router/_lib/router.py", "_DLL_DEFAULT"),
    ("Relational/Sssp/_lib/sssp.py", "_DLL_DEFAULT"),
    ("Relational/Join/_lib/native.py", "_DLL_DEFAULT"),
    ("Relational/Join/_lib/native_i64.py", "_GNU_DEFAULT"),
]


def _load_by_path(rel, mod_name):
    """Import one file standalone, the way a research script does.

    `src` goes on sys.path because the module's resolver imports
    `numfast._lib.native_env` -- but nothing here imports `numfast` itself, and
    the assertion below is that the RESULT is right without a boot having
    pinned NUMFAST_NATIVE_DLL.
    """
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    path = os.path.join(SRC, rel.replace("/", os.sep))
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("rel,attr", MODULES,
                         ids=[r.split("/")[-1] for r, _ in MODULES])
def test_standalone_default_is_this_platforms_binary(rel, attr):
    """No module may name a foreign-platform artefact as its default."""
    mod = _load_by_path(rel, "nf_std_" + rel.replace("/", "_")[:-3])
    default = getattr(mod, attr, None)
    assert default, "%s has no %s" % (rel, attr)
    if sys.platform.startswith("win"):
        assert default.endswith(".dll"), default
    else:
        # The whole defect: on a non-Windows host a .dll was the only candidate.
        assert not default.endswith(".dll"), default
        assert os.path.basename(default).startswith(
            ("numfast_native.so", "libnumfast_native.so")), default


@pytest.mark.parametrize("rel,attr", MODULES,
                         ids=[r.split("/")[-1] for r, _ in MODULES])
def test_standalone_default_never_names_a_dead_build_tree_slot(rel, attr):
    """The resolved default must be a file this host can actually open.

    Deliberately NOT a source-text assertion for `x86_64-pc-windows-gnu`: the
    triple still appears in these files, in two places that are both correct --
    the explanatory comments, and `_LEGACY_DLL_DEFAULT`, the last-resort literal
    kept for a research script loaded by path with `numfast` not importable at
    all. Grepping for the string would fail on the honest documentation and pass
    on a real regression, so it is not the assertion.

    What must hold is about the PATH: if the default points into a cargo build
    tree, that directory has to exist. `target/x86_64-pc-windows-gnu/release/`
    does not exist on a Linux host, so a default landing there is the defect.
    """
    mod = _load_by_path(rel, "nf_std_dead_" + rel.replace("/", "_")[:-3])
    default = getattr(mod, attr, None)
    assert default, "%s has no %s" % (rel, attr)
    parts = os.path.normpath(default).split(os.sep)
    if "target" in parts:
        idx = parts.index("target")
        slot = os.sep.join(parts[:idx + 3])       # target/<triple>/release
        assert os.path.isdir(slot), (
            "%s resolves to a build-tree slot that does not exist: %s" % (rel, slot))


def test_every_named_module_still_imports_standalone():
    """All eleven, in one process, as the STANDALONE contract promises."""
    for i, (rel, attr) in enumerate(MODULES):
        mod = _load_by_path(rel, "nf_std_all_%d" % i)
        assert getattr(mod, attr, None), rel