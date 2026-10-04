# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Test gates: маркеры fast/heavy. Default = только fast; heavy — только явно (-m heavy).

Two run modes, one suite
------------------------
This suite was written against a source checkout. Most modules build their
kernel with ``from builder import MAIN; MAIN["build"](APP_DIR)``, where
APP_DIR is the repository root (``Path(__file__).resolve().parents[2]``) and
``builder`` lives in a separate repository. Against an *installed* numfast
neither holds: ``builder`` is absent, and APP_DIR names no engine. That is
why the artefact this repository ships had never been tested -- the suite
could not be pointed at it.

When ``numfast`` resolves outside this checkout's ``src/``, conftest:

* registers a ``builder`` stand-in whose ``MAIN["build"]`` returns the
  installed package's own kernel. An installed package resolves its fork root
  from ``numfast/full.toml`` beside ``numfast/_ext/``, so the application
  directory is not an input it can accept -- hence the installed kernel is
  preferred over the path. The stand-in rebuilds per call rather than sharing
  the singleton: two modules install a pyarrow blocker and then assert the
  Arrow-less build refuses, and a shared kernel would quietly satisfy them
  with the Arrow-enabled one;
* leaves the modules listed in ``SOURCE_ONLY`` uncollected, and says why at the
  end of the run. Those read engine source text or import engine internals as
  top-level packages, neither of which exists in an installed wheel -- which
  does ship that source, at ``numfast/_ext/``. Repointing them is a per-module
  change; until then they are named rather than left to fail obscurely.

In a checkout none of this engages: ``builder`` stays the real Builder, every
module is collected, and the run is the run that was measured before.
"""

import sys
from pathlib import Path

import pytest

# Test tooling: собственный _builder пакета; dev-запуск из корня репо через src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_CHECKOUT_SRC = (Path(__file__).resolve().parents[1] / "src").resolve()

#: Modules that need the source checkout, and the one reason each cannot run
#: against an installed package. Keyed by test module file name; empty of
#: effect in a checkout.
SOURCE_ONLY = {
    "test_calibration_guards.py": (
        "reads the Planner's routing source text (calibrate.py / planner.py)"),
    "test_calibration_profile_memo.py": (
        "imports the engine as a top-level package (`from _lib.calibrate`) "
        "and reads calibration.toml through the checkout root"),
    "test_consumer_is_null.py": (
        "reads Extension source text from src/Semantic/TableExpr/_lib/"),
    "test_consumer_silent_regressions.py": (
        "reads Extension source text from src/Semantic/TableExpr/_lib/"),
    "test_consumer_surface.py": (
        "reads Extension source text from src/Semantic/TableExpr/_lib/"),
    "test_import_guard.py": (
        "asserts the Builder rejects source-tree cross-Extension import paths"),
    "test_ops_dict_ucs4.py": (
        "imports engine internals as top-level packages (`Storage.Dictionary`)"),
    "test_ops_filter.py": (
        "loads and compares against src/Drivers/CPU/_lib/native_cpu.py as a "
        "reference implementation"),
    "test_ops_gpu_filter.py": "reads src/Drivers/GPU/_lib/gpu.py source text",
    "test_ops_gpu_lookup.py": (
        "imports engine internals as top-level packages (`Drivers`) and reads "
        "src/Drivers/GPU/_lib/gpu.py source text"),
    "test_ops_gpu_resident_accum.py": (
        "opens src/Drivers/GPU/_lib/gpu.py at import time"),
    "test_ops_gpu_scan.py": (
        "imports engine internals as top-level packages (`Drivers`) and reads "
        "src/Drivers/GPU/_lib/gpu.py source text"),
    "test_ops_lookup.py": (
        "puts src/ on sys.path and imports `Drivers` / `Relational` / `_lib` "
        "as top-level packages"),
    "test_ops_lookup_api.py": (
        "loads src/Relational/Join/_lib/join.py as a reference implementation"),
    "test_ops_pairinsert_bitmasksweep.py": (
        "loads src/Relational/{PairInsert,BitmaskSweep}/_lib/*.py as reference "
        "implementations"),
    "test_ops_panic_safety.py": (
        "reads src/Drivers/CPU/_lib/native_cpu.py source text and imports "
        "`Drivers.CPU._lib` as a top-level package"),
    "test_ops_segmented.py": (
        "puts src/ on sys.path and imports `Relational.Segmented._lib` as a "
        "top-level package"),
    "test_ops_text_affix.py": (
        "imports the native CPU module by its bare name (`native_cpu`), which "
        "only resolves under the source-tree layout"),
    "test_planner_auto_v1.py": (
        "puts src/Runtime/Planner on sys.path and imports `_lib.calibrate` as "
        "a top-level package"),
    "test_release_blockers.py": (
        "asserts on engine source text (text guards over the Extension sources)"),
}


def _numfast_is_installed():
    """True when `numfast` resolves outside this checkout's src/."""
    try:
        import numfast
    except Exception:
        return False
    pkg = Path(numfast.__file__).resolve().parent
    return not pkg.is_relative_to(_CHECKOUT_SRC)


INSTALLED = _numfast_is_installed()

#: Filled by pytest_ignore_collect so the end-of-run summary can name what was
#: left out and why.
_not_collected = []


def _install_builder_stand_in():
    """`from builder import MAIN` resolves to the installed package's kernel.

    The 99 `MAIN["build"](APP_DIR)` call sites are deliberately left alone: they
    are correct against a checkout and they are the coupling being removed. Here
    they resolve to a stand-in that takes the kernel from the installed package
    instead of from an application directory that does not exist.

    Rebuilt per call, not shared: `MAIN["build"]` was a fresh build per call,
    and test_domain_lut_noarrow.py / test_ops_noarrow_fallback.py install a
    pyarrow import blocker and then assert that the Arrow-less build refuses.
    A cached kernel would hand them the Arrow-enabled one and the assertion
    would pass without testing anything.
    """
    import types

    import numfast

    def _build(*_args, **_kwargs):
        return numfast.get_kernel(fresh=True)

    mod = types.ModuleType("builder")
    mod.__doc__ = (
        "conftest stand-in for the app-builder package, active only when "
        "numfast resolves outside a source checkout. MAIN['build'] returns a "
        "freshly built kernel of the installed package; the application "
        "directory is ignored because an installed package resolves its own "
        "fork root.")
    mod.MAIN = {"build": _build}
    mod.__numfast_conftest_stand_in__ = True
    sys.modules["builder"] = mod


if INSTALLED:
    _install_builder_stand_in()


def pytest_configure(config):
    config.addinivalue_line("markers", "fast: seconds-scale tests, default selection")
    config.addinivalue_line("markers", "heavy: memory/load tests, run only with -m heavy")
    config.addinivalue_line(
        "markers", "source_tree: needs the source checkout, not an installed package")


@pytest.fixture(scope="session")
def installed_kernel():
    """The kernel of the installed numfast package.

    Outside a checkout this is the only kernel the suite may use: it is the one
    an installed package builds from its own numfast/full.toml. Inside a
    checkout the fixture skips, so a dev run cannot silently pass against a
    stale installed wheel instead of the tree it is meant to be testing.
    """
    if not INSTALLED:
        pytest.skip("installed_kernel: numfast resolves inside this checkout")
    import numfast
    return numfast.get_kernel()


def pytest_ignore_collect(collection_path, config):
    """Leave the SOURCE_ONLY modules out of an installed-package run.

    They fail at import time (top-level engine imports) or read engine source
    under the checkout root, so they cannot produce a result here. They are
    not silently dropped: pytest_terminal_summary names each one.
    """
    if not INSTALLED:
        return None
    name = Path(str(collection_path)).name
    if name in SOURCE_ONLY:
        _not_collected.append(name)
        return True
    return None


def pytest_collection_modifyitems(config, items):
    if config.option.markexpr.strip():
        return
    skip_heavy = pytest.mark.skip(reason="heavy: run explicitly with pytest -m heavy")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _not_collected:
        return
    w = terminalreporter.write_line
    w("")
    w("source_tree only -- %d module(s) not collected against an installed "
      "package:" % len(_not_collected))
    for name in sorted(_not_collected):
        w("  %-40s %s" % (name, SOURCE_ONLY[name]))
