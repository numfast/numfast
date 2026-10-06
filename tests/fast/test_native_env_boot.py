# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Native discovery at boot: which file is offered, and what is said when none is.

`ensure_native_env()` decides what `NUMFAST_NATIVE_DLL` is pinned to, and
`native_info()` reports the same decision. Three properties are held here, each
against a synthetic tree so the assertions do not depend on which build tree
this machine happens to carry:

* the binary NAME is derived from the running platform, so discovery never
  offers a file the host cannot load -- on Linux never the Windows `.dll` a
  checkout keeps tracked in `src/numfast/_native/`;
* the binary may live in the package (an installed wheel) or in the cargo
  build tree (a checkout), in that order, and `NUMFAST_NATIVE_DLL` overrides
  both;
* when nothing exists, every probed path is NAMED -- returned, and warned with a
  path list -- instead of a silent `None`.

The real end-to-end resolution of a real tree is `test_packaging_adapters.py::
test_native_dll_package_relative`; this file is the unit-level contract behind
it.
"""

import os
import sys
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = str(REPO / "src")
sys.meta_path = [f for f in sys.meta_path
                 if "__editable___numfast_" not in type(f).__module__]
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from numfast._lib import native_env  # noqa: E402

NATIVE = "numfast_native"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A checkout-shaped tree: <root>/full.toml + <root>/src/numfast/_native/.

    `full.toml` is what `fork_root` walks to, so its presence is what makes the
    build-tree half of `probed_paths` reachable at all. Both native env vars
    are cleared: a suite run that inherited either would silently change what
    these tests measure.
    """
    (tmp_path / "full.toml").write_text("", encoding="utf-8")
    pkg = tmp_path / "src" / "numfast"
    (pkg / "_native").mkdir(parents=True)
    monkeypatch.delenv("NUMFAST_NATIVE_DISABLE", raising=False)
    monkeypatch.delenv("NUMFAST_NATIVE_DLL", raising=False)
    return tmp_path, pkg


class _no_warning:
    """Assert the body warns about NOTHING -- the silent-success half of the
    contract, which a test that only checks failures never reaches."""

    def __enter__(self):
        self._cm = warnings.catch_warnings(record=True)
        self._log = self._cm.__enter__()
        warnings.simplefilter("always")
        return self._log

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        assert not self._log, [str(w.message) for w in self._log]
        return False


def _stage(root, rel):
    """Create a stub binary at `root/rel`; content is irrelevant, existence is
    the only thing discovery reads."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x7fELF-stub")
    return p


# -- the name is the platform's own ----------------------------------------

@pytest.mark.parametrize("platform,name", [
    ("win32", NATIVE + ".dll"),
    ("linux", NATIVE + ".so"),
    ("darwin", NATIVE + ".dylib"),
    ("freebsd13", None),
])
def test_binary_name_derived_from_platform(monkeypatch, platform, name):
    """One name, derived. A platform with no binary gets None, not a guess."""
    monkeypatch.setattr(sys, "platform", platform)
    assert native_env.native_binary_name() == name


def test_linux_never_offers_the_tracked_windows_dll(tree, monkeypatch):
    """B(i): a checkout's `_native/` holds a Windows .dll; Linux must not take it.

    This is the exact failure 01eb849 fixed: a candidate LIST let each platform
    take the first name it recognised, so on Linux the tracked
    `numfast_native.dll` was offered and loaded and every native lane then
    failed silently into the fallback. The assertion is on the WHOLE probe, not
    on the winner: the .dll must not be a candidate at all, so a legal `.so`
    later in the list cannot be reached past it either.
    """
    root, pkg = tree
    _stage(root, "src/numfast/_native/" + NATIVE + ".dll")
    monkeypatch.setattr(sys, "platform", "linux")
    probed = native_env.probed_paths(pkg)
    assert probed, "a known platform must offer at least one candidate"
    assert not [p for p in probed if p.suffix != ".so"], probed
    with pytest.warns(RuntimeWarning):
        native_env.ensure_native_env(pkg)
    assert "NUMFAST_NATIVE_DLL" not in os.environ


def test_linux_pins_the_build_tree_not_the_windows_dll(tree, monkeypatch):
    """Both artefacts present: the .so wins because the .dll is not offered."""
    root, pkg = tree
    _stage(root, "src/numfast/_native/" + NATIVE + ".dll")
    built = _stage(root, "numfast-native/target/release/lib" + NATIVE + ".so")
    monkeypatch.setattr(sys, "platform", "linux")
    with _no_warning():
        native_env.ensure_native_env(pkg)
    assert os.environ["NUMFAST_NATIVE_DLL"] == str(built)


# -- where the file may live ------------------------------------------------

def test_packaged_path_wins_over_the_build_tree(tree, monkeypatch):
    """The installed-wheel shape: the binary is inside the package."""
    root, pkg = tree
    monkeypatch.setattr(sys, "platform", "linux")
    staged = _stage(root, "src/numfast/_native/" + NATIVE + ".so")
    _stage(root, "numfast-native/target/release/lib" + NATIVE + ".so")
    with _no_warning():
        native_env.ensure_native_env(pkg)
    assert os.environ["NUMFAST_NATIVE_DLL"] == str(staged)


def test_build_tree_found_when_nothing_is_staged(tree, monkeypatch):
    """A clean checkout: `_native/` is a manual `cp` that need not have run.

    This is the whole point of the second probe. Without it a Linux checkout has
    to have NUMFAST_NATIVE_DLL set by hand to reach a native run at all, because
    nothing downstream can derive the platform's own filename from the build
    tree.
    """
    root, pkg = tree
    monkeypatch.setattr(sys, "platform", "linux")
    built = _stage(root, "numfast-native/target/release/lib" + NATIVE + ".so")
    probed = native_env.ensure_native_env(pkg)
    assert os.environ["NUMFAST_NATIVE_DLL"] == str(built)
    assert probed[-1] == built, probed


@pytest.mark.parametrize("triple", [
    "x86_64-unknown-linux-gnu", "x86_64-pc-windows-gnu", "wasm32-unknown-unknown",
])
def test_every_target_triple_release_dir_is_probed(tree, monkeypatch, triple):
    """`cargo build --release` and `cargo build --release --target <t>` write to
    different places and nothing records which one ran. The triple is read off
    the directory, not hardcoded, so a new host is picked up without an edit."""
    root, pkg = tree
    monkeypatch.setattr(sys, "platform", "linux")
    built = _stage(root, "numfast-native/target/%s/release/lib%s.so"
                   % (triple, NATIVE))
    probed = native_env.ensure_native_env(pkg)
    assert os.environ["NUMFAST_NATIVE_DLL"] == str(built)
    assert built in probed, probed


def test_bare_and_lib_prefixed_names_are_both_probed(tree, monkeypatch):
    """Cargo calls a Unix cdylib `lib<stem>.so`; setup.py checks only the
    SUFFIX (`resolve_flavour` -> `_suffix_for_tag`), so a wheel may legitimately
    carry either spelling and only probing both makes the packaged head of the
    list match what the wheel actually staged."""
    root, pkg = tree
    monkeypatch.setattr(sys, "platform", "linux")
    staged = _stage(root, "src/numfast/_native/lib" + NATIVE + ".so")
    with _no_warning():
        native_env.ensure_native_env(pkg)
    assert os.environ["NUMFAST_NATIVE_DLL"] == str(staged)


def test_env_var_overrides_discovery(tree, monkeypatch):
    """An operator naming a file is an override, not a hint: discovery does not
    run, so it cannot take the packaged file instead. The returned list is the
    operator's file alone -- listing the defaults next to it would point at the
    wrong thing."""
    root, pkg = tree
    monkeypatch.setattr(sys, "platform", "linux")
    _stage(root, "src/numfast/_native/" + NATIVE + ".so")
    named = pkg.parent.parent / "elsewhere" / "custom.so"
    _stage(pkg.parent.parent, "elsewhere/custom.so")
    monkeypatch.setenv("NUMFAST_NATIVE_DLL", str(named))
    with _no_warning():
        considered = native_env.ensure_native_env(pkg)
    assert os.environ["NUMFAST_NATIVE_DLL"] == str(named)
    assert considered == [str(named)], considered


def test_disable_considers_nothing_and_does_not_warn(tree, monkeypatch):
    """`NUMFAST_NATIVE_DISABLE=1` is a request for the fallback, not a failure
    to report, so it must not warn and must not go looking."""
    root, pkg = tree
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("NUMFAST_NATIVE_DISABLE", "1")
    with _no_warning():
        assert native_env.ensure_native_env(pkg) == []
    assert "NUMFAST_NATIVE_DLL" not in os.environ


# -- degrading visibly ------------------------------------------------------

def test_nothing_found_warns_with_every_probed_path(tree, monkeypatch):
    """The list is the deliverable: "not found" alone does not say WHERE."""
    root, pkg = tree
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.warns(RuntimeWarning) as rec:
        probed = native_env.ensure_native_env(pkg)
    message = str(rec[0].message)
    assert probed, "an empty probe list cannot be reported"
    for p in probed:
        assert str(p) in message, f"{p} not named in: {message}"
    assert NATIVE + ".so" in message
    assert "NUMFAST_NATIVE_DLL" in message


def test_nothing_found_still_returns_the_list(tree, monkeypatch):
    """No silent None: the return value is the same list the warning names."""
    root, pkg = tree
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.warns(RuntimeWarning):
        probed = native_env.ensure_native_env(pkg)
    assert probed == native_env.probed_paths(pkg)
    assert not [p for p in probed if p.exists()], probed
    assert "NUMFAST_NATIVE_DLL" not in os.environ


def test_found_does_not_warn(tree, monkeypatch):
    root, pkg = tree
    monkeypatch.setattr(sys, "platform", "linux")
    _stage(root, "src/numfast/_native/" + NATIVE + ".so")
    with _no_warning():
        native_env.ensure_native_env(pkg)


# -- the reporting probe stays package-relative -----------------------------

def test_package_only_probe_is_what_reporting_offers(tree, monkeypatch):
    """`default_native_path` is the package-relative offer a caller that only
    reports facts wants. It must never name the build tree, and it must agree
    with the packaged HEAD of `probed_paths` -- those are two different readers
    of one decision and a disagreement between them is the bug 01eb849 fixed."""
    root, pkg = tree
    monkeypatch.setattr(sys, "platform", "linux")
    assert native_env.default_native_path(pkg) is None
    staged = _stage(root, "src/numfast/_native/" + NATIVE + ".so")
    assert native_env.default_native_path(pkg) == str(staged)
    assert native_env.probed_paths(pkg)[0] == staged
    _stage(root, "numfast-native/target/release/lib" + NATIVE + ".so")
    assert native_env.default_native_path(pkg) == str(staged)


def test_unknown_platform_offers_nothing(tree, monkeypatch):
    root, pkg = tree
    monkeypatch.setattr(sys, "platform", "freebsd13")
    _stage(root, "src/numfast/_native/" + NATIVE + ".so")
    assert native_env.native_binary_name() is None
    assert native_env.probed_paths(pkg) == []
    assert native_env.default_native_path(pkg) is None
    with pytest.warns(RuntimeWarning):
        assert native_env.ensure_native_env(pkg) == []
    assert "NUMFAST_NATIVE_DLL" not in os.environ


def test_fork_root_is_a_strict_ancestor(tree):
    """A wheel's marker is at `numfast/full.toml` -- the package directory
    ITSELF, not an ancestor -- so a wheel has no fork root and the build-tree
    probe is inert there. Asserted so a future marker check cannot quietly move
    the build tree one level too high in a checkout."""
    root, pkg = tree
    assert native_env.fork_root(pkg) == root.resolve()
    wheel = pkg / "_ext" / "Driver" / "nowhere"
    wheel.mkdir(parents=True)
    (pkg / "full.toml").write_text("", encoding="utf-8")
    assert native_env.fork_root(wheel) == pkg.resolve()