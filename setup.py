# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Packaging hook for the numfast wheel / sdist. Copy-only, no code is modified.

Three jobs:

1. Vendor the engine Extensions named by ``full.toml`` into
   ``build_lib/numfast/_ext/<DirName>/`` plus ``full.toml`` itself, so the
   packaged builder resolves extensions package-relatively and the repo tree
   stays free of duplicated sources. A ``[[extensions]]`` entry whose directory
   is missing is a hard error: skipping it silently is how the 0.2.1 wheel came
   to ship 18 of 29 Extensions.

2. Vendor the Rust Corresponding Source into
   ``build_lib/numfast/_corresp_src/numfast-native/`` and write SHA256SUMS over
   exactly the bytes that shipped. The wheel conveys ``numfast_native.dll`` --
   object code -- so AGPL-3.0 section 6 requires the source that builds it to
   travel with it. See CORRESPONDING-SOURCE.md.

3. Pick the wheel flavour from what ``src/numfast/_native/`` actually holds. A
   binary present gives a platform-tagged wheel carrying it; no binary gives
   ``py3-none-any`` with no native path at all. A wheel that carries a Windows
   ``.dll`` and claims ``py3-none-any`` is a lie that installs broken on Linux
   and macOS, so the two facts are derived from one place and cannot disagree.

``NUMFAST_WHEEL_NATIVE=0`` forces the pure-Python flavour, ``=1`` requires a
binary to be present. Unset means: follow the contents of ``_native/``.
"""

import hashlib
import os
import shutil
import sysconfig
import tomllib
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

ROOT = Path(__file__).resolve().parent

#: Generated packaging metadata setuptools rewrites on every command. It is
#: listed in .gitignore and carries nothing the build needs.
EGG_INFO = "src/numfast.egg-info"

NATIVE_DIR = ROOT / "src" / "numfast" / "_native"
CRATE = ROOT / "numfast-native"

#: Data files the engine reads from the fork root at run time.
RUNTIME_DATA = ("calibration.toml", "calibration_dataset.json")

#: Everything `cargo build` needs to reproduce the shipped binary, and nothing
#: else. Each entry is (source path relative to numfast-native/, dest suffix).
CORRESPONDING_SOURCE = (
    ("Cargo.toml", "Cargo.toml"),
    ("Cargo.lock", "Cargo.lock"),
    # Cited by src/lib.rs, which ships in every wheel.
    ("REUSE.md", "REUSE.md"),
    (".cargo/config.toml", ".cargo/config.toml"),
    ("tools/nf-link.bat", "tools/nf-link.bat"),
    ("tools/nf-link.py", "tools/nf-link.py"),
)

BINARY_SUFFIXES = (".dll", ".so", ".dylib")


def _native_binaries():
    if not NATIVE_DIR.is_dir():
        return []
    return sorted(p for p in NATIVE_DIR.iterdir()
                  if p.is_file() and p.suffix in BINARY_SUFFIXES)


def _host_tag():
    tag = sysconfig.get_platform().lower()
    for ch in "-. ":
        tag = tag.replace(ch, "_")
    return tag


def _suffix_for_tag(tag):
    """The binary suffix that a wheel tagged `tag` is allowed to carry."""
    if tag.startswith("win"):
        return ".dll"
    if tag.startswith("darwin") or tag.startswith("macos"):
        return ".dylib"
    if tag.startswith("linux"):
        return ".so"
    return None


def resolve_flavour():
    """-> (include_native, plat_name|None, binaries). Raises on inconsistency.

    The tag is derived from the build host, and the host's binary suffix is
    cross-checked against every binary about to be packed. A ``.dll`` found
    while building for Linux is a stale or foreign artefact; shipping it under
    this host's tag is the exact failure this check exists to stop.
    """
    binaries = _native_binaries()
    forced = os.environ.get("NUMFAST_WHEEL_NATIVE")
    if forced not in (None, "", "0", "1"):
        raise SystemExit(
            "NUMFAST_WHEEL_NATIVE must be 0 or 1, got %r" % (forced,))
    want = (forced == "1") if forced else bool(binaries)

    if not want:
        # Pure flavour: assert nothing native leaks in, whatever is on disk.
        return False, None, binaries

    if not binaries:
        raise SystemExit(
            "NUMFAST_WHEEL_NATIVE=1 but %s holds no %s file. Build the crate "
            "first, or set NUMFAST_WHEEL_NATIVE=0 for the pure-Python wheel."
            % (NATIVE_DIR, "/".join(BINARY_SUFFIXES)))

    tag = _host_tag()
    expected = _suffix_for_tag(tag)
    if expected is None:
        raise SystemExit(
            "cannot derive a native binary suffix for host platform %r; "
            "refusing to guess a wheel tag." % (tag,))
    for b in binaries:
        if b.suffix != expected:
            raise SystemExit(
                "%s is a %s file but the build host is %s. Shipping it would "
                "produce a %s wheel whose native path cannot load. Remove it, "
                "or build the wheel on a matching host."
                % (b.name, b.suffix, tag, tag))
    return True, tag, binaries


INCLUDE_NATIVE, PLAT_NAME, BINARIES = resolve_flavour()

# bdist_wheel derives py3-none-<plat> from root_is_pure + plat_name. The wheel
# IS pure Python -- the .dll is loaded through ctypes and is not a CPython
# extension module -- so Root-Is-Purelib must stay true and py3 must stay the
# implementation tag. Only the platform component changes. Setting
# has_ext_modules() instead would produce cp3xx-cp3xx-win_amd64 and pin the
# package to one interpreter.
OPTIONS = {"bdist_wheel": {"plat_name": PLAT_NAME}} if PLAT_NAME else {}


class sdist(_sdist):
    """sdist that carries no generated egg-info.

    ``setuptools.command.sdist.run`` does, in this order:

        self.filelist = ei_cmd.filelist
        self.filelist.append(os.path.join(ei_cmd.egg_info, 'SOURCES.txt'))

    The append happens *after* ``egg_info`` has applied MANIFEST.in, so
    ``prune src/numfast.egg-info`` in MANIFEST.in can only stop the directory
    from being grafted -- it cannot stop this one explicit append, and
    SOURCES.txt lands in the archive regardless. Nothing in the build reads
    it: it is setuptools' own index of the very tree the sdist already carries.
    So it is dropped here, where the append actually happens.

    Everything else is stock. ``make_release_tree`` additionally writes a
    generated ``setup.cfg`` holding ``[egg_info] tag_build / tag_date``; that
    is setuptools pinning the version so a wheel built from this sdist gets the
    same number, and it is left in place on purpose.
    """

    def make_distribution(self):
        parts = EGG_INFO.replace("/", os.sep).split(os.sep)

        def inside(path):
            norm = os.path.normpath(path).split(os.sep)
            return norm[:len(parts)] == parts

        before = len(self.filelist.files)
        self.filelist.files = [f for f in self.filelist.files if not inside(f)]
        dropped = before - len(self.filelist.files)
        if dropped:
            print("sdist: dropped %d generated egg-info entries" % dropped)
        super().make_distribution()


class build_py(_build_py):
    def run(self):
        super().run()
        self._vendor_extensions()
        self._vendor_corresponding_source()
        self._vendor_runtime_data()
        self._vendor_native_binary()

    def _dest(self, *parts):
        p = Path(self.build_lib).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- engine Extensions ---------------------------------------------------

    def _vendor_extensions(self):
        full = ROOT / "full.toml"
        if not full.exists():
            raise SystemExit(
                "%s is missing: the wheel would ship no engine at all."
                % (full,))
        data = tomllib.loads(full.read_text(encoding="utf-8"))
        dest_root = Path(self.build_lib) / "numfast" / "_ext"
        dest_root.mkdir(parents=True, exist_ok=True)
        for ext in data.get("extensions", []):
            name = ext.get("name", "")
            rel = ext.get("path", "")
            if not rel:
                raise SystemExit(
                    "full.toml: [[extensions]] entry %r has no path" % (name,))
            src = (ROOT / rel).resolve()
            if not src.is_dir():
                raise SystemExit(
                    "full.toml names Extension %r at %s, which does not exist. "
                    "Refusing to ship a wheel with a silently missing "
                    "Extension." % (name, rel))
            if not (src / (src.name + ".toml")).exists():
                raise SystemExit(
                    "Extension %r at %s has no %s.toml; the loader addresses "
                    "Extensions by directory name." % (name, rel, src.name))
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

    # -- AGPL-3.0 section 6: Corresponding Source of the conveyed binary ------

    def _vendor_corresponding_source(self):
        base = Path(self.build_lib) / "numfast" / "_corresp_src"
        dest = base / "numfast-native"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)

        shutil.copytree(
            CRATE / "src", dest / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        missing = [rel for rel, _ in CORRESPONDING_SOURCE
                   if not (CRATE / rel).exists()]
        if missing:
            raise SystemExit(
                "Corresponding Source incomplete, missing from %s: %s. The "
                "wheel conveys numfast_native.dll; without these files AGPL-3.0 "
                "section 6 is not satisfied."
                % (CRATE, ", ".join(missing)))
        for rel, dst_rel in CORRESPONDING_SOURCE:
            target = dest / dst_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(CRATE / rel, target)

        shutil.copyfile(ROOT / "CORRESPONDING-SOURCE.md",
                        base / "CORRESPONDING-SOURCE.md")

        # Written from the bytes that shipped, so the claim is checkable:
        # `sha256sum -c SHA256SUMS` in the unpacked wheel must pass.
        digests = []
        for path in sorted(p for p in dest.rglob("*")
                           if p.is_file() and p.name != "SHA256SUMS"):
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            digests.append("%s  %s" % (h, path.relative_to(dest).as_posix()))
        (dest / "SHA256SUMS").write_text(
            "\n".join(digests) + "\n", encoding="utf-8")

    # -- runtime data -------------------------------------------------------

    def _vendor_runtime_data(self):
        for name in RUNTIME_DATA:
            src = ROOT / name
            if not src.exists():
                raise SystemExit(
                    "%s is missing: the Planner reads it from the fork root at "
                    "run time and degrades to default cost routing without it."
                    % (src,))
            shutil.copyfile(src, self._dest("numfast", name))

    # -- the native binary, only in the flavour that declares it -------------

    def _vendor_native_binary(self):
        if not INCLUDE_NATIVE:
            return          # pure flavour: no _native/ at all, and nothing to
            #               fall back to except the NumPy path
        for b in BINARIES:
            shutil.copyfile(b, self._dest("numfast", "_native", b.name))


setup(cmdclass={"build_py": build_py, "sdist": sdist}, options=OPTIONS)
