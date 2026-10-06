# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Native DLL env boot (packaged-first resolution).

Boot resolution and the boot-time REPORTING of the same file are one decision
here, so they cannot disagree: `ensure_native_env` pins NUMFAST_NATIVE_DLL and
`default_native_path` is the same probe for callers that only want to report.

The binary NAME is derived from the running platform, never a list of candidate
names. A list makes every platform take the first name it recognises, and in a
checkout `src/numfast/_native/` holds the tracked Windows `numfast_native.dll`,
so Linux was offered -- and loaded -- a Windows artefact (measured on Ubuntu
24.04, `sys.platform == "linux"`, NUMFAST_NATIVE_DLL unset:
`native_info()["dll"] -> src/numfast/_native/numfast_native.dll`,
`dll_exists: True`). The same list would have offered that file on macOS too.

The suffix mapping mirrors `setup.py:_suffix_for_tag`, which hard-fails the
build when a `.dll` is packed for a Linux host; that guard is build-time only and
never covered this run-time path. setup.py is unchanged.
"""


def native_binary_name(stem="numfast_native"):
    """The native binary FILENAME this platform may load, or None.

    None means this platform ships no binary with the project, and the caller
    then offers nothing at all rather than naming a file it cannot load.
    """
    import sys
    if sys.platform.startswith("win"):
        return stem + ".dll"
    if sys.platform == "darwin":
        return stem + ".dylib"
    if sys.platform.startswith("linux"):
        return stem + ".so"
    return None


def default_native_path(pkg_dir):
    """-> the platform's own binary inside the package, or None.

    What discovery OFFERS when NUMFAST_NATIVE_DLL names nothing. Package-relative
    only: the checkout build tree is `ensure_native_env`'s business, and a caller
    that only reports facts does not want a build tree.
    """
    name = native_binary_name()
    if name is None:
        return None
    cand = pkg_dir / "_native" / name
    return str(cand) if cand.exists() else None


def ensure_native_env(pkg_dir):
    import os
    if os.environ.get("NUMFAST_NATIVE_DISABLE") == "1":
        return
    if "NUMFAST_NATIVE_DLL" in os.environ:
        return
    name = native_binary_name()
    if name is None:
        return
    cand = pkg_dir / "_native" / name
    if cand.exists():
        os.environ["NUMFAST_NATIVE_DLL"] = str(cand)
        return
    gnu = pkg_dir.parents[1] / "numfast-native" / "target" / "x86_64-pc-windows-gnu" / "release" / name
    if gnu.exists():
        os.environ["NUMFAST_NATIVE_DLL"] = str(gnu)
        return
