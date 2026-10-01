# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Native DLL env boot (packaged-first resolution).

Moved verbatim from numfast.__init__._ensure_native_env: packaged
_native/ first, gnu-target fallback. No behavior change.
"""


def ensure_native_env(pkg_dir):
    import os
    if os.environ.get("NUMFAST_NATIVE_DISABLE") == "1":
        return
    if "NUMFAST_NATIVE_DLL" in os.environ:
        return
    for cand in (pkg_dir / "_native" / "numfast_native.dll",
                 pkg_dir / "_native" / "numfast_native.so"):
        if cand.exists():
            os.environ["NUMFAST_NATIVE_DLL"] = str(cand)
            return
    gnu = pkg_dir.parents[1] / "numfast-native" / "target" / "x86_64-pc-windows-gnu" / "release" / "numfast_native.dll"
    if gnu.exists():
        os.environ["NUMFAST_NATIVE_DLL"] = str(gnu)
        return
