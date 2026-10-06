#!/bin/bash
# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#
# Build the PORTABLE Linux wheel: py3-none-manylinux_2_28_x86_64.
#
# WHY THIS EXISTS. A wheel tagged `py3-none-linux_x86_64` is the least portable
# tag in the format: it means "built on this distro, no guarantee", so ordinary
# Linux users are never offered it. `pip install numfast` on Ubuntu must be
# served by a manylinux-tagged wheel, and only a real `auditwheel repair` may
# put that tag on the file. Renaming the linux_x86_64 artefact is not an
# alternative to the audit; it is the audit's result forged by hand, and it is
# a false portability claim shipped to users. Nothing here relabels anything.
#
# The build runs INSIDE the manylinux image, not on a normal Linux box, for two
# reasons that are not interchangeable:
#   * glibc. The image is AlmaLinux 8.10 / glibc 2.28. Build on a newer distro
#     (Ubuntu 24.04 = 2.39) and the .so picks up symbols the image's libc does
#     not have; auditwheel then refuses to lower the tag. What makes the wheel
#     portable is the BUILD HOST, so the build host is the image.
#   * the toolchain. numfast-native is a Rust crate, so rustup installs
#     1.98.1 inside the container. The image ships no rustc.
#
# Usage (from the repository root, Docker + a rust-capable host):
#     docker run -d --name nfbuild quay.io/pypa/manylinux_2_28_x86_64 sleep infinity
#     docker cp <this repo> nfbuild:/src
#     docker exec -w /src nfbuild bash tools/build_manylinux_wheel.sh
#     docker cp nfbuild:/wheelhouse/ ./wheelhouse/
#
# The archive, not the working tree, is what gets built. A tracked
# src/numfast/_native/numfast_native.dll rides along in every `git archive HEAD`
# and it is a WINDOWS artefact: setup.py:resolve_flavour hard-fails a .dll
# packed under a linux tag, which is the correct outcome and the reason this
# script stages the .so explicitly instead of relying on what was in the tree.

set -euo pipefail

SRC="${SRC:-/src}"
OUT="${OUT:-$SRC/wheelhouse}"
PYBIN=/opt/python/cp312-cp312/bin        # the image's interpreter; `pip` is not on PATH
export PATH="$PYBIN:/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"

STAGED_SO="src/numfast/_native/libnumfast_native.so"

echo "=== 0. toolchain ==="
if ! command -v rustc >/dev/null; then
    curl -sSf https://sh.rustup.rs -o /tmp/rustup.sh
    sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.98.1 \
        --target x86_64-unknown-linux-gnu >/tmp/rustup.log 2>&1
fi
pip install -q "setuptools>=77" build wheel twine 2>/dev/null || \
    pip install -q "setuptools>=77" build wheel twine
python3 -V; rustc -V; python3 -c "import setuptools;print('setuptools',setuptools.__version__)"

echo "=== 1. crate -> a Linux .so ==="
( cd "$SRC/numfast-native" && cargo build --release --target x86_64-unknown-linux-gnu )
SO="$SRC/numfast-native/target/x86_64-unknown-linux-gnu/release/libnumfast_native.so"
test -f "$SO"

echo "=== 2. stage it where setup.py packs from ==="
mkdir -p "$SRC/src/numfast/_native"
# The tracked Windows .dll is removed, never renamed. Leaving it in place makes
# resolve_flavour raise, which is the build refusing to ship a Windows binary
# under a linux tag -- the behaviour this whole script depends on.
rm -f "$SRC/src/numfast/_native"/*.dll
cp "$SO" "$SRC/$STAGED_SO"

echo "=== 3. refuse to continue if a foreign binary is still staged ==="
ls -1 "$SRC/src/numfast/_native/"

echo "=== 4. the wheel, platform-tagged from the build host ==="
rm -rf "$SRC/build" "$SRC/dist" "$SRC/src/numfast.egg-info"
( cd "$SRC" && NUMFAST_WHEEL_NATIVE=1 python3 -m build --wheel --no-isolation )
ls -1 "$SRC/dist/"

echo "=== 5. the audit that earns the tag ==="
mkdir -p "$OUT"
auditwheel repair "$SRC"/dist/numfast-*.whl -w "$OUT"

echo "=== 6. the verdict, quoted (not inferred from the filename) ==="
auditwheel show "$OUT"/manylinux*.whl | head -8

echo
echo "Portable wheel: $OUT/numfast-*-manylinux_2_28_x86_64.whl"
echo "Note: auditwheel may widen the tag (e.g. manylinux_2_31) if the toolchain"
echo "emits newer symbols. The tag in step 6 is the one that is true."