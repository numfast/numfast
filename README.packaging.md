# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# numfast distribution — what ships, and what a user gets

`pip install numfast` gives a self-contained `import numfast`:

- Engine Extension sources are vendored into the wheel (`numfast/_ext/*`)
  verbatim at build time by `setup.py`; the repo tree stays clean. A
  `[[extensions]]` entry in `full.toml` whose directory is missing is a **hard
  build error** — silently skipping it is how the old `0.2.1` wheel shipped a
  partial Extension set.
- The wheel builder (`src/numfast/_builder/`) is a fresh minimal
  implementation: no import from the dev-tree app-builder,
  no hardcoded filesystem roots.
- Optional deps: `pip install numfast[pandas]`, `numfast[arrow]`.
  Without pyarrow the CPU path keeps working (NumPy fallback).

## Three artefacts, one version

The Rust binary is built per platform, so the release is **three distribution
artefacts at the same version** — two wheels and the sdist. All three must
exist before anything is published: PyPI rejects a filename that already
exists, so a partial upload leaves users on a subset.

| Artefact | Tag | Native payload | Who gets it |
|---|---|---|---|
| `numfast-0.2.1-py3-none-win_amd64.whl` | `py3-none-win_amd64` | `numfast_native.dll` | Windows x86-64: full engine, native path live |
| `numfast-0.2.1-py3-none-manylinux_2_28_x86_64.whl` | `py3-none-manylinux_2_28_x86_64` | `libnumfast_native.so` | manylinux_2_28 x86-64: full engine, native path live |
| `numfast-0.2.1.tar.gz` | sdist | source only | anyone building from source |

`Root-Is-Purelib` is `true` in both wheels. Nothing here is a CPython
extension module — the native library is loaded through `ctypes` — so the
implementation tag stays `py3` and neither wheel is pinned to one interpreter.

There is no `py3-none-any` wheel and no macOS artefact. The manylinux wheel is
not a renamed linux-tagged one: it is produced by
`tools/build_manylinux_wheel.sh` inside
`quay.io/pypa/manylinux_2_28_x86_64`, which does a real `auditwheel repair`.
Its portability comes from the build host's glibc, so the build must happen in
that image — see `docs/INSTALL.md` for the recipe and the measured
`auditwheel show` output.

The library is also buildable straight from the Corresponding Source the wheel
ships, with no Docker at all: `numfast-native/.cargo/config.toml` is scoped to
`[target.x86_64-pc-windows-gnu]`, so one `cargo build --release` needs no `zig`
and no other host-specific tool.

The native path can be forced off explicitly with `NUMFAST_NATIVE_DISABLE=1`, and
overridden with `NUMFAST_NATIVE_DLL` when a locally built binary exists.

## Building

```bash
python -m build                          # sdist, then the wheel from the sdist
NUMFAST_WHEEL_NATIVE=0 python -m build   # a wheel with no native payload
```

`python -m build` builds the wheel **from the sdist**, so an incomplete
`MANIFEST.in` fails the build instead of producing a quietly broken artefact.

The manylinux wheel is built in the container, not on the build host:

```bash
docker run -d --name nfbuild quay.io/pypa/manylinux_2_28_x86_64 sleep infinity
docker cp . nfbuild:/src
docker exec -w /src nfbuild bash tools/build_manylinux_wheel.sh
```

The flavour follows the contents of `src/numfast/_native/`: a binary present
gives the platform-tagged wheel, nothing present gives a wheel with no native
payload. `NUMFAST_WHEEL_NATIVE=0` / `=1` override it. The tag comes from the
build host and is cross-checked against the binary's suffix, so a `.dll` cannot
be packed into a wheel tagged for another platform.

`SHA256SUMS` over the Corresponding Source is generated **during the build**
and is not a stored constant: no `SHA256SUMS` file is tracked in the repository.
Verify it in the build that produced the artefact.

## Licence and Corresponding Source

AGPL-3.0-only. `LICENSE` and `NOTICE` reach `dist-info/licenses/`; `METADATA`
carries `License-Expression: AGPL-3.0-only`.

Both wheels convey a native binary — object code — so AGPL-3.0 section 6
requires the Corresponding Source to travel with it. It does:
`numfast/_corresp_src/numfast-native/` carries the crate's 47 `.rs` files,
`Cargo.toml`, `Cargo.lock`, the project-local `.cargo/config.toml`, the linker
shim it names, and a build-time `SHA256SUMS`. The sdist carries the same source
at its root. **The written-offer route is deliberately not used** — reasoning in
`CORRESPONDING-SOURCE.md`.

Version: one number for the whole project, owned by `pyproject.toml`
`[project].version`. `numfast-native/Cargo.toml` and
`numfast-native/ts/package.json` mirror it, and the JS build refuses to run when
`package.json` disagrees.

## Boundary checks after install

```bash
python -I -c "import numfast as nf; s = nf.from_numpy([1,2,3]); print(nf.to_numpy(s * 2))"
python -I -c "import numfast as nf; print(nf.native_info())"
python -I -c "import numfast as nf; print(nf.from_numpy([1.5,2.5]).reduce('sum'))"
NUMFAST_NATIVE_DISABLE=1 python -I -c "import numfast as nf; print(nf.to_numpy(nf.from_numpy([1,2,3]) * 2))"
```
