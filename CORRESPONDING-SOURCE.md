# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# Corresponding Source shipped with the numfast distribution

AGPL-3.0-only section 6 requires that a conveyed *object code* travel with the
**Corresponding Source** of the work, or that the source be offered under a
valid written offer. This distribution conveys object code: the compiled Rust
crate `numfast-native` is loaded at run time through `ctypes` from
`numfast/_native/numfast_native.dll` (or `.so` / `.dylib`).

**NumFast does not use the written-offer route.** The source is shipped. A
written offer is legal, but for a package distributed through an index it is
the worse option: the offer has to stay valid for as long as any copy of the
object code exists, and nothing in a wheel or an sdist can enforce that. Here
the Corresponding Source is ~300 KB of Rust and a manifest, so it goes in the
artefact and the obligation is discharged by construction rather than by a
promise.

## Where this source is in each artefact

| Artefact | Location of the Corresponding Source |
|---|---|
| sdist `numfast-<v>.tar.gz` | `numfast-native/` at the archive root |
| wheel `numfast-<v>-py3-none-<plat>.whl` | `numfast/_corresp_src/numfast-native/` |
| wheel `numfast-<v>-py3-none-any.whl` | `numfast/_corresp_src/numfast-native/` -- present even though that wheel conveys no object code |

This document ships as `CORRESPONDING-SOURCE.md` at the sdist root and as
`numfast/_corresp_src/CORRESPONDING-SOURCE.md` in the wheel.

`LICENSE` (the verbatim AGPL-3.0 text) and `NOTICE` reach
`numfast-<v>.dist-info/licenses/` in both sdist and wheel, and
`METADATA` carries `License-Expression: AGPL-3.0-only`. They are declared in
`pyproject.toml` under `[project] license-files` (PEP 639), so their arrival is
a property of the metadata, not of a copy step someone can forget.

## What is here, and why each item is here

Everything under this directory is exactly what `cargo build` needs to turn
this tree back into the shipped binary. Nothing else was copied in.

| Path | Why it is required to rebuild the binary |
|---|---|
| `Cargo.toml` | crate manifest: `[lib] crate-type`, and the normative `[profile.release]` (`opt-level`, `lto`, `panic = "abort"`) that fixes the code the binary contains |
| `Cargo.lock` | the resolved dependency graph. Here it holds exactly one package, `numfast-native`, i.e. the crate has no third-party dependencies |
| `src/**/*.rs` | the crate source |
| `.cargo/config.toml` | project-local cargo config. On this project it selects the Windows link step; see "What is not here" |
| `tools/nf-link.bat`, `tools/nf-link.py` | the linker shim that `.cargo/config.toml` points at |

`numfast-native/SHA256SUMS` is written **at build time** from the bytes that
went into the wheel. It is the evidence that the shipped source is the source
in the repository, and it is regenerable: `sha256sum -c SHA256SUMS` must pass
after unpacking.

The crate uses no `include!`, `include_str!` or `env!` (verified: zero
matches under `numfast-native/src`), so there is no generated or embedded
build input that has to travel besides the sources above.

## Rebuilding

```bash
# from the unpacked sdist
cd numfast-native
cargo build --release --target x86_64-pc-windows-gnu
# -> target/x86_64-pc-windows-gnu/release/numfast_native.dll

# from an unpacked wheel
cd numfast/_corresp_src/numfast-native
cargo build --release
```

`cargo build --release` with no `--target` reproduces the native binary for the
host you build on. The `wasm32-unknown-unknown` build of the same crate is the
artefact published separately as `@numfast/kernels`; it is not built from the
Python distribution and is therefore not part of *its* Corresponding Source.

## What is not here, and why

- **`numfast-native/target/`, `ts/node_modules/`, `ts/dist/`, `vectors/`,
  `results/`** — build outputs and generated test vectors, reproducible from
  the sources above (`cargo build`, `npm ci`, `tools/gen.py`, seed 42). They
  are not source and are not in version control either.
- **`numfast-native/tools/`** other than the linker shim — benchmarks,
  parity harnesses and JS oracle drivers. They do not contribute a single byte
  to the binary. The ones that gate the *correctness* of the crate are part of
  this repository and travel in the sdist under `numfast-native/tools/`.
- **The absolute linker path inside `.cargo/config.toml`.** It names a linker
  on the machine that produced the release, and that path does not exist on
  yours. This is build-environment configuration, not project configuration:
  it selects a linker, it does not change the program. The flags that *do*
  change the program are in `[profile.release]` in `Cargo.toml` and ship
  verbatim. Repoint the linker line, or delete the file to use your default
  toolchain.
- **The measured `calibration.toml` / `calibration_dataset.json`** — these are
  *not* Corresponding Source of the Rust crate and are not under this
  directory. They are runtime data of the Python engine, shipped one level up
  as `numfast/calibration.toml` and `numfast/calibration_dataset.json`. They
  are measurements taken on one Windows host; `NUMFAST_CALIBRATION_DIR`
  overrides them.
- **Third-party source.** None. The crate has no third-party dependencies (see
  `Cargo.lock`) and no third-party Rust or C source is vendored here or in the
  wheel. See `NOTICE`.

## Licence

AGPL-3.0-only, the same terms as the rest of the distribution. The full text is
in `LICENSE` at the distribution root, reproduced in
`numfast-<v>.dist-info/licenses/LICENSE`.
