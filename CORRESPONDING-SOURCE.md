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
| npm `@numfast/kernels` | `corresp_src/numfast-native/` in the tarball — the same 47 `.rs` + 6 build inputs + `SHA256SUMS`, vendored at `prepack` |

This document ships as `CORRESPONDING-SOURCE.md` at the sdist root, as
`numfast/_corresp_src/CORRESPONDING-SOURCE.md` in the wheel, and as
`corresp_src/CORRESPONDING-SOURCE.md` in the npm tarball.

`LICENSE` (the verbatim AGPL-3.0 text) and `NOTICE` reach
`numfast-<v>.dist-info/licenses/` in both sdist and wheel, and
`METADATA` carries `License-Expression: AGPL-3.0-only`. They are declared in
`pyproject.toml` under `[project] license-files` (PEP 639), so their arrival is
a property of the metadata, not of a copy step someone can forget. Both also
travel at the root of the npm package `@numfast/kernels`.

## The npm package: the same mechanism, vendored at `prepack`

`@numfast/kernels` conveys object code too — `dist/numfast_native.wasm`, a
compiled build of this same crate — so the same obligation applies to it. It is
discharged by the **same mechanism**, not by a second one:

`numfast-native/ts/corresp-src.mjs` runs at `prepack`, copies the crate's
`src/` plus the six build inputs above into `corresp_src/numfast-native/`,
copies this document beside them, and writes `SHA256SUMS` **from the bytes that
shipped**. `files` in `package.json` carries `corresp_src` into the tarball.
`npm run corresp` re-verifies an existing copy against the crate: every
`sha256sum -c` line must verify and every shipped byte must equal the crate's.

**Why vendoring and not relocating the files into `ts/`.** npm cannot include
files from outside the package directory, so the choice is a copy or a move. A
committed copy under `ts/` would be a second source of truth: the `.wasm` is
built from `numfast-native/src/`, the tarball would ship `ts/corresp_src/…`, and
`SHA256SUMS` — regenerated at pack time — would hash the stale copy and pass. The
obligation would then be satisfied in appearance only, which is the one outcome
this mechanism exists to make impossible. Copying at pack time keeps the shipped
bytes and the built bytes the same bytes by construction. The generated
directory is a build output and is gitignored, exactly as `dist/` is.

The **written-offer route is not used on this channel either.** It remains a
decision for the copyright holder, and taking it for npm alone would make two
channels of one project answer the same question differently.

## What is here, and why each item is here

Everything under this directory is exactly what `cargo build` needs to turn
this tree back into the shipped binary. Nothing else was copied in.

| Path | Why it is required to rebuild the binary |
|---|---|
| `Cargo.toml` | crate manifest: `[lib] crate-type`, and the normative `[profile.release]` (`opt-level`, `lto`, `panic = "abort"`) that fixes the code the binary contains |
| `Cargo.lock` | the resolved dependency graph. Here it holds exactly one package, `numfast-native`, i.e. the crate has no third-party dependencies |
| `src/**/*.rs` | the crate source |
| `REUSE.md` | the REUSE matrix the crate's own `src/lib.rs` cites: how the hot kernels share one physical core. Documentation, not build input -- but it is cited from source that ships, so it travels with it |
| `.cargo/config.toml` | project-local cargo config. On this project it selects the Windows link step; see "What is not here" |
| `tools/nf-link.bat`, `tools/nf-link.py` | the linker shim that `.cargo/config.toml` points at |

`numfast-native/SHA256SUMS` is written **at build time** from the bytes that
went into the wheel. It is the evidence that the shipped source is the source
in the repository, and it is regenerable: `sha256sum -c SHA256SUMS` must pass
after unpacking. The npm tarball carries the same file over the same 53 paths,
written the same way by `corresp-src.mjs`, and verifies the same way.

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

# from an unpacked npm tarball (the .wasm this package conveys)
cd corresp_src/numfast-native
cargo build --release --target wasm32-unknown-unknown
# -> target/wasm32-unknown-unknown/release/numfast_native.wasm
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
- **`numfast-native/tools/`** other than `nf-link.bat`, `nf-link.py` and
  `gen.py` — standalone benchmarks, parity harnesses and JS oracle drivers.
  They do not contribute a single byte to the binary. The crate's own
  correctness harnesses are the 37 `#[test]` unit tests inside `src/**`, and
  those travel here; this distribution carries exactly three files from
  `tools/`, not the harnesses a reader might expect to find under that name.
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
  are measurements taken on one Windows host. `NUMFAST_CALIBRATION_DIR`
  overrides them; otherwise the shipped file drives routing only where its
  `[hardware]` block matches the machine, and otherwise the planner says so
  (`alias['calibrate_info']()['routing']`) rather than routing on another
  machine's timings.
- **Third-party source.** None. The crate has no third-party dependencies (see
  `Cargo.lock`) and no third-party Rust or C source is vendored here or in the
  wheel. See `NOTICE`.

## Licence

AGPL-3.0-only, the same terms as the rest of the distribution. The full text is
in `LICENSE` at the distribution root, reproduced in
`numfast-<v>.dist-info/licenses/LICENSE`.
