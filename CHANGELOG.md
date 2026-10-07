# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Scope of this file

**This file starts at 0.2.1.** Everything before that belongs to a superseded
design generation — a WebGPU/wgpu-py-centric engine with a `Series`-and-`.compute()`
API and an npm package of its own. That generation is not carried forward: its
history is not part of this repository's published history, and none of its
capabilities are claimed here.

If you are looking for the previous generation's documentation, it is not in this
repository. Neither is its audit record, its research material or its benchmark
archive: 0.2.1 is also the release that reduced the repository to the product,
and the reason each category left is recorded under 0.2.1 → Changed.

Release state: publishing is a **manual** step. There is no publish workflow in
this repository and no `v*` tag trigger in CI, so no commit in this repository can
publish anything by itself.

---

## 0.2.1 — 2026-10-05

The first release of the current generation: a columnar engine with a Rust
compute-kernel layer, a GPU path for a declared subset of operations, and a small
new consumer facade.

### Fixed before release — the information surface

The release is one artefact set and one description of it. The following were
measured against the built artefacts and corrected; each was a place where two
channels described the same release differently, or where one of them described
it wrongly.

- **PyPI now renders this release's own page.** `pyproject.toml` pointed
  `readme` at `README.packaging.md`, a developer note containing a
  `python -m build` recipe, so the project's PyPI long description was a build
  instruction. It now points at `README_PYPI.md`, which is added to `MANIFEST.in`
  so the wheel built from the sdist can still find it. `README.packaging.md`
  stays, and stays linked, as the developer-facing note it actually is.
- **`[project.urls]` added.** The PyPI page carried no link to the source, the
  documentation or the changelog.
- **The npm page said 85 kernels. There are 86.** `numfast-native/ts/README.md`
  hardcoded `85` in nine places, `68` unwrapped, `20` i64 exports, a stale
  `staticDataEnd` and a stale `.wasm` byte count, while the same repository's
  `README.md`, `docs/ARCHITECTURE.md` and `CHANGELOG.md` all said 86, and the
  generated `SCOPE.statement` said 86. Every one of those numbers now matches
  the build. The generated statement is what makes it stick: it is derived from
  `TOTAL_EXPORTS`, which the build checks against the artefact's bytes.
- **The npm package claimed "runnable in any host with a WebAssembly runtime" and
  "`LICENSE` and `NOTICE` travel with the package".** Measured: the packed
  tarball bundles for the browser with three errors (`node:module`, `node:fs`,
  `node:crypto`), so it is **Node-only**, and it carried no licence file at all.
  `LICENSE` and `NOTICE` now ship at the package root, and the host claim says
  what is true: the `.wasm` is host-independent, the package's entry point is
  not. The browser channel is recorded as unsupported in
  `docs/INSTALL.md` rather than implied to work.
- **The npm package had no links.** `repository`, `homepage` and `bugs` added, so
  the npm page leads back to the source and its issues.
- **`build.mjs` asserted the version against `package.json` only.** `Cargo.toml`
  is checked too, so a mirror that drifts stops the build instead of shipping a
  kernel package that cannot be traced to a commit. The function's docstring
  also still claimed `Cargo.toml` said `0.1.0`; it has said `0.2.1` for some time.
- **The stated test results were stale in every figure.** Re-measured on a
  `git archive HEAD` extraction into an empty directory (Windows, Python
  3.14.6): clean checkout **772 passed, 4 skipped, 0 failed in 156.20 s**; the
  same run without `wgpu` **707 passed, 5 skipped, 64 failed**; installed wheel
  **503 passed, 5 skipped, 0 failed in 122.76 s**. The README quoted 771/5 and
  473/4 and named a skip table whose modules do not match what skips.
  `ci.yml` quoted a Linux figure of "10 failed, 627 passed"; measured on
  WSL2/Ubuntu 24.04 it is **74 failed, 664 passed, 38 skipped** without the
  native library and **10 failed, 729 passed, 37 skipped** with it built from
  its own Corresponding Source.
- **`docs/INSTALL.md` added** — one installation page with the five sections
  Python/PyPI, JavaScript/npm, Browser/WASM, Linux native, Windows native. Each
  states its install command, its requirements, a minimal example, how to verify
  it, and — where the channel is not delivered — **"not supported"**.
- **`SECURITY.md` shipped a placeholder mailbox.** `security@numfast.example`
  uses the reserved `.example` TLD and reached nobody. Removed; the policy is
  GitHub private vulnerability reporting alone, which needs no mailbox.
- **The published registries already carry the previous generation.** PyPI
  `numfast` holds `0.0.1` (yanked), `1.0.0a1` and `1.0.0a2`; npm carries
  `@numfast/numfast` 1.0.0-alpha.1/2/3. `pip install numfast` selects 0.2.1
  because a final release outranks a pre-release, and `npm install
  @numfast/kernels` is a different package name. A pre-release pin installs the
  old generation. This is now stated on both landing pages rather than left for
  a reader to discover.

### Known gaps in this release, recorded rather than papered over

- **The npm package conveys the `.wasm` and no Rust source.** AGPL-3.0 §6 is
  discharged by construction for the Python wheels and **not** for the npm
  tarball. `CORRESPONDING-SOURCE.md` names the two routes that close it and says
  plainly that neither is a documentation change.
- **No browser entry point.** The `.wasm` is host-independent;
  `dist/bridge.js` bundles for the browser, but it is unreachable through the
  published `exports` map and `dist/index.js` is Node-only.
- **No Linux native artefact.** The library is buildable on Linux with one
  `cargo build --release` and no `zig` — `.cargo/config.toml` is scoped to the
  Windows target — and the wheel's own Corresponding Source is enough to do it.
  It is not published, no Linux wheel exists, no Linux CI job exists, and the
  suite on Linux is red. Recorded, not fixed by building a pipeline for it.

### Added

- **The consumer facade — 44 public names, new in this release.** `app()`,
  `from_numpy` / `from_arrow`, `query()` → `filter` / `derive` / `group` /
  `reduce` / `sort` / `limit` → `compile`, plus a 23-name expression vocabulary
  (`+ - * / % **`, comparisons, `and_`, `not_`, `isin`, `is_null`, `cumsum`,
  `shift`, and five text predicates). `jobs()` exposes the IR; `explain()` prints
  it without executing.
- **Loud refusals, pinned by tests.** A NULL `group` key, multi-measure `group`,
  arithmetic on a TEXT column, ordering on a TEXT column, an array exponent, an
  out-of-`int32` value in an `int32` column, a BIGINT key or literal, a CPU-only
  operation requested on the GPU, and a fused indicator plan carrying a parameter
  key no operation consumes — each raises and names the cause.
- **A native Rust compute-kernel layer** (`numfast-native`, no third-party
  dependencies), loaded through `ctypes`: sort and lexsort, hash join, group-by
  variants, pack/unpack, carry, segmented reduce, row-wise k-way reductions,
  single-source shortest paths, cost travel, bounded select, pair insert, text
  encode and RNG.
- **AGPL-3.0 §6 Corresponding Source inside the wheel.** 47 `.rs` files plus the
  crate manifest, lockfile, `REUSE.md`, cargo config and link shims under
  `numfast/_corresp_src/numfast-native/`, with a 53-entry `SHA256SUMS`. The
  written-offer route is deliberately not used.
- **Two wheels at one version.** `py3-none-win_amd64` carries the native library;
  `py3-none-any` does not, and says so through `native_info()` rather than
  failing at run time. Python `>=3.11`.
- **A WebAssembly build of the same kernels** as `@numfast/kernels`: 86 function
  exports, one memory, zero imports, 17 typed wrappers. A portable kernel library,
  not a compute core — there is no executor inside the `.wasm`.
- **A measured calibration profile** (`calibration.toml`, `source =
  "measured:seed42"`) that travels with both wheels and drives `backend='auto'`
  routing.
- **Two wheels' worth of packaging checks** in CI: build, install into a clean
  venv, import; and a Builder-free test subset.

### Changed

- **The repository was reduced to the product.** Removed: benchmark archives and
  result JSON, research experiments, the ClickBench material and every figure
  derived from it, an internal auto-tuner, audit reports from a previous design
  generation, a superseded `specs/` tree, session scratch, and a self-labelled
  disposable shim. What remains is the engine, its tests, the Rust Corresponding
  Source, the npm kernel package and user-facing documentation. Nothing removed
  was reachable from `full.toml`, so no Extension changed.
- **No wall-clock performance number is claimed in this distribution.** The
  artefacts that backed them, and the scripts that produced them, are gone; the
  benchmark inputs were never in the repository and cannot be. The README and
  `README_PYPI.md` state the performance characteristics as design facts
  instead, and say plainly that none is measured here.
- The GPU path is described by what it is: **15 of 33 operations**, measured on
  one RTX 2060 over Vulkan, with the other 18 CPU-only. **No GPU speedup is
  claimed**; the recorded GPU timings at the sizes measured were slower than the
  CPU path.
- The `sort` NULL-key refusal was **removed**. Its stated premise was false — NULL
  rows come *last*, matching pandas `na_position="last"` — and NULL sort keys are
  now supported and pinned by a parity test.
- Five of the six arithmetic operators on a TEXT column were unguarded and
  computed on dictionary codes. All six now refuse before any node is emitted.

### Known limitations

[`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) is the register, and it is not a
short one. The headline items: `window`, `or_`, `join`, `replace` and
`fill_null` are absent from the consumer surface; `group` silently drops a key
whose measure is entirely NULL; `to_numpy()` reads an integer NULL as `0`; and
there are no cheap strided views and no resident GPU execution.

### Performance

No figure is quoted here, for the reason given under Changed. What can be said
without a harness: one planner call per query means per-query overhead does not
scale with the number of operations in the query, which favours small
aggregate-shaped queries; high-cardinality grouping is several times behind
DuckDB structurally, because the CPU driver is a NumPy reference executor; and
the GPU produces the same computed results rather than speed.

