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
repository.

Release state: publishing is a **manual** step. There is no publish workflow in
this repository and no `v*` tag trigger in CI, so no commit in this repository can
publish anything by itself.

---

## 0.2.1 — 2026-10-05

The first release of the current generation: a columnar engine with a Rust
compute-kernel layer, a GPU path for a declared subset of operations, and a small
new consumer facade.

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
  crate manifest, lockfile, cargo config and link shims under
  `numfast/_corresp_src/numfast-native/`, with a 52-entry `SHA256SUMS`. The
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

- Claims were reconciled against the artefacts they came from. Two were removed
  because the artefacts contradict them: "ClickBench 8/43 supported" (this tree
  measures **34 of 43**) and "Q30 affine 43.02× faster" (an unconfirmed prototype
  the current code does not reproduce).
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

34 of 43 ClickBench queries run and return a verified answer; 9 do not (8 from
BIGINT narrowing, 1 from a missing conditional primitive). NumFast is **7.39× to
2.59× slower than DuckDB** on eight high-cardinality grouping and distinct-count
queries, and faster on the small aggregate-shaped ones. Every number, its
command, its environment and whether it can be re-run is in
[`BENCHMARKS.md`](BENCHMARKS.md). No wall-clock number is claimed anywhere that was
not produced by a command named there.

