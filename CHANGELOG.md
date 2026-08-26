# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0a1] - 2026-08-26

### Added

- Public `nf.*` API: series creation, random generation, `Series` operations,
  comparisons, filter, topk, groupby
- JavaScript/WebGPU parity surface: 26 functions with identical semantics
  in Python and JS runtimes
- PCG-u32 cross-language bit-exact random generation (Python == JS, same seeds,
  same streams)
- WebGPU GPU driver (`{gpu: true}` execution paths)
- Documentation set E1-E8: verified executable examples
  (see `docs/EXAMPLES.md`, `docs/QUICKSTART.md`)

### Fixed

- `Compare`: output dtype now correctly capped (no accidental upcast)
- `Map`: function uniforms packed correctly for multi-uniform signatures
- npm test plumbing (`npm test` runs the full parity suite end-to-end)

### Known Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for the current list of
supported dtypes, shapes, and operations.
