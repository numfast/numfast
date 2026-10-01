# NumFast Specs

Public specification layout. Version: v0.2 (2026-09-04). Status: frozen.

## Layout

- `core/` — normative specs (13 files: 00–12) + `conformance-profile.toml`
- `adr/` — architecture decisions (`columnar-result`, `resident-deferred`, `fused-elementwise`, `scan-chunking`)
- `benchmarks/` — benchmark-only docs (`execution-economics`, `h2o/overview`, `clickbench/overview`)

## Rules

- Core is normative. History lives outside `specs/`.
- No benchmark-specific rules in `core/`.
- Conformance profile: `core/conformance-profile.toml`.
- ADRs record decisions; they do not override `core/`. `resident-deferred` is a deferred proposal; `fused-elementwise` is an Extension spec, not a core norm.
