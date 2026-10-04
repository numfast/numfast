# NumFast Specs

**Status: HISTORICAL. Read this before reading anything in this directory.**

`core/`, `adr/` and `benchmarks/` were written for an earlier design generation
and are marked "frozen, v0.2 (2026-09-04)". They describe an API that **does not
exist in this tree**: `LazyExpr`, `.compute()`, `.data()`, `nf.zeros`, `nf.mean`,
`nf.var`, `ColumnView`, `returns()`, `rolling_mean()`, `nf.mods`, `nf.optimize`,
and a `WebGpuDriver` capability matrix with the operation names of that era
(`Map`, `MapBinary`, `Scan`, `RollingSum`, …). The live operation set is 33
different names, and the live public surface is the 44-name consumer facade.

What is still worth reading here: the *decisions* — the IR contract, the planner
cost model, the storage encodings, the benchmark methodology rulebook, the
security stance, and the four ADRs. What is not: anything presented as a
description of the current API.

For the API that exists, read [`docs/API.md`](../docs/API.md). For the shape of
the system as built, read [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

## Layout

- `core/` — normative specs (13 files: 00–12) + `conformance-profile.toml`
- `adr/` — architecture decisions (`columnar-result`, `resident-deferred`, `fused-elementwise`, `scan-chunking`)
- `benchmarks/` — benchmark-only docs (`execution-economics`, `h2o/overview`, `clickbench/overview`)

## Rules

- Core is normative **for the generation it describes**. History lives outside `specs/`.
- No benchmark-specific rules in `core/`.
- Conformance profile: `core/conformance-profile.toml`. Its numeric tolerances
  are still loaded by the test suite (`specs-rebuilt/conformance-profile.toml`
  is the copy the shipped tests read).
- ADRs record decisions; they do not override `core/`. `resident-deferred` is a
  deferred proposal; `fused-elementwise` is an Extension spec, not a core norm.

## Note on the conformance profile

`core/00-principles.md` still lists a guard written as
`grep -r "_wgpu"→0; …; ls _wgpu→0`. That module belongs to the superseded
generation; the check is kept as a "this must not come back" rule, not as a
description of the tree.