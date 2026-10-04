# `docs/` — what is here, and what to believe

This directory holds two different kinds of document. They are not
interchangeable, and the difference is stated here because a reader who assumes
otherwise will form a wrong belief about the API.

## Describes the current tree — start here

| Document | What it covers |
|---|---|
| [API.md](API.md) | the 44-name consumer surface, the package boundary, the kernel-level names, and every guard |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the IR, the Builder Extension model, the planner, the CPU driver, the native Rust kernels, the GPU split, the WASM path, and where the semantic contract lives |
| [EXAMPLES.md](EXAMPLES.md) | runnable programs with their real output |

The repository root also carries [README.md](../README.md),
[BENCHMARKS.md](../BENCHMARKS.md), [KNOWN_LIMITATIONS.md](../KNOWN_LIMITATIONS.md),
[README.packaging.md](../README.packaging.md),
[CORRESPONDING-SOURCE.md](../CORRESPONDING-SOURCE.md),
[COMMERCIAL-LICENCE.md](../COMMERCIAL-LICENCE.md) and [SECURITY.md](../SECURITY.md).

## Internal design record — historical, mostly not in English

These are the working record of how the engine was built and audited. They are
kept because they are the evidence behind several claims, and several of them
contain numbers this work has since superseded. **They are not an API reference
and they are not current.**

| Document | Language | Note |
|---|---|---|
| `NUMFAST_LANGUAGE.md`, `audit_history/CORE_CLEAN.md`, `COST_TRAVEL_AUDIT.md`, `JS_RECON.md`, `MIGRATE_1.md`, `NATIVE_ALL.md`, `NATIVE_AUDIT.md`, `WASM_1.md` (8 files) | Russian | the audit trail: native reconstruction, WASM, JS recon, migration, cost-travel audit. Predates the current generation. |
| `audit_history/MISSION_Q36_LOG.md`, `RUNTIME_MANIFEST.md`, `RUST_WGPU_KNOWLEDGE_MAP.md` (3 files) | English | session logs and a runtime manifest from the same period |

`docs/` therefore holds 11 documents: 8 substantially Russian, 3 English.

The same applies to two other directories of internal record that a reader will
meet: `develop/` and `tests/research/`. `develop/benchmarks_public/` holds five
benchmark summaries whose source artefacts were deleted — each `summary.json`
says so in a machine-readable `provenance` field, and
[BENCHMARKS.md §3](../BENCHMARKS.md) lists what they claim and why the claims are
superseded.

## Also historical

`specs/` is marked "frozen, v0.2" and describes an earlier design generation:
`LazyExpr`, `.compute()`, `.data()`, `nf.zeros`, `nf.mean`, `ColumnView`,
`WebGpuDriver`. None of that is the API in this tree. Read it for design
decisions; read [API.md](API.md) for the surface that exists.

## About the language of these documents

User-facing documentation in this repository is in English. The internal record
above is substantially Russian and has deliberately **not** been translated or
rewritten: it is a record of what was decided and measured at the time, and
rewriting it would destroy the thing that makes it evidence.