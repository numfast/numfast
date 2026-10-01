# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 09 — Serialization + NFS (normative)

## PURPOSE

Persist off the hot path; stream facts in chunks, keep dictionaries resident.

## INPUT / OUTPUT

- INPUT: `Table` persist, CSV/`.dzst/.csv.zst` ingest, `dict` (resident hint) + `fact` chunks, `fact_chunk_n`.
- OUTPUT: files/chunks + `metadata {resident_dictionaries[], chunking_plan, incremental_reduce}`; `uploads=1 builds=1 fact_lookups=N`.

## OWNER / DEPENDENCIES

- OWNER: Storage + Planner. DEPENDENCIES: `03`, `05`.

## INVARIANTS

- `resident` (product/tax/dict) vs `streaming fact` (large fact tables); `dense local id→gather` preferred (hash is CPU-only); `build per chunk` forbidden; spill above `80% safe` is future work.
- `load_dzst→(int64[N,6],multiplier,power)` is a utility/backtest path, not a core type; `read_csv(.dzst)` follows the same `EncodingPlanner` path.
- NFS/artifacts cover the persist/ingest boundary only; benchmark adapters live outside core.

## GLOBAL DICTIONARY CONSTRUCTION for streaming (normative)

- Problem: a streaming fact arrives in chunks (`fact_chunk_n`), but the dictionary (`value→id`) must be global and stable before execution (otherwise `gather` on local ids is wrong).
- Architecturally allowed options:
  - **A. Two-pass ingestion** (`pass1→dictionary, pass2→encode`): the first pass collects the value set (or a reservoir sample under a memory bound), builds the dictionary, the second pass encodes chunks. Pro: exact global dictionary, stable ids. Con: two passes over the source (the source must be re-readable or spilled).
  - **B. Append-only global dictionary + stable IDs**: the dictionary grows incrementally over chunks (`new value → assign next id, never reassign`), encoded chunks immediately write `id`, the dictionary stays resident for the whole ingest. Pro: single pass. Con: ids depend on chunk order (deterministic only with fixed order + seed); the dictionary grows unbounded.
  - **C. Pattern encoding without a dictionary**: for columns with regular structure (dates/sequences/small enums) — a computable `value→id` mapping (`base+step`); the dictionary is not materialized. Pro: O(1) memory. Con: applies to pattern columns only (the detector must prove the pattern, else fallback).
  - **D. Hybrid pattern/int + dictionary**: try C, on failure fall back to B (or A when the source is re-readable); the decision is observable in `metadata.resident_dictionaries[]` (`{column, method: pattern|dict-two-pass|dict-append, ids_stable: bool}`).
- DEFAULT (normative): **D hybrid** — try pattern (C) first, else **B append-only** for single-pass sources; when the source is re-readable and the dictionary is unbounded (above the calibration-profile threshold) — **A two-pass** as the bounded-memory fallback. The choice is recorded in `metadata` before encode; changing the method mid-stream is forbidden.
- FALLBACK: on dictionary overflow (`>max_dict_entries` from the profile) — `abort ingest with an explicit error` OR `spill dictionary + recompute chunking_plan`, but never a silent `fact→GPU→CPU→dict lookup` in the hot path and never an undocumented shortcut around the dictionary contract.
- Guarantees: `uploads=1 builds=1 fact_lookups=N`; `build per chunk` forbidden; ids stable within one ingest (under B — only with fixed chunk order, recorded in metadata).

## PUBLIC / PRIVATE

- PUBLIC: `read_csv/load_dzst/optimize`. PRIVATE: chunk layout, dict encoding.

## WHAT MUST NEVER HAPPEN

- `fact→GPU→CPU→dict lookup`; persist in the hot path; benchmark-specific shortcuts in core.
