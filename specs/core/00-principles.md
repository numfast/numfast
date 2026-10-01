# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 00 — Principles (normative)

## PURPOSE

Define the immutable restart rules. Violation = revert (Guardian).

## INPUT / OUTPUT / OWNER / DEPENDENCIES

- INPUT: any change / Extension / benchmark.
- OUTPUT: `PASS/REJECTED` against the checklists below.
- OWNER: Guardian + Specification. DEPENDENCIES: none (this is the root).

## INVARIANTS

1. One IR (`ExecutionGraph`), one Runtime singleton, one WebGPU stack, one `backend_contract`.
2. Explicitness: `backend="gpu"` without capability = `RuntimeError`; `auto` = `min(cpu_total,gpu_total)` + observable `execution_info`.
3. GPU intermediates are resident; full download for repack is forbidden; `analyze_range` reads 16B of metadata only; one `H2D` and one `D2H` per chain.
4. TOML-only manifests (`alias==mods`); JSON is forbidden.
5. Types: int32 logical, int64 accumulators/scaled, f32 GPU default, f64 explicit/display only.
6. Bulk compute (`>1000 ops`) uses the GPU path through the cost model (CPU is an oracle on small samples only); `transient never pack` is normative; there is no fixed `N→GPU` threshold in code or spec — all numbers live in the calibration profile only (see `03`).
7. Seed 42; stage breakdown; max_diff first; loss>1% = STOP.
8. Conflict priority: NumFast guardrails > ponytail > expert skills.

## PUBLIC / PRIVATE

- Public: this file + Guardian checks. Private: none.

## WHAT MUST NEVER HAPPEN

- A second Runtime/IR/stack/model; silent fallback; `sys.path` in `_lib`; cross-Extension private imports outside the hierarchy (see `07` — only the parent/common level or an explicit `depends/alias/setup`); copying a helper to avoid an import; `float32 lossless`; f64 default; build without reuse/VRAM accounting; materialization of intermediates; edits to frozen components without an Issue; production claim without ≥1000 fuzz cases; hardcoded cost numbers or pseudo-thresholds as norms; confusing `schema.offset` with `block.base` (see `05`: `logical_value → schema_transform() → physical_value → block_transform() → packed_value`).

## Guardian checks

`grep -r "_wgpu"→0; grep extension.json→0; grep "if.*>.*5000.*gpu"→0; ls _wgpu→0; pack_bits∉{8,16,32} on GPU→error/fallback observable.`
