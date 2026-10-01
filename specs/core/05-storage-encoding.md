# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 05 — Storage + Encoding (normative)

## PURPOSE

Separate two boundaries: Storage (disk/persist) and GPU representation (resident compute).

## INPUT

- `nf.table/read_csv(path,schema)`, canonical schema (`version,name,columns[{name,dtype,encoding:{type:auto}}]`), `EncodingPlanner`, CSV/`.dzst` ingest.

## OUTPUT

- `Table` proxy + `ColumnView` zero-copy + `PackingPlan/ColumnLayout` + `dzst B=256` traversal-delta for all rows on persist.

## OWNER / DEPENDENCIES

- OWNER: Storage. DEPENDENCIES: `03` (repack decisions), `02` (`as_expr` bridge).

## INVARIANTS

- Canonical formula (literal, must not be confused): `logical_value → schema_transform() → physical_value → block_transform() → packed_value`. Reverse path: `packed_value → block_untransform() → physical_value → schema_untransform() → logical_value`.
- `schema.offset` is a SEMANTIC value transform (part of `schema_transform()`): together with `scale` it maps `logical ↔ physical` (`physical = (logical − offset) / scale` for scaled-int; identity with explicit `offset=0, scale=1` for plain int/float). It lives in schema/column metadata, is identical for all blocks of a column, and affects semantics (comparisons and arithmetic must account for `schema_transform`). It must not be changed for local compression gains.
- `block.base` is LOSSLESS COMPRESSION METADATA of one block (part of `block_transform()`): e.g. `base=min(physical_block)`, `delta=physical−base`, then `bitpack(delta, bits)`. It lives in the per-block header (`B=256`), differs across blocks, and does NOT change semantics — `block_untransform` must restore `physical` bit-for-bit (for int/scaled-int) before `schema_untransform` applies. Re-compression (`base=min(block)` and similar) is the right of the block encoder only, within these bounds.
- Storage/transfer form: **scaled-int + per-block base/delta/bitpack**.
- `Storage.pack_rows` covers `Disk/Code→Host` only; GPU `quantize/pack/repack` is resident only (`execute/execute_fused`); `analyze_range` reads 16B only.
- `NumericRepresentation {physical_dtype,scale,offset,signed,bits,semantic,error_contract}`; `float32 bounded exact until 2^24, never lossless`; `int64>2^24 via f32→explicit error`; overflow is contract-aware (`lossless→OverflowError`, `bounded→clamp+warning+observable`).
- `new_col(derive=)` Create→Swap→Free (new id/generation); logical derived columns are 0-bit virtual.
- `optimize()` is the canonical repack through Storage; `DZST/load_dzst` is a backtest/utility format, not a core schema type.

## PUBLIC / PRIVATE

- PUBLIC: `table/head/tail/to_numpy/filter/where/sort/update/append/lookup/info/optimize/new_col/release`, `ColumnView.to_list` (final readback). PRIVATE: registry, plans, layouts, buffers, `bucket/scale/packed`.

## WHAT MUST NEVER HAPPEN

- `pack_rows` on GPU intermediates; `.to_numpy/.data/restore` in the hot path; `12-bit` on GPU (CPU fallback only); compute in Storage.
