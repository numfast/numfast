# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# ADR-002: Columnar GroupBy Result (decision)

Status: ACCEPTED (Variant B). Default `result='dict'` frozen. Measurements: `benchmarks/h2o/overview.md`.

## Decision

- Public columnar result = existing `ColumnCarry` (`ukeys, counts, sums`) under explicit `result='carry'` + materializers `.to_dict_flat()/.to_dict_single()/.to_dict_multi()`.
- `evaluate()`/`Runtime` unchanged: they return `bufs[last]`; the columnar path is enabled by the `_carry_result` branch point, not by a Runtime rewrite.
- Variant A (flip default) REJECTED (breaks frozen default + GPU contract). Variant C (dict-only) is the fallback.

## Contract

- `ukeys: int64[C]` sorted; `counts: int64[C]`; `sums: col → int64[C] | float64[C]`; `mean` derived `sums/counts` (f64, cached); empty input → `ngroups == 0`.
- Carry has NO per-row validity sidecar — groups are valid by construction; invalid input is excluded before aggregation.
- Parity: `carry.to_dict_*() == dict-path` (scalar types, key order identical).
- Zero-copy reads via `np.asarray`/`memoryview`; `.tolist()` only inside `to_dict_*`.

## Boundaries (never change silently)

- GPU + `result='carry'` = explicit error (Carry lives in the CPU extension; duplicating it is forbidden). GPU dict mirrors stay bit-compatible with `to_dict_*`.
- `threads > 1` only for compat assembly. No dict+carry on the hot path at once.
