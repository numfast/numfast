# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# SPEC 12 — Relational family (normative)

## FAMILY

`src/Relational/{Join,GroupBy,Filter,Sort,Aggregate}`. Implementation only in
`Join/_lib/join.py` (1:1 move from `src/Compute/CpuJoin`, alias/mods 1:1);
the rest are member markers without code (toml+py+empty `_lib`).

## JOIN CONTRACTS (normative, from the implementation)

- `inner(x_keys, x_v1, build)`: probe-order compact `(keys, v1, v2)`.
- `left(x_keys, x_v1, build)`: full `(keys, v1, v2, valid)`; miss -> `v2=0`,
  `valid=0` (NULL == invalid, never NaN/sentinel).
- `dtypes`: int32 keys/payloads, int64 chk accumulators, float64 display only.
- `unique-keys`: build-side keys are unique, else `ValueError` (dupes collapse).
- `chk`: `chk_inner=sum(v1)+sum(v2)` int64; `chk_left` sums `v2` over `valid`.

## PLANNER INTENTION (no implementation)

Planner intention without implementation: future `ir_join/ir_groupby/ir_filter/ir_sort`
map onto family members; SPEC 12 changes no IR/Planner code.

## GROUP_COUNT_DISTINCT CONTRACT (CPU op — not a new IR)

- `group_count_distinct(values, keys)`: CPU op inside the driver, no new
  `ir_group_count_distinct` is created.
- Contract: `_valid_mask` -> compact -> `_sort_perm` over `(keys, values)` ->
  one vectorized pass (group boundaries + distinct transitions) ->
  `(ukeys sorted, counts int64)` + `#strategy="sorted_dedup"`.
- Output: `{key: nunique}` dict (keys sorted), counts int64; the validity
  sidecar is filtered before sorting (invalid rows excluded).
- Co-aggregates (sum/count/mean) are separate nodes, not part of this op.
- Composite keys arrive packed from upstream; this op takes a single integer
  keys input.
- `chunkable=false`: single-pass ordering requirement (single-chunk sort
  contract), not a mathematical prohibition — chunked backends must re-sort
  each chunk (as for `groupby`/`sort`).

## NATIVE EXTENSION POINT

Native extension point: a future hash-join/sort/groupby kernel in
`numfast-native` under the same `alias` surface; the CPU join is the parity
reference.
