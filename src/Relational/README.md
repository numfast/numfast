# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# Relational family

`src/Relational/Join` — the CPU hash join. It moved 1:1 from
`src/Compute/CpuJoin`: the aliases and mods are unchanged and it depends on
`Core`, `IR` and `CPU`. Contracts — inner/left/NULL/dtypes/unique-keys — are
pinned by `tests/fast/test_ops_cpujoin.py`, against
`tests/oracles/hash_join_oracle.py`.

`GroupBy` / `Filter` / `Sort` / `Aggregate` — **member markers with no
implementation**: each registers its names so the Planner can carry the intent,
and none of them computes anything. The native extension point is the same alias
surface, so an implementation can be added later without changing a caller.

The same family's other members — `Segmented`, `Sssp`, `Router`, `PairInsert`,
`BitmaskSweep`, `DomainLUT`, `SortedLookup`, `CompositeGroup` — each carry their
contracts in their own `*.toml` `[metadata]`, their `_lib/` docstring, and a test
in `tests/fast/`.
