# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# Relational family (SPEC 12)

`src/Relational/Join` — the CPU hash join. It moved 1:1 from
`src/Compute/CpuJoin`: the aliases and mods are unchanged and it depends on
`Core`, `IR` and `CPU`. Contracts — inner/left/NULL/dtypes/unique-keys — are in
`specs/core/12-relational-family.md`.

`GroupBy` / `Filter` / `Sort` / `Aggregate` — **member markers with no
implementation**: each registers its names so the Planner can carry the intent,
and none of them computes anything. The native extension point is the same alias
surface, so an implementation can be added later without changing a caller.

Compatibility: `_compat/cpu_join_shim.py` is a disposable re-export shim that
keeps the old `src/Compute/CpuJoin` import path alive. Nothing in the tree imports
it any more; it is kept only so that a stale external path does not break, and it
is marked for deletion.