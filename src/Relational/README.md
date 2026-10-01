# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# Relational family (SPEC 12)

`src/Relational/Join` — CPU hash-join (переезд 1:1 из `src/Compute/CpuJoin`,
alias/mods 1:1, depends `Core/IR/CPU`). Контракты: inner/left/NULL/dtypes/
unique-keys — см. `specs/12-relational-family.md`.

`GroupBy/Filter/Sort/Aggregate` — member-маркеры без кода (Planner-намерение
без реализации). Native-точка расширения — тот же alias surface.

Совместимость: `_compat/cpu_join_shim.py` (disposable, удалить после миграции).
