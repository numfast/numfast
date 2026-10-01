# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 11 — Security (normative)

## PURPOSE

Minimal guarantees.

## INVARIANTS

- `derive` string eval uses a restricted expression whitelist only; full SQL/Python is forbidden; complex logic uses `LazyExpr`.
- `read_csv` accepts explicit paths only; no network fetch in core; parsing follows the dtype contract (overflow per `error_contract`).
- `calibrate` writes only `numfast.calibration.toml`, `~/.cache/numfast/`, `session/` (JS: its own IndexedDB store); no silent overwrite of foreign TOML files.
- `initWebGPU({force:true})` is an explicit opt-in; no silent device capture.
- FROZEN ABI without an Issue — including security fixes, which go through the Guardian.

## WHAT MUST NEVER HAPPEN

- Silent downcast/overwrite/VRAM heuristics as hidden decisions; exec/import-hook in `_lib`.
