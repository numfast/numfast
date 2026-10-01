# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 10 — Benchmark methodology (normative, benchmark-free core)

## PURPOSE

Reproducible measurements, separated from core norms.

## INPUT / OUTPUT

- INPUT: `benchmarks/<category>.py + run_all.py`, synthetic data, seed 42.
- OUTPUT: `time+throughput+memory` + stage breakdown (`compile/alloc/H2D/dispatch/D2H/pool cold vs warm`) → `.project_index/benchmarks.json` after validation.

## OWNER / DEPENDENCIES

- OWNER: Performance. DEPENDENCIES: `00`.

## INVARIANTS

- Correctness first (`max_diff`), then performance; `OLD→NEW/RATIO`; `H2D/D2H/dispatches` always reported; integrity `loss>1%=STOP` before measurement.
- Modularity: each new measurement is a new file; editing existing benchmarks is forbidden.
- `verify_ast/stress_test_ast` are reference artifacts, keep them.
- Benchmark-only workarounds are measurement tools, not architecture. They must never appear as core rules; concrete store findings live in `specs/benchmarks/`, never here.

## PUBLIC / PRIVATE

- PUBLIC: results in `.project_index/benchmarks.json`. PRIVATE: raw logs in `session/`.

## WHAT MUST NEVER HAPPEN

- Bulk (>1000) CPU measurements presented as a claim where a GPU path exists; a claim without a stress test (`≥1000 random cases, seed 42 + leak/double-release/device-loss`); backtest research without an explicit request.
