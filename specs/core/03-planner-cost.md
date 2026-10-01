# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 03 — Planner + Cost Model (normative)

## PURPOSE

Every decision is `data vs modeled cost` with an observable `why`. No `if N>X`.

## INPUT

- `ExecutionGraph`, `n` (after filter pre-mask), `reuse_factor/expected_reuses`, `vram_mb/safe`, `edge lifetime/bits/semantic`.

## OUTPUT

- Enriched graph: `metadata {repack_decisions, representation_transitions, edge_representations, access_structures, resident_dictionaries, chunking_plan, incremental_reduce}` + `execution_info {requested,actual,reason,driver,dispatches,h2d/d2h,cost_estimate}`.

## OWNER / DEPENDENCIES

- OWNER: Runtime (`planner/cost_model/representation/access_structure/resident_dictionary/chunking`). DEPENDENCIES: `06` (capability), calibration profile `calibrated_v1`.

## INVARIANTS

- Formula (normative — STRUCTURE, not numbers): `min(Σ(a_cpu·n+b_cpu), Σ(a_gpu·n+b_gpu)+H2D+D2H+dispatch·passes+compile+VRAM+repack−reuse_benefit)`; fused `f_fused·Σ+max(b)`. All coefficients (`a_cpu/b_cpu/a_gpu/b_gpu`), costs (`H2D/D2H/dispatch/compile/VRAM/repack/reuse_benefit`) and `f_fused` are **calibrated parameters** from the active calibration profile (`calibrated_v1` schema). No hardcoded transfer rates, launch costs, or fixed ratios are normative. Any single-device measurement is one profile example, not a rule.
- The calibration profile is the single source of numbers: `nf.calibrate()` fits parameters to the current hardware/driver; `EXPLAIN` must show `profile_version/source/age/matched/warning`. Changed hardware without recalibration = `warning` in `execution_info`, never a silent decision.
- Repack per edge (normative — rules, NOT thresholds): `transient→stay` is normative; `persistent` is decided by the cost model comparing `repack_cost vs reuse_benefit` under the active profile; the decision is observable in `metadata.repack_decisions`.
- Access structures: `BUILD iff build_cost<reuse_benefit && memory<VRAM_budget` (both quantities from the cost model + profile); without `expected_reuses` there is no build; `builds=1` per array is normative.
- Chunking is Planner-driven and the Runtime must split (see `04`, `operation.chunkable` property): if `chunkable=true` the Runtime must split the operation when `n>backend_limit`; if `chunkable=false` then `num_chunks=1` or an observable CPU fallback. Chunking guarantees `peak<full`; incremental Reduce covers associative ops (`sum/count/mean/min/max`) with scalar D2H only.
- Crossover points are profile-specific and illustrative only: which backend wins at a given N depends on the fitted `a/b` coefficients, operation mix, reuse, and transfers. Any `if N>X → gpu` branch in code is REJECT. Thresholds live only in the profile as fitted `a/b`, never in the spec and never in code.

## PUBLIC / PRIVATE

- PUBLIC: `explain/explain_analyze/trace/compile(backend=)`, `get_model_quality`, `select_backend` reason. PRIVATE: fitted `a/b`, calibration paths.

## WHAT MUST NEVER HAPPEN

- Threshold branches; a second Cost Model; repack via full CPU download; build per chunk; a decision without `why`.
- Numbers hardcoded in code/spec as norms; all numbers live in the calibration profile only.
