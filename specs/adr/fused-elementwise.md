# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# ADR: Fused elementwise indicators (extension spec, not core norm)

## STATUS

Extension spec (additive, non-normative for core): no frozen component is
touched (Runtime, Planner, Drivers, IR, cost model unchanged); no existing
kernel is modified. New Extension `Fused` (`src/Compute/Fused/`, registered
in `full.toml`). This document constrains the `Fused` Extension only; it does
not establish a general fusion norm.

## FUSION BOUNDARY (explicit, enforced by errors, never silent)

FUSES: ONE float32 input series (N) -> K outputs, one thread per row,
one WGSL source, ONE dispatch:
- pointwise maps: `axpb` (a*x+b), `map_div` (x/v)
- lag reads: `mom` (x[i]-x[i-lag]), `roc` ((x[i]/x[i-lag]-1)*100),
  `returns` (lag-1, no x100)
- same-N sliding windows (min_periods = window, full windows only):
  `sma`, `rsum`, `rmin`, `rmax`, `rstd` (ddof), `rsi` (Cutler-style simple
  means over window deltas, d_0 := 0; flat -> 50.0), `boll_up`/`boll_lo`
  (mean +/- k*sd), `zscore` (sd==0 -> 0.0), `stoch_k` (hi==lo -> 50.0)
- i32 signal masks in the same thread: `above_sma`, `rsi_lt`/`rsi_gt`,
  `cond_and` (product of two EARLIER i32 masks, same thread i)

NOT FUSED (explicit ValueError naming this file):
- groupby/groupby_multi: different key spaces, per-key atomic domains
  sized by M != N -- one-pass per-thread model cannot express
- sort: global order, multi-pass bitonic
- join/lookup: hash tables, probe order
- scalar reductions to M=1: tree + host merge need a second dispatch shape
- EMA/recursive filters: cross-thread sequential dependency (y[i]<-y[i-1])
- multi-input graphs: one input series in this spec; non-float32 input: cast first

## CODEGEN CONTRACT

- `compile_spec([{name, op, params?}])` -> `{source, names, dtypes, pu_vals,
  pf_vals, ...}` (pure, no GPU). cond_and deps must be earlier i32 masks.
- Self-contained emitters: each output recomputes from the input (except
  cond_and sibling-mask reads, same-thread write-then-read, ordered) =>
  fused[K] vs separate[K x 1] bit-exact by construction (test asserts).
- Uniforms (stride-16 legal): `pu: array<vec4<u32>,2>`
  (pu[0].x = N, then windows/lags/ddof), `pf: array<vec4<f32>,2>` (float
  scalars). Literal lane indices -- no dynamic indexing. Source depends
  only on op sequence + dtypes => pipeline cached across N and values.
- Caps: N <= 65535*256 (one dispatch dim x); <= 7 int params;
  <= 8 float params; float32 rank-1 only; n == 0 -> empties, no dispatch.
- Toolchain constraints: `x != x` must not be used for NaN (may fold to
  false); NaN detection is `(bitcast<u32>(v) & 0x7F800000u) == 0x7F800000u &&
  (bitcast<u32>(v) & 0x007FFFFFu) != 0u`; uniform scalar arrays need
  vec4 packing (stride-16 rule).

## CORRECTNESS CONTRACT

- fused vs separate: BIT-EXACT incl. NaN placement (seed 42, random-walk
  quotes + 2% NaN injection + edges: n=0/1/<w, const, all-NaN, w=1,
  lag>=n, ddof=1, div-by-zero, flat-RSI/stoch/zscore branches).
- vs numpy f64 oracle: f32 profile tolerance; documented per-op absolute
  bounds for cancellation-prone ops only (roc (ratio-1)*100: atol 5e-5;
  zscore: atol 1e-3; derived from f32 eps, not tuned).
- The new Extension owns a minimal local wgpu context: GPU internals are
  not public surface, and touching the GPU extension for reuse is
  forbidden -- duplication of the small dispatch block is the compliant choice.

## MEASURED (warm, seed 42; tests/heavy/bench_fused_indicators.*)

- 5 indicators, N=1M/4M/10M: fused==separate bit-exact; e2e ~= parity
  (D2H of 5 full outputs dominates BOTH modes equally -- same bytes);
  submit 1x0.5ms vs 5x~0.5ms; dispatches 1 vs 5.
- K-scaling law @10M: K=1 saved 0.00ms / K=5 saved 1.87ms / K=8 saved
  3.41ms of submit overhead; fused submit flat, separate linear.
- Vision payoff (outputs feeding further GPU work without D2H) is
  architectural, not measured here: with full D2H the win is dispatch
  overhead only; it grows with K and dominates when D2H shrinks
  (masks/scalars) or downstream stays resident.

## INTEGRATION POINT (future work, out of scope for this spec)

Planner cost model untouched: Fused is standalone (`fused_build/run/
source/oracle` via kernel.alias). Routing ExecutionGraph elementwise
subgraphs into it would be a Planner change -- explicitly out of scope.
