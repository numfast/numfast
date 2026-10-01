# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 08 — JS boundary (normative)

## PURPOSE

Python-CPU === JS parity; GPU uses the same WGSL through wgpu/Dawn.

## INPUT / OUTPUT

- INPUT: `@numfast/numfast`, `nf.zeros/arange/series/scan/sort/groupby`, `opts.gpu`, `nf.initWebGPU({force:true})`, `nf.calibrate()` (IndexedDB).
- OUTPUT: same semantics/dtypes/NaN/null; `max_diff` first.

## OWNER / DEPENDENCIES

- OWNER: JS adapter (isolated module). DEPENDENCIES: `01`, `02`, `06`.

## INVARIANTS

- ES2024+, TS strict, ESM only; `subarray` (view) vs `slice` (copy) used deliberately; heavy work in a Worker, transferables for large buffers; event-loop blocks >50ms forbidden; sync IO on the hot path forbidden.
- One WGSL source through wgpu bindings; runtime models are not mixed (Node≠browser≠WebGPU≠WASM, adapters at the boundaries).
- `tan` unavailable in JS; `true_range/state_kernel` not wired; `>1k` elements use TypedArray/GPU ops, not per-element loops; silent f32↔f64 conversion forbidden.

## NUMERIC PARITY (normative)

- **Semantic parity** (normative): API semantics are identical in Python-CPU and JS — same names/arguments/operation order/defaults, same errors (`log(0)→ValueError`, `Table.join→NotImplementedError`), same lazy→eager lifecycle. A semantic-parity violation = REJECT.
- **Numeric contract** (normative, replaces any bit-for-bit float expectation):
  - `integers (int32 logical)` — exact: Python/NumPy/JS must match bit-for-bit.
  - `scaled integers` — exact after `schema_untransform`: `physical` bit-for-bit, `logical` within `scale/2` (half a quantum) under documented rounding (`round-half-even`, stated in schema).
  - `float (f32 default, f64 explicit)` — tolerance/ULP contract, NOT bit-for-bit: `max_abs_diff ≤ max(atol, rtol·|ref|)` with `atol/rtol` from the calibration/conformance profile (conformance defaults: `f32: 1e-5 abs / 1e-5 rel or ≤4 ULP`; `f64: 1e-12 abs/rel or ≤2 ULP`); comparison is `max_diff first` with index/operands reported. Requiring bit-for-bit float equality across Python/NumPy and JS is FORBIDDEN (different libm/SIMD/WGSL).
  - `NaN/±0 explicit semantics` (normative): `NaN != NaN`; `NaN` propagates through Map/Reduce (Reduce with NaN → NaN + warning unless `skipna` is explicit); `+0 == −0` in comparisons, but `1/+0=+Inf`, `1/−0=−Inf`, `signbit` preserved in bitpack round-trip; serialization distinguishes `NaN/Inf/−Inf` (JSON as strings, dzst as bit patterns).
- Conformance tests must check both layers separately: a `semantic suite` (same calls/errors) + a `numeric suite` (exact for int / tolerance for float / explicit NaN/±0).

## PUBLIC / PRIVATE

- PUBLIC: JS API mirrors Python. PRIVATE: Dawn handles, IndexedDB store `numfast-calibration`.

## WHAT MUST NEVER HAPPEN

- A JS shortcut breaking parity; Node API in the browser bundle and vice versa; `any/ts-ignore` without a reason.
