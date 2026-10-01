# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 06 — Drivers GPU/CPU (normative)

## PURPOSE

One contract, two executors: CPU oracle and WebGPU workhorse.

## INPUT / OUTPUT

- INPUT: `ExecutionPacket[]` + uniforms + capability query.
- OUTPUT: resident buffers → scalar/vector D2H; `ValueError` on dispatch overflow; `RuntimeError` on explicit-gpu without capability.

## OWNER / DEPENDENCIES

- OWNER: `CpuDriver` (numpy, small-N, conformance) + `WebGpuDriver` (wgpu-py→Vulkan/DX12, browser Dawn with `force:true`). DEPENDENCIES: `00`, `03`.

## INVARIANTS

- Capability matrix: Map/MapBinary/Reduce/Scan/RollingSum/Shift/Gather/Compare/Fill/StateKernel/MatMul/Sort — GPU supported (with stated caveats); Filter/GroupByMean/rank/lookup/rolling_window-2D/count/storage — CPU-only observable. Each capability entry includes a `chunkable_hint` + `backend.max_dispatch/max_buffer_bytes`; the Planner reads limits only from there.
- Pack bits `{8,16,32}`; 12-bit→CPU fallback; Sort power-of-two on GPU else CPU; Scan runs as a whole dispatch (follows from `chunkable=false`, see `04` — revisable as an operation property); MapF64 segfault-guard (f64 explicit only).
- VRAM from `nvidia-smi/rocm-smi` only, else `"unknown"`; never `maxBufferSize`.
- FROZEN: Driver ABI, Kernel ABI, Executor ABI — no change without an Issue.

## PUBLIC / PRIVATE

- PUBLIC: `backend=auto|cpu|gpu` + reason; `set_backend` (rare, per-call preferred). PRIVATE: wgpu handles, pool stats, WGSL source (visible in trace only).

## WHAT MUST NEVER HAPPEN

- Silent fallback; a second stack; CUDA/Metal without an ADR plus measured need; bypassing frozen ABIs.
