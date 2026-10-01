# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 04 — Runtime + Execution (normative)

## PURPOSE

`ExecutionGraph → Packets/Waves → Driver` with resident intermediates.

## INPUT / OUTPUT

- INPUT: enriched `ExecutionGraph` + `kernel_table` + `wgsl_serializer`.
- OUTPUT: `ExecutionPacket[] {dispatch(dx,dy,dz), buffers, uniforms, bindings}` → `Driver.execute/execute_fused/execute_wave` → final readback.

## OWNER / DEPENDENCIES

- OWNER: Runtime (`builder.py`, `execution_scheduler.py`). DEPENDENCIES: `03`, `06`.

## INVARIANTS

- `execute_fused` with `len(packets)>1` keeps intermediates in `GpuBufferPool`; `extract_column` on intermediates is forbidden.
- The dispatch limit belongs to a SPECIFIC BACKEND (hardware/ABI), not to the math model: the backend publishes `backend.max_dispatch_{x,y,z}` + `max_buffer_bytes` in its capability; the Runtime must read the limit from there. `_check_dispatch_limit→ValueError` on overflow after splitting.
- The Runtime must split an operation when its semantics allow it: every operation declares `operation.chunkable: bool` (+ `chunk_constraints`, e.g. `associative_only | elementwise | window_overlap`). `chunkable=true` → the Runtime must slice into `num_chunks` so that each chunk is `≤backend_limit` and `peak<full`. `chunkable=false` → `num_chunks=1` only, or an observable CPU fallback in `execution_info`.
- Current `chunkable` assignments (operation property, revisable — not a permanent law): `Map/MapBinary/Filter-Compare/Fill/Reduce-associative=true`; `Scan/MatMul/Sort/GroupBy=false`. Changing `false→true` requires an ADR with a semantics-preservation proof, not a silent code edit. The change must include a chunked-vs-unchunked semantic-equivalence test.
- `EXPLAIN` without exec; `ANALYZE/TRACE` with exec: semantic→optimized→fusion→representation→backend→passes→memory→transfers→timing cold/warm.
- Calibration: `nf.calibrate()/quick/force` → TOML (`project`, `~/.cache`, `session`), JS IndexedDB; `EXPLAIN` shows `profile_version/source/age/matched/warning`.

## PUBLIC / PRIVATE

- PUBLIC: `evaluate/compute(backend=auto|cpu|gpu)`, `execution_info`, `explain/trace/compile`, `calibrate/calibrate_info/device_info/profile`. PRIVATE: pool lifecycle, serializer, packet layout.

## WHAT MUST NEVER HAPPEN

- A user-level `if N>limit` loop; a second Builder/Scheduler; materialization of intermediates on CPU.
