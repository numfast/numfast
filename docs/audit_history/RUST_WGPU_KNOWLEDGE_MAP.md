# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# RUST + WGPU KNOWLEDGE MAP — NumFast (branch `rebuilt`, 2026-09-05)

> Read-only mission. No production code changed. One file only.
> Scope: Rust CPU (portable SIMD, FFI/ABI, profiling, builds, WASM) + GPU (wgpu/WGSL compute).
> Anchor: `Python API → IR → Planner → Runtime → Native ABI → (Rust CPU | Rust WASM → Node/Browser)`.
> Frozen: 8 C-symbols in `numfast-native` (`nf_group_sum_count`, `nf_group_multi_sum_count`,
> `nf_pack_i32_direct`, `nf_pattern_encode`, `nf_sorted_run_i64/f64`, `nf_carry_build_i64/f64`),
> C ABI + observable behaviour frozen per `numfast-native/REUSE.md`. GPU part not started.
> Rule: wgpu accepts WGSL natively — rust-gpu is NOT required.

## 0. Version status (verified 2026-09-05, all via web)

| Component | Stable | Notes / change |
|---|---|---|
| Rust stable | **1.98.1** (2026-09-03; 1.98.0 2026-08-20) | Fix: vtable null-ptr miscompilation from 1.98.0. `rustup update stable`. |
| wgpu crate | **30.0.0** (2026-07-01), 29.0.4 same day | MSRV policy: `wgpu` lib **1.87**, workspace/test **1.93**, never > `stable-3`. Backends: Vulkan/Metal/DX12/GLES native + WebGPU on wasm. Core of Firefox/Servo/Deno WebGPU. |
| wasm-bindgen | **0.2.126** (2026-06-24) | MSRV lib 1.77 / CLI 1.86. Import namespace changed `__wbindgen_placeholder__` → `./{name}_bg.js` for node/deno/module (cross-target sharing). Threads need explicit `-Clink-arg=--export=__heap_base` on nightly ≥2026-05-06. `-Cpanic=unwind` now emits modern exnref EH by default → needs Node ≥22.22.3 (`WebAssembly.JSTag`); legacy via `-Cllvm-args=-wasm-use-legacy-eh`. |
| wasm-pack | 0.13–0.14 line (0.14 restores npm binary, wasm64 target, `--panic-unwind`, `--no-opt`, arbitrary `--target`) | Orchestrates cargo→wasm-bindgen→wasm-opt→pkg/. Pin crate==CLI version, commit Cargo.lock. |
| WGSL spec | **Candidate Recommendation 2026-03-10** | Sharpens subgroups, barriers, buffer layout. Still Working Draft family; wgpu/Naga may lag spec. |
| Browser WebGPU | **Shipped in all 4 majors (late 2025)** | Chrome/Edge 113+ (Dawn; Linux Intel Gen12+ @144, Android 121+), Firefox 141 (Windows) / 145 (macOS Apple Silicon+Tahoe 26), Safari 26 (macOS/iOS/visionOS 26). Linux-FFx/Android-FFx still behind flag, in progress 2026. `navigator.gpu`, secure context only. |
| Node.js | **24.x Active LTS (Krypton, to 2028-04-30), 26.x Current, 22.x Maintenance** | WASM SIMD since Node 16.4; threads via `wasm32-wasip1-threads`/workers + SharedArrayBuffer (needs COOP/COEP in browser). Modern exnref EH needs Node 22.22.3+. riscv64: known SIGILL issues, workaround `--no-wasm-lazy-compilation` (platform-specific, not x64/ARM). |
| WASM targets | `wasm32-unknown-unknown` (web), `wasm32-wasip1/2`, new `wasm32-wasip3`/`wasm32-component`/`wasm32-component-web` (2026 goals, tier-2/3) | `wasm32-wasi` renamed → `wasip1`. WASI 0.3 async/component-model in progress. `std::thread` on wasip3 planned. |
| Naga | via wgpu (translates WGSL↔SPIR-V↔GLSL/MSL/HLSL) | Use Naga for translation, not rust-gpu. |

## 1. Topic → Source/Skill → NumFast rule

### R1. Portable SIMD without per-CPU lock-in
- Sources: `doc.rust-lang.org/stable/std/simd/` (portable SIMD module); `rust-lang/portable-simd` beginners-guide + repo; Caleb Zulawski rust-simd-book ch.2; `rust-lang.github.io/packed_simd/perf-guide`; 2026-06 Boardor SIMD guide (std::simd vs wide vs pulp/macerator/fearless_simd); 2026-03 Wren SIMD; 2026-04 DEV `std::simd` audio/image/dot-product examples.
- Applicable rule: kernels stay flat/specialized per `REUSE.md` (no N-param mega-helper); lane types i32 logical / i64 accumulators; accumulation order bit-exact (sequential `i=0..n`) — SIMD must not reorder float adds. Portable `Simd<T,N>` compiles everywhere (scalar fallback where unsupported); vendor intrinsics only via explicit opt-in path.
- Take: default = `std::simd`-style portable abstraction (or `pulp`/`macerator` on stable for multiversion); `wide` only if no multiversion needed.

### R2. Runtime CPU feature dispatch (x86 pain point)
- Sources: portable-simd beginners-guide "Target Features" (default x86_64 = SSE/SSE2 only; enabling AVX in RUSTFLAGS = UB on older CPUs); Boardor multiversion section (`pulp::Arch::dispatch`, `multiversion` crate); `std::is_x86_feature_detected!` + `#[target_feature(enable)]` pattern.
- Applicable rule: one binary must run on SSE2 baseline and use AVX2/AVX-512/NEON where present — no `-Ctarget-cpu=native` shipping binary. Native ABI stays single entry point; dispatch inside.
- Take: baseline build + `is_x86_feature_detected` gate (or pulp-style dispatch). Never ship `RUSTFLAGS=-Ctarget-cpu=native` artifacts as public binaries.

### R3. Zero-copy FFI / C ABI stability
- Sources: Rustonomicon FFI; Reference `items/external-blocks` + `abi` (`#[unsafe(no_mangle)]`, `extern "C"`/`"C-unwind"`); Effective Rust Item 34; Microsoft Pragmatic Rust Guidelines (M-ISOLATE-DLL-STATE, M-FFI-TRANSLATES).
- Applicable rule: 8 symbols frozen; `#[repr(C)]` structs; sized ints; `*const/*mut` + len, never `&T/Box/Vec` across boundary; allocate+free same side (`Box::into_raw`/`from_raw` pairs); `catch_unwind` at boundary, never let panic cross `extern "C"` (abort or error code per `errors.rs`); prefix `nf_`; business logic in core crate, `-ffi` glue only translates; no `static`/TypeId leakage between DLLs.
- Take: existing `core::buffers::borrow/borrow_mut` single unsafe wrap point is the correct pattern — keep. New GPU surface must reuse same contract style.

### R4. `unsafe` safety discipline
- Sources: Nomicon (raw ptr, aliasing, null-Never types → `Option<NonNull>`/`as_ref` null check); Effective Rust (RAII Drop wrapper, heap not stack across FFI); Reference (extern blocks `unsafe`, `safe` qualifier only when provable).
- Applicable rule: `unsafe` isolated in `core::buffers`/`checks`; safe wrappers for all 8 FFI fns; OOR contract: `dense_scatter` intentionally unchecked (safe-Rust trap) vs `fused` `-2`-checked — preserve exactly.
- Take: any new kernel copies this split; audit list = `borrow` sites only.

### R5. Minimal native binary / release profile
- Sources: `numfast-native/Cargo.toml` (`opt-level=3, lto=false, panic=abort`; comment: lto crashes zig 0.16 COFF/LLD here); RustTraining ch.3 + OneUptime profiling guides (LTO biggest cheap win; PGO after).
- Applicable rule: keep `cdylib+rlib`, frozen symbols exported; revisit `lto=true` only with real toolchain; `panic=abort` (no unwind across C ABI).
- Take: profile before toggling; record size + `cargo bench` delta on any profile change.

### R6. Profiling / benchmarking (CPU)
- Sources: Microsoft RustTraining ch.3 (Criterion vs Divan vs hyperfine vs perf+flamegraph + PGO + CI); OneUptime 2026-02-01/2026-01-07; codehowtoguide 2026-04-07; flamegraph-rs docs; criterion 0.5/0.8 patterns (`black_box`, groups, `bench_with_input`, html_reports).
- Applicable rule: fixed seed 42; stage breakdown ms/stage; integrity check before bench (loss >1% = STOP); new measures = new files.
- Take: Criterion (stats + regression) for micro, hyperfine for binary, `perf record --call-graph=dwarf` + `cargo flamegraph` for hotspots, keep `debug=true` in bench profile. No `Instant::now()` hand-rolls as evidence.

### R7. Cross-platform builds (native)
- Sources: wgpu MSRV policy (1.87 lib / 1.93 repo, stable-3 rule); Rust release notes; `cargo build --release` + `rust-toolchain.toml` pinning practice.
- Applicable rule: MSRV ≥ wgpu floor if wgpu dep added; Windows via wgpu-py path today (Vulkan/DX12), no CUDA; no VS/mingw-hack binaries as releases.
- Take: pin toolchain + lockfile; CI matrix win/linux/mac; document MSRV bump as breaking.

### R8. WASM builds + binary size
- Sources: rustwasm book "Shrinking .wasm" (`lto`, `opt-level=z/s`, `wasm-opt -Oz`, strip, `codegen-units=1`, dlmalloc ~10–20KB, `wee_alloc` legacy); wasm-bindgen "Small Wasm files"; Wasm Docs wasm-pack guide (metadata `wasm-opt=["-Oz","--enable-bulk-memory"]`, precedence replaces defaults; `twiggy top`, `wasm-validate` in CI; size table 96KB→41KB example); wasm-bindgen #4591 (measure post-`wasm-bindgen`, not raw cargo output).
- Applicable rule: `Rust WASM → Node/Browser` path; keep compute core GC-free, narrow JS boundary (typed arrays, no per-row marshaling); deterministic numerics same as native.
- Take: release profile `opt-level=z, lto=true, codegen-units=1, panic=abort, strip=true` + `wasm-opt -Oz`; budget gate in CI (`wc -c` check); `twiggy` for surprises; keep dlmalloc default (wee_alloc unmaintained); pin wasm-bindgen==CLI.

### R9. wgpu Rust host API (native + wasm target)
- Sources: `crates.io/wgpu` (30.0.0), `wgpu.rs` docs, `github.com/gfx-rs/wgpu` README/CHANGELOG (features `vulkan/metal/dx12/gles/webgpu`, env `WGPU_BACKEND/ADAPTER_NAME/DX12_COMPILER`); Learn Wgpu + WebGPU Fundamentals + wiki; rust-gpu blog 2025-07-25 (multi-backend via wgpu+naga).
- Applicable rule: GPU = WebGPU API in Rust; one kernel source (WGSL) runs native (Vulkan/Metal/DX12) and browser (WebGPU backend) — matches "portable kernels without native perf loss".
- Take: depend on `wgpu` (not rust-gpu); request adapter once, cache pipelines, pool buffers; read `adapter.limits` and clamp dispatch.

### R10. WGSL compute optimization
- Sources: WGSL CR 2026-03-10 summary (techbytes.app); youngju.dev hands-on (two-stage reduction, `var<workgroup>`, barriers); mysimulator.uk compute; teachme.sh workgroup org; spatialvisualization.org (opt flags, occupancy, geometry filtering); cazala/webgpu-skill `references/compute-patterns.md`.
- Applicable rule: numeric determinism (bit-exact float order) constrains tree-reduction shape; int32/float32 lanes; no float64 series data.
- Take:
  - `@workgroup_size` product multiple of 32/64; portable default **64 (1D) / 8×8 (2D) / 256 max**; never exceed `maxComputeInvocationsPerWorkgroup` (floor 256); Z ≤64.
  - `var<storage,read>` inputs, `read_write` only for outputs; `var<workgroup>` tile + `workgroupBarrier()` (uniform control flow only); two-stage reduction (per-workgroup partials → stage B/host).
  - Bound guard `if (gid.x >= n) return` *before* barrier-sensitive code; hierarchical atomics (local reduce → 1 global `atomicAdd` per workgroup); `dispatchWorkgroupsIndirect` for data-driven counts.
  - Buffer layout = part of API: same field order/align/stride both sides, explicit padding, `@align(16)` for vec4; roles explicit (ro input / rw scratch / wo output).

### R11. GPU memory layout / limits
- Sources: same as R10 + `GPUSupportedLimits` tables (`maxBufferSize` 256MiB, `maxStorageBufferBindingSize` 128MiB, `maxComputeWorkgroupStorageSize` 16KiB floor, per-axis caps).
- Applicable rule: Storage = traversal delta for ALL rows (dzst); chunk dispatches when exceeding binding size.
- Take: budget ~24 B/row filter set example; request limits above defaults; chunk oversize datasets.

### R12. Workgroup sizing / occupancy
- Sources: spatialvisualization occupancy guide (`min(Wmax, M/m_wg, R/(r·|wg|))`), teachme.sh, compute-patterns.md.
- Applicable rule: profile every execution; sweep sizes with timestamp queries; stop at knee.
- Take: start 64/256, tune per device class; smaller tile at high occupancy beats max tile; WGSL `override WORKGROUP_SIZE` for per-device specialization; never assume subgroup size (gate subgroup ops as optional path).

### R13. Synchronization
- Sources: WGSL CR barriers; compute-patterns "uniform control flow"; youngju/mysimulator barrier examples.
- Applicable rule: same abort/overflow contracts on GPU as CPU.
- Take: `workgroupBarrier` only (never cross-workgroup); pass boundaries for global deps; mask/neutral-value before barrier, guard writes after; no barrier around pure-register work.

### R14. GPU profiling
- Sources: timestamp-query sweep pattern (occupancy guides); youngju checklist (warm-up, measure from call ≥5, cache pipelines, pool buffers, batch `mapAsync`); Dawn/WebGPU timestamp queries.
- Applicable rule: stage breakdown + fixed seed + integrity gate apply to GPU passes too.
- Take: `timestamp` query sets for dispatch sweep; avoid GPU↔CPU ping-pong (stay on GPU, bind outputs as next-pass inputs); `onSubmittedWorkDone` over polling; no per-call device/adapter/shader recompile.

### R15. Cross-platform WebGPU matrix (ship checklist)
- Sources: gpuweb Implementation-Status; web.dev 2025-11-25; caniuse/webstatus; Khronos 2026 slides (compat mode, `webgpu.h` stable 2025-09).
- Applicable rule: Windows-first (wgpu-py/Vulkan/DX12), browser second.
- Take: guard `if (!navigator.gpu)` + WebGL2 fallback; test Chrome/Edge + FFx-Win + Safari-26 + Chrome-Android; Linux-FFx/Android-FFx = experimental flag path; compat-mode subset for wide reach.

## 2. Ready-made skills/guides — verdict

| Candidate | Verdict | Reason |
|---|---|---|
| Local `.opencode/skills/*` (builder, caveman, numfast-performance, ponytail, python-expert, js-ts-expert) | PARTIAL — reuse, not enough | No Rust/WGPU coverage. `numfast-performance` (dtype/stage-bench) + `javascript-typescript-expert` (TypedArrays/WebGPU-on-web) partially apply. |
| `cazala/webgpu-skill` (`references/compute-patterns.md`) | TAKE as GPU pattern ref | Current, portable-focused (limits clamp, ping-pong, phase composition, uniform barriers). Maps to R10–R14. Not Rust-specific — pair with wgpu docs. |
| Official docs (Rust std::simd, Nomicon FFI, Reference ABI, wgpu.rs/docs.rs, rustwasm book, wasm-bindgen guide, GPUWeb wiki, MDN WebGPU) | TAKE as primary skill content | Most current (2026 versions above). This file distills them; no third-party "Rust agent skill" found that is newer or more authoritative. |
| Generic old tutorials / random blogs pre-2024 | REJECTED (stale) | API drift (wgpu major/minor, WGSL drafts, wasm-bindgen namespace/EH change). Use only version-pinned official docs. |
| `packed_simd` crate guides | REJECTED for new code | Legacy (`packed_simd` retired in favour of `std::simd`/portable-simd). Keep as perf-history ref only. |
| `wee_alloc` as default allocator | REJECTED as default | Unmaintained, single-threaded, fragments. Keep dlmalloc; use only on tiny alloc-light modules with benchmark proof. |
| `rust-gpu` (rustc_codegen_spirv) as required GPU path | **REJECTED as mandatory** (per task) | Nightly-pinned, SPIR-V-only backend, no compat guarantee, u64/u128 gaps, rough DX. Correct use: optional experiment; ship path = WGSL via wgpu+Naga (bijective SPIR-V↔WGSL where needed). |
| CUDA-only / PTX / `rust-cuda` as primary | REJECTED | Windows target is Vulkan/DX12 via wgpu; CUDA breaks portability + WASM story. |
| `-Ctarget-cpu=native` shipping builds | REJECTED | Instant UB on older CPUs. Use baseline + runtime dispatch (R2). |
| `RUSTFLAGS` with unpinned nightly for stable code | REJECTED | Breaks MSRV + reproducibility. Pin toolchain. |
| Per-row JS↔WASM marshaling / `mapAsync` per iteration / device-per-call | REJECTED (perf) | Stalls pipeline; stay on GPU, pool, batch (R14). |
| `workgroup_size` 1024+, non-multiples of 32, barriers in divergent branches | REJECTED (correctness/perf) | Fails mobile validation / hangs / wastes lanes (R10–R13). |
| f16 without feature guard, single giant shader, float-reordering "optimizations" | REJECTED | Breaks portability + bit-exactness. |

No single purchasable/downloadable "Rust+WGPU agent skill" newer than the official docs was found; this map **is** the local skill content (distilled, version-pinned, rule-mapped).

## 3. Deterministic numerics (cross-cutting)

- Float accumulation order fixed (`i=0..n` sequential on CPU; staged reduction with documented order on GPU) — SIMD/workgroup tree must reproduce the same order or document deviation + parity test.
- int32 logical, int64 accumulators, float64 display-only. No float64 series, no `.to_numpy()` materialization in hot path.
- Parity: scalar reference → SIMD/GPU vs scalar on thousands of random inputs with tolerance; `REUSE.md` abort/overflow codes preserved.

## 4. Pre-implementation checklist (next mission reads this)

1. Pin `rust-toolchain.toml` (≥1.87 for wgpu; recommend 1.93+/stable 1.98.1) + `wasm32-unknown-unknown` target + wasm-bindgen==CLI + wasm-pack.
2. CPU: baseline SIMD + `is_x86_feature_detected`/pulp dispatch; Criterion benches (seed 42) + flamegraph; keep 8 symbols frozen.
3. WASM: `wasm-pack --target web/bundler/nodejs` per consumer; `wasm-opt -Oz`; size gate + `twiggy` + `wasm-validate`; Node 24 LTS test, browser matrix (R15).
4. GPU: WGSL kernels via wgpu; limits clamp; 64/256 start size; two-stage reductions; timestamp sweep; no rust-gpu dependency.
5. REJECTED list enforced in review (Guardian role).

---
*Sources fetched 2026-09-05 via websearch: crates.io wgpu/wasm-bindgen, gfx-rs/wgpu repo+CHANGELOG, Rust blog 1.98.x + releases, wasm-bindgen CHANGELOG/issue #4591, rustwasm size book, Wasm Docs wasm-pack guides, Boardor/Wren/DEV SIMD 2025–26, portable-simd book/repo, Nomicon/Reference/Effective-Rust/Microsoft FFI+bench guides, youngju/mysimulator/teachme/spatialvisualization/cazala WGSL-compute guides, techbytes WGSL-CR, gpuweb status, web.dev, caniuse/webstatus, Khronos slides, nodejs/release + docs. Local grounding: `numfast-native/REUSE.md`, `Cargo.toml`, `src/` layout on branch `rebuilt`.*
