# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# RUST RECONSTRUCTION MATRIX — second pass (branch `rebuilt`, 2026-09-05)

Principle: PASS parity ≠ optimal. Prior code = PROVEN CORRECT, not FINAL.
Scope: `numfast-native` only (8 C-symbols, C ABI + observable behaviour frozen
per `REUSE.md`). GPU untouched. SPEC v0.2 untouched. Python prod path intact
(`src/Drivers/CPU/_lib/cpu.py` NOT modified — pre-existing diff left alone).
1B never run (max 100M). Push never done (local commits only).

Toolchain (as-is): stable 1.98.1 + `x86_64-pc-windows-gnu` via zig-wrap
(`scratch/feas`), `wasm32-unknown-unknown` installed. Baseline flags
(`opt-level=3, lto=false, panic=abort`) unchanged in final state.
Box: Xeon E5-2698 v4 (Broadwell, AVX2, no AVX-512), 40 log. CPUs, 128GB.
Box noise: ±5–11% run-to-run on IDENTICAL dll (A/A self-test dense B/A=1.112) —
decision threshold set at consistent ≥10% across reps≥15 + min+median agreement.

## 1. Audit table (STEP 1 — every primitive)

| primitive | current | cost model (measured) | problem | alternative tried/considered | expected gain | risk | DECISION |
|---|---|---|---|---|---|---|---|
| dense_scatter | 1-pass scatter, safe idx, 1.2ms/1M (7.3x numpy) | 12MB/1.2ms ≈ 10GB/s; 2 RMW + 2 loads ≈ 2.6cy/row, L1-resident table (g=256) | none structural; collisions forbid SIMD; OOR trap contract forbids unchecked | — (analysis only) | ~0 | — | UNCHANGED |
| fused_scatter_soa | i-outer/c-inner, per-row range check, 3.1ms/1M/3c (5.2x) | 3 cols ≈ 2.6x dense-1col → linear scaling; L1-port bound (~3cy/row/col); addr math hidden | c*n+i multiply + len checks per inner iter | chunks_exact split staged, then REVERTED (dead code + fallback changed trap semantics) | <5% | behaviour change | UNCHANGED (backend-bound proof) |
| pack_codes | scalar i64 mul + 2 sign branches, 1.05ms/1M (1.35x numpy) | 12MB/1.05ms ≈ 11.4GB/s; 2.3cy/el, front-end tight | branches/ALU per element | AVX2 kernel + dispatch: ~100 lines for ≤30% (traffic ceiling ≥17GB/s, sorted proves) | ≤30% on 1ms-kernel = Amdahl-zero | high complexity | UNCHANGED (REJECTED w/ bandwidth proof) |
| pattern_scan_parse | per-row slice+prefix+digit loop, ~5.5ms/500k golden (140x py-loop) | 0.5GB/s — NOT bandwidth-bound, latency/branch-bound → headroom real | >10-digit bail faked overflow (VALUE-divergence vs prod, proven §2) | saturating → hybrid w>=18 → versioned + cold scan_long | neutral (±noise) | medium | REWRITTEN (correctness fix; speed neutral) |
| sorted_runs | 1-pass run scan, generic lane, 0.7ms/1M (16–18x numpy) | ~17GB/s ceiling; serial key-chain, 2 predictable branches/run | none — inherently serial | — | ~0 | — | UNCHANGED |
| buffer_compact (carry) | 1-pass fwd compact + branch, 1.7ms/1M@10% (3.5x numpy) | density sweep: 1%→0.74ms, 10%→1.8, 50%→5.0, 100%→4.1ms (mispredict peak @50%) | branch mispredicts at mid density | branchless always-store: flat ~4.1ms all densities → 5.5x SLOWER at 1% | wins only >85% density | low-density catastrophe | UNCHANGED (REJECTED w/ sweep) |
| FFI wrappers (8) | null-check + borrow, ~14.5µs/call incl. tiny allocs at n=1000 | ns-scale per call; 1 crossing per kernel op | none | — | 0 | — | UNCHANGED |
| fill_zero / atoms | generic zeroing, 1–2 store atoms | warm prologue, negligible at g≤1M | none | memset-equivalent already (LLVM folds) | 0 | — | UNCHANGED |
| hash atoms / facade | reserved, no consumers | — | — | — | — | — | UNCHANGED (surface not started, per REUSE) |

## 2. Rewrites (before/after/why/speedup/memory/correctness)

### REWRITTEN: pattern digit scan (`transform/pattern.rs` + doc-only `core/numeric.rs`)
- BEFORE: per-byte `w>=10` bail set `mag=MAX` → any 11+-digit body = BAD_RANGE.
- AFTER: loop versioned on `body.len()` — ≤18 digits plain Horner (exact, no
  wrap possible: max 10^18−1 < 2^63); 19+ digits → `#[cold] #[inline(never)]
  scan_long` (saturating i64). Overflow decided by VALUE vs `int_limit(neg)`.
- WHY (correctness, not speed): old code diverged from prod on
  `P00000000000000000001`-class inputs. Proven: prod Arrow path
  (`safe cast to int32`) → code=1/valid/width=20; old native → rc=-2 err_row=3;
  NEW native → identical to prod (codes/valid/width=20, adversarial 7–9-row
  corpora incl. boundaries ±2^31, garbage-after-long-digit, invalid-wins).
- Speed: neutral within box noise (golden 500k: B/A_med 0.94–1.17 across
  runs/methods; min-estimator 1.04–1.14 — unresolvable, kept for correctness).
- Memory: identical (no alloc, same buffers). Correctness: 6/6 golden parity
  PASS native+wasm; error contracts (null=-1, overflow=-2+err_row,
  malformed=-3) re-verified; rows-before-err written / err_row+ untouched.
- Micro-lessons recorded in code: `wrapping_sub` single-compare digit test
  REVERTED (measured ~2–3% slower than range form on this toolchain — LLVM
  folds range to sub+cmp anyway); per-byte `w>=18` branch versioned OUT of
  the loop; saturating block extracted `#[cold]` (no speed delta either way,
  kept for icache hygiene).

### OPTIMIZED: none (no kernel met the ≥10% consistent-win bar — honest zero)

### REJECTED (measured, with proof)
- `codegen-units=1`: B/A ≈ 1.0 everywhere (2 runs) → reverted.
- thin-LTO: speed ≈ 1.0 (3 runs, all kernels); size 1079KB → 359KB BUT only
  via nonstandard `cargo rustc -- -C embed-bitcode=yes -C lto=thin` (cargo 1.98
  emits NO `-C lto` for `lto="thin"` in Cargo.toml — verified in verbose
  rustc cmd; old "fat-LTO crashes zig" note left standing). No committable
  standard path → profile untouched (size is not a native constraint).
- `strip=true`: zero delta (zig link already `--strip-debug`) → reverted.
- Portable SIMD (`std::simd`): PROVEN nightly-only on 1.98.1 (E0658 probe) →
  no stable portable path; vendor-intrinsic dispatch rejected per-kernel
  (scatter/serial/memory-bound proofs above; pack capped ~30% on 1ms kernel).
- `target-cpu=native` shipment: rejected (R2) — all artifacts baseline x86-64.
- Fused pack+next (intermediate elimination): needs new C symbol; 8-symbol
  ABI frozen → rejected, recorded (host composition only possible path).
- Internal AoS accumulate for high-card dense: needs scratch ABI (core is
  no_std-compatible, no alloc) → rejected; degradation quantified instead
  (g-sweep §4).
- Branchless carry as default: 5.5x slower at 1% density → rejected.
- `wee_alloc`/nightly/`packed_simd`/rust-gpu: rejected per KNOWLEDGE_MAP R12-list.

### FALLBACK: none needed (native never worse than numpy on any kernel:
worst = pack 1.23x; wasm-pack 0.89x noted below, not a fallback case).

### UNKNOWN: fat-LTO link behaviour on current zig (not retried — thin sufficed
for the size question); exact L3 size/share on this VM (g-sweep infers only).

## 3. Pass verdicts
- CPU SIMD: evaluated → rejected (proofs §1–2). CPU DISPATCH: rejected (no
  kernel compute-bound enough to amortise; baseline ships everywhere).
- MEMORY (copy elimination/fusion/in-place): nothing to eliminate — zero-copy
  already (caller buffers, 1 crossing/op @ ~14.5µs incl. allocs at n=1000;
  no temp buffers exist to fuse; in-place N/A — outputs distinct).
- DATA LAYOUT: SoA outputs retained (ABI); high-card spill quantified
  (5M rows: g=256→7.0ms, 64K→42.6, 1M→78.7; numpy 52/89/121ms).
- MULTI-ALGORITHM: dense linear scaling proven (3c ≈ 2.6x 1c); sorted
  g-independent (11.2ms/10M); skew HELPS dense (zipf1.5 @g=64K: 42.1→10.7ms,
  hot keys pin L1). No Planner change (Python path frozen).
- NUMPY-INDEPENDENT PATH: resident→pack→group→carry needs only NumPy for
  factorize (keys); crossings = 1/op; crossing cost ~µs vs ms kernels.
- COLUMNAR RESULT: carry preferred path unchanged (dict never materialised
  in kernels); Q-carry 1M measured §1.
- GROUPBY strategies: dense (low-card) + sorted-run (pre-sorted) present;
  hash reserved-STOP. No H2O branches added.
- MULTITHREAD HOST (no Rust change — pure host composition, 8 symbols intact):
  contiguous partition + per-thread partials + ordered reduction; reentrant
  ✓ (no statics); ladder 10M: 1T 17.9 → 16T 7.07ms (2.54x, saturation ≈16T,
  32T regresses 2.32x); 100M: 1T 142.9 → 16T 36.5ms (3.91x, 33GB/s ≈ BW wall).
  counts bit-exact; sums 1-ULP deterministic (max|d| 6.9e-11 @100M, rel 2e-16;
  same tolerance class as accepted sorted-vs-reduceat 1.36e-12); bitwise
  run-to-run identical ✓. WEIGHTED OWNERSHIP: not integrated (compat kept).
- BLOCK METADATA/FILTER: layouts untouched (compat only).
- WASM SECOND PASS: rebuilt (25.3KB, +69B from pattern fix); 6/6 parity EXACT
  incl. error contracts; bench 1M: dense 5.9x, multi 2.8x, pack 0.89x (only
  sub-numpy kernel — traffic-bound wasm store path, documented, no fallback:
  native remains the perf path), pattern 133x, sorted 5.6x, carry 3.35x.
  Same algorithm both targets ✓; SIMD-in-wasm N/A (no SIMD in kernels);
  boundary (n < 2^31) unchanged; size gate holds (25KB ≪ budget).
- BINARY SIZE: DLL 1079KB (std statically linked, gnu target), wasm 25.3KB.
  No `no_std`/opt-z/codegen-units changes (all measured ≈0). Performance not
  sacrificed for size anywhere.
- CROSS-PLATFORM: single source, baseline x86-64 + wasm32 from same code;
  no target-cpu=native artifacts; dispatch N/A (rejected).
- TOOLCHAIN/ABI: stable C ABI, 8 symbols, `borrow` single unsafe point;
  no PyO3/maturin; ctypes contract unchanged.
- BENCHMARK PROTOCOL: correctness→(15-rep interleave, med+min)→10M→100M(MT);
  wall ms + RSS (35→684MB @10M harness) + crossings + cold/warm (5 warm) +
  integrity-first (exact-STOP) followed throughout.
- COMPETITORS (10M query-only, prep separate): native-1T 13.4ms = polars
  13.5ms (1.00x; prep 4.5ms); duckdb query 164.5ms (12.2x slower as measured,
  prep 469ms — likely unoptimized plan on this box, recorded as-is);
  MT-16T native (7.1ms) ≈ 1.9x polars. fp-order diffs: polars 6.5e-12,
  duckdb 0.0.
- SKEW: uniform/mild(zipf helps 3.9x, §3) covered at native level; full
  Q1–Q5 skew harness is Python-path work (out of scope, prod path frozen).
- 1B: NOT run (only estimate: dense-16T ≈ 365ms + RSS ~12GB by linearity).

## 4. Benchmarks (final artifacts, seed 42)
1M medians — dense 6.73x np / wasm 5.93x; multi(3c) 4.85x / 2.78x;
pack 1.23x / 0.89x; pattern 140.8x py / 132.8x; sorted 15.84x / 5.60x;
carry(10%) 3.50x / 3.35x. 10M: dense np 104.7→rust 14.1ms (7.4x);
multi 39.3ms; pack np 26.7→12.5 (2.1x); sorted 11.2ms. 100M MT §3.
New tools (all seed-42, integrity-first): ab_compare.py, pat_ab.py,
carry_skew.py, bench_10M.py, bench_mt.py, bench_gcard.py, bench_compete.py.

## 5. REUSE check after rewrite
New code (`scan_long`) is the cold 19+-digit path — physical reuse: shares
`is_digit_byte` atom + `int_limit` tail with the hot loop; no mega-helper
(one purpose: rare-path scan); hot loops stay flat. `is_digit_byte` keeps its
consumer (deliberately NOT inlined — REUSE atom preserved). Duplication
re-check: no second copies introduced (fused experiment fully reverted).

## 6. Skills verdict (honest, with examples)
- ponytail: HELPED most. Killed the fused-chunk experiment (dead code +
  trap-semantics change — reverted same session), stopped the pattern speed
  chase at the noise floor (4 build cycles → verdict NEUTRAL, kept for
  correctness), rejected LTO/strip churn and pack-SIMD (Amdahl-zero on a 1ms
  kernel). HARMLESS nowhere; but mission overrode it once: exhaustive
  evaluate-everything (SIMD/dispatch/layout) instead of ponytail's skip.
- python-expert: HELPED. ctypes harness correctness (argtypes/restype per
  symbol incl. u8/i64 variants), GIL-release-aware MT design (I/O-free C
  calls parallelise), contiguous partition + preallocated partials, RSS via
  psutil. No conflicts.
- numfast-performance: HELPED (methodology backbone: seed 42, integrity-first
  STOP, OLD→NEW/RATIO, new-file discipline, dtype discipline in harnesses).
  IGNORED once: its golden-first framing — the pattern divergence was found
  only by ADVERSARIAL probing past the golden corpus (width≤10 never covered
  >10-digit bodies). Lesson recorded: golden ⊆ contract; probe the edges.
  Its ">1000 ops → GPU" rule N/A (GPU explicitly out of mission).
- builder: HELPED structurally (modular bench tools, no feature scripts, no
  _lib violations, REUSE atom discipline). Formal Extension machinery N/A to
  a Rust crate — applied as spirit.
- javascript-typescript-expert: NEUTRAL. No JS/WGSL work in mission; existing
  Node harness reused for wasm parity only. No influence on decisions.
- RUST_WGPU_KNOWLEDGE_MAP (base): HELPED decisively. R1 → SIMD probe
  (nightly-only proven, killed portable-SIMD hope in 5 min); R2 → no
  target-cpu=native artifacts + dispatch rejection framing; R3 → 8 symbols +
  borrow-point untouched, `-2` semantics table honoured; R5 → LTO/strip/size
  experiments; R6 → bench discipline; R8 → wasm 25KB gate held; REJECTED-list
  enforced (no rust-gpu/wee_alloc/packed_simd/nightly). GPU sections R10–R14
  correctly N/A. Nowhere ignored; one correction to it: "LTO crashes zig"
  is stale for THIN lto (links fine) — but still rejected (nonstandard flags,
  zero speed gain).

## 7. Commits (local ONLY, no push anywhere)
- (this session) `recon: pattern value-semantics + audit harnesses + results`
  files: src/transform/pattern.rs, src/core/numeric.rs (comment), 7 new
  tools/*.py, results/*.json refreshed, this matrix file.
- Push: NOT performed (forbidden). Prod Python path: unmodified.
- Temp: scratch/ab_*.dll + scratch probe files stay in untracked scratch/
  (not committed); no temp files inside numfast-native/.

STOP.
