# JS_RECON — numfast + roadgraph JS/WASM inventory (2026-09-26, read-only)

Scope: first-party JS/TS, wasm artifacts (*.wasm/*.wat/pkg/), package.json,
bundler configs, JS↔Rust/wasm bridge. Excluded: `.opencode/node_modules`,
venv `emscripten_fetch_worker.js`, zig vendored `main.js`. No build run.

## Table: компонент | JS-статус | WASM-статус | годность

| Компонент | JS-статус | WASM-статус | Годность |
|---|---|---|---|
| `parity_test.js` (root, 2026-08-26) — Python vs JS parity (Compute 31 alias, ExecutionPlan, WGSL SHA256) | EXISTS, refs dead path `numfast/src/math/Compute/Compute.js` + `COMPUTE_LIB`, path absent on disk | n/a | ABANDONED (broken ref) |
| `npm-placeholder/` `package.json` v0.0.2 + `index.js` (2026-08-11, AGPL-3.0, exports `{version, description}` only) | EXISTS, placeholder, no compute | n/a | ABANDONED (name-reserve only) |
| `numfast/numfast-native/tools/*.mjs` (~30 files, 2026-09-10..12): `wasm_map/cumsum/carry/multi/pack/pattern/select/shift/sorted/text/unique/mem/run`, `smoke_wasm/smoke_q1_wasm`, `r_block/r_colsink/r_h2o/r_rng`, `nfs_random/nfs_table`, `test_*`, `bench_r_h2o`, `dbg_text` — raw `readFile+WebAssembly.instantiate` harness | EXISTS, working harness, NOT npm lib | paired `tools/numfast_native.wasm` (2026-09-11, 114461 B, copy for harness) | WORKING harness only |
| `numfast/scratch/dump_h2o.mjs`, `dump_r_stream.mjs` | EXISTS, scratch | n/a | ABANDONED |
| `numfast/scratch/nf_rng_test.wasm` (97760 B) | n/a | EXISTS, one-off rng probe | ABANDONED |
| `numfast/numfast-native/target/wasm32-unknown-unknown/release/numfast_native.wasm` (2026-09-12, 120282 B) + `deps/`, `debug_text.wasm` | n/a | EXISTS, raw cdylib (`crate-type=["cdylib","rlib"]`, zero external deps, no wasm-bindgen) | WORKING artifact, raw-memory ABI only |
| `pkg/` (wasm-pack output: `.js/.d.ts/_bg.wasm`) in `numfast/`, `numfast-native/`, `roadgraph/` | NONE | NONE | ABSENT |
| `*.ts` first-party / `package.json` in `numfast/` or `roadgraph/` / bundler configs (`webpack/vite/rollup/tsconfig`) | NONE (verified `ls` misses) | n/a | ABSENT |
| Bridge: `wasm-bindgen`/`js-sys`/`web-sys`/`wgpu-web`/`navigator.gpu` in `Cargo.toml`, `src/*.rs`, `tools/*.mjs`, `parity_test.js` | NONE (grep empty) | n/a | ABSENT |
| `roadgraph/` first-party `*.js/*.mjs/*.ts/*.wasm/*.html` | NONE (`dir /s /b` empty) | NONE | ABSENT |
| Rust `nf_*` (~60 in `lib.rs`): `nf_sssp_batch`, `nf_sssp_csr`, `nf_sssp_csr_pred`, `nf_cost_intern`, `nf_cost_travel_batch`, `nf_rowwise_kway_time_argmin_gather` (+ group/join/sort/unique/text/carry/mask/select/pack/pattern/map/cumsum/shift) | NO JS wrappers for `sssp/cost/kway` (grep `sssp\|cost_travel\|cost_intern\|kway\|rowwise` over `tools/*.mjs+parity_test+placeholder` = empty). Covered in JS harness only: group/map/select/pack/pattern/cumsum/shift/sorted/text/unique/carry/mask | symbols exported from both native `.so/.rlib` and wasm module (raw `usize→i32` ABI) | GAP: sssp/cost/kway Rust-ready, JS-uncovered |

## Notes

- `parity_test.js` CHECK1/2/3 targets (`numfast/src/math/Compute/`) do not exist;
  live Python Compute is `numfast/src/Compute/Fused/` only.
- `Cargo.toml`: single-package workspace root, `opt-level=3, lto=false, panic=abort`.
- Lookup/solver JS: none found in either tree.
