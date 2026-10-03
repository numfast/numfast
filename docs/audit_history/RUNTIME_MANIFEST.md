# NumFast Native Runtime Manifest — 2026-09-21

Consumer build (incl. future Agent 1). No push/PR. Old artifacts kept
(`*.bak-*` next to each DLL).

## 1. Versions

- `numfast` Python package: **0.2.1** (`src/numfast/__init__.py: __version__`)
- `numfast-native` Rust crate: **0.1.0** (`numfast-native/Cargo.toml`)
- Toolchain: rustc/cargo **1.98.1**, target `x86_64-pc-windows-gnu`
  (linker shim `numfast-native/tools/nf-link.bat`), profile `release`
  (`opt-level=3`, `lto=false`, `panic=abort`)

## 2. DLL

Build command (from `numfast/numfast-native/`):

```text
cargo build --release --target x86_64-pc-windows-gnu
```

| slot | path (repo-relative to `numfast/`) | sha256 | bytes |
|---|---|---|---|
| primary (`_DLL_DEFAULT` in `src/Drivers/CPU/_lib/native_cpu.py`) | `numfast-native/target/x86_64-pc-windows-gnu/release/numfast_native.dll` | `3db09442b832f8e3540516e87d18713c504ffda1edccfae2d176072e05d3705c` | 1236992 |
| packaged (wins via `ensure_native_env`) | `src/numfast/_native/numfast_native.dll` | `e539461f58ab9c1225accd5943260467ab0e3760f523e73560ba43d89280d812` (refreshed 2026-09-26 CORE-NEWPRIM: adds `nf_pair_insert_i64` + `pair_insert` module) | 82 symbols |
| previous packaged | *(removed 2026-10-04)* | was `src/numfast/_native/numfast_native.dll.bak-20260921`, sha256 `614f323bc7fa2944…`, 1344725 bytes | 1344725 |

Previous packaged copy (2026-09-17) predated `groupby/hash_grouped.rs`
changes (2026-09-21) and was refreshed from this build.

**2026-10-04 — the eight `.bak-*` copies of the native DLL were deleted from the
tree** (9.6 MB). Nothing loaded them: `ensure_native_env` resolves only the
exact names `numfast_native.dll` / `numfast_native.so`, `pyproject.toml` ships
`_native/*.dll` (which the `.bak-*` names do not match), and no code or test
referenced any of them. Two of the eight were byte-identical
(`.bak-pre-rowmin` and `.bak-pre-rowmin-agent2`, sha256 `5a0b…`). This table
records the sha256 of the removed copies so the history stays checkable; the
binaries themselves are recoverable from git history at any commit before
`2026-10-04`.

Resolution order (`src/numfast/_lib/native_env.py::ensure_native_env`):

1. `NUMFAST_NATIVE_DISABLE=1` → no native, pure fallback.
2. `NUMFAST_NATIVE_DLL` env → that file (if set).
3. packaged `src/numfast/_native/numfast_native.dll` (else `.so`).
4. gnu-target `numfast-native/target/x86_64-pc-windows-gnu/release/numfast_native.dll`.

## 3. Kernel list (78 FFI symbols, all verified present via ctypes)

- groupby dense/fused: `nf_group_sum_count`, `nf_group_multi_sum_count`,
  `nf_group_mixed_sum_only`, `nf_group_count_only`,
  `nf_group_owner_2i32_1f64`, `nf_group_variant_f64`,
  `nf_group_variant_i64`, `nf_group_variant_i32`
- hash_grouped: `nf_ghash_count_i64`, `nf_ghash_fp_i64`,
  `nf_ghash_occ_count`, `nf_ghash_occ_fill`, `nf_ghash_pins_i64`,
  `nf_ghash_scatter_i64`
- sorted/carry: `nf_sorted_run_i64`, `nf_sorted_run_f64`,
  `nf_carry_build_i64`, `nf_carry_build_f64`
- sort: `nf_sort_perm_i32`, `nf_sort_perm_i64`
- unique: `nf_unique_inverse_i32`, `nf_unique_inverse_i64`,
  `nf_unique_dict_utf8`
- pack: `nf_pack_i32_direct`, `nf_pack_sum_count_i32`,
  `nf_pack_sum_count_f64`
- select/mask: `nf_select_count`, `nf_select_scatter_i32/i64/f32/f64/u8`,
  `nf_mask_and`, `nf_mask_or`, `nf_mask_not`
- shift: `nf_shift_i32/f32/f64`
- cumsum: `nf_cumsum_i32/f32/f64`
- map (9): `nf_map_i32`, `nf_map_scalar_i32`, `nf_map_fscalar_i32`,
  `nf_map_f32`, `nf_map_scalar_f32`, `nf_map_f32_divpow`,
  `nf_map_scalar_f32_divpow`, `nf_map_f64`, `nf_map_scalar_f64`
- join: `nf_join_build`, `nf_join_probe`, `nf_join_gather_i32`,
  `nf_join_fused_left_i32`, `nf_join_fused_inner_i32`
- pattern/text: `nf_pattern_encode`, `nf_text_contains`, `nf_text_endswith`,
  `nf_text_equals`, `nf_text_length`, `nf_text_startswith`
- segmented: `nf_segment_count`, `nf_segment_reduce_f32/i32`
- rowwise: `nf_rowwise_min4_argmin_gather`, `nf_rowwise_min4_time_argmin_gather`
  (argmin over D-lanes / over T-lanes respectively, strict `<`, tie = smallest index;
  parity: `numfast/numfast-native/results/parity_rowmin_time.json`;
  bench: `numfast/numfast-native/results/bench_rowmin_time.json`)
- rowwise K-way (canonical, generic K 1..=256): `nf_rowwise_kway_time_argmin_gather`
  (strict `<` from lane 0, ties keep smallest index; float32 bit-exact gather;
  parity: `numfast/numfast-native/results/parity_rowmin_kway.json` (1025 cases, 1000 fuzz + edges);
  bench: `numfast/numfast-native/results/bench_rowmin_kway.json` (50k-5M vs K4, 27-50% faster))
- series/misc: `nf_adjacency_gather`, `nf_adjacency_slice`,
  `nf_cost_intern`, `nf_cost_travel_batch`
- rng: `nf_rng_compat_runif`, `nf_rng_compat_sample`, `nf_rng_fill_f64`,
  `nf_rng_fill_i32`, `nf_rng_map_round`, `nf_rng_permutation`,
  `nf_rng_sample_no_replace`

## 3b. SSSP (canonical, generic CSR — 2026-09-22)

- symbols: `nf_sssp_csr`, `nf_sssp_csr_pred`, `nf_sssp_batch`
  (81 symbols total; frozen ABI above untouched, additive only)
- contract: `indptr[V+1]` u32 monotone + `indices[E]` u32 + `weights[E]`
  u32 + source(s) -> `dist` u32 (`UINT32_MAX` = unreachable/reserved).
  INF weights skipped, never relaxed; saturating add (overflow lands
  on INF, never wins); heap order `(dist, vertex)` (tie = smallest id,
  heapq parity); full run to exhaustion (no target set, no early
  exit, unlike `nf_router_route` i64 multi-source early-exit)
- `nf_sssp_csr`: single source -> `dist[V]`; `nf_sssp_csr_pred`: plus
  `pred[V]` i32 (`-1` = root/unreached); `nf_sssp_batch`: `sources[K]`
  -> `out[K*V]` row-major in source order (thread-count invariant,
  `nthreads` clamped `[1, 64]`, scoped threads per source chunk —
  same budget as the join probe; `K == 0` returns 0)
- batch layout decision: full `KxV` matrix (targets resolved
  caller-side; no compact per-target mode on this surface)
- codes: 0 ok, -1 null, -2 bad range (reserved lane, bad source,
  non-monotone, empty graph), -3 bad geometry (`indptr[V] != E`,
  lane outside `[0, E]`, short output)
- Python: `src/Relational/Sssp/` Extension (`Sssp.toml` alias/mods
  `sssp_csr/sssp_batch`, `kernel.metadata["Sssp"]`, `full.toml`
  wired); `_lib/sssp.py` native-first + bit-exact heapq fallback
  (same order/saturating-add/INF guard); generic lanes only, no GPU
- parity: `numfast-native/results/parity_sssp.json` (50/50 pairs
  native == fallback, reachable + distance; batch MT == ST ==
  single rows; pred ok)
- bench: `numfast-native/results/bench_sssp.json` (V=5000/E=20000/K=50,
  seed 42: single p50 0.803ms p99 0.998ms; batch p50 10.168ms p99
  10.576ms, 4917.2 src/s)
- GAP next: batch predecessors (`pred` matrix) not on this surface

## 3c. PairInsert (generic batch-assignment pair insert — 2026-09-26)

- symbols: `nf_pair_insert_i64` (82 symbols total; frozen ABI above
  untouched, additive only)
- contract: `keys[n]` + `vals[n]` i64 typeless lanes (u64 callers pass
  bit-identical) -> caller-owned `tkeys/tvals[0..cap]` i64 +
  `used[0..cap]` u8 (`cap` power of two, `cap >= 1`; `used` WARM-zeroed
  first). Input order, duplicate key = last-write-wins (never an error,
  unlike `nf_join_build` unique contract); full-range keys incl MIN/MAX
  (no reserved lane); probe `probe_step(key as u64)` mul-shift +
  linear `next_probe` (same `core::hash` atoms as `join_build`).
- codes: `ng >= 0` occupied slots; -2 null-state (like sorted/carry
  lane-returning symbols); -3 bad geometry (cap not pow2, cap == 0,
  short table)
- single-threaded ST order (no threads: WASM-compatible by construction,
  no wasm export — native CPU research surface, same gate as `sssp`);
  no GPU path; `core`-only (no alloc, no collections)
- Python: `src/Relational/PairInsert/` Extension (`PairInsert.toml`
  alias/mods `pair_insert/pair_insert_available`,
  `kernel.metadata["PairInsert"]`, `full.toml` wired);
  `_lib/pair_insert.py` native-first + bit-exact fallback (same
  mul-shift/linear-probe/last-wins in numpy/Python); generic lanes only
- parity: `numfast-native/results/parity_pair_insert.json`
  (native == fallback bit-exact, n=2000/cap=4096 seed 42, ng=1999;
  last-wins lookup spotcheck ok)
- bench (n=2000/cap=4096, seed 42): native p50 0.037ms vs fallback p50
  6.489ms
- tests: `tests/fast/test_ops_pairinsert_bitmasksweep.py` (4 passed)

## 3d. BitmaskSweep (generic u64 bitmask sweep — 2026-09-26)

- symbols: none new on the Rust ABI (WGSL + CPU surface; 82 symbols
  total unchanged)
- contract: `masks[n]` u64 + `probes[p]` u64 predicate-swizzle subset
  queries -> `hits[n*p]` u8 row-major
  (`hits[r*p+j] = 1 iff (masks[r] & probes[j]) == probes[j]`),
  `counts[p]` uint64 per-probe hits, `popcnt[n]` uint64 per-row
  popcounts. Caller-owned fresh outputs; `n == 0 || p == 0` returns
  empty lanes. Generic lanes only (u64/i64 views bit-identical).
- WGSL (`_lib/bitmask_sweep.py::_SWEEP_WGSL`, inline string per native
  gate — no standalone `.wgsl` outside `numfast-native`): u32 lo/hi
  split pairs (WGSL has no u64; subset test bit-exact on split lanes),
  `var<uniform> prm: vec4<u32>` runtime params (N/K/M never baked,
  SPEC-DELTA-9); GPU writes `hits`, host merges `counts`/`popcnt`
  (same hybrid discipline as GroupBy); bit-exact numpy fallback owns
  the path when wgpu is missing (GPU without CPU fallback forbidden)
- Python: `src/Relational/BitmaskSweep/` Extension
  (`BitmaskSweep.toml` alias/mods
  `bitmask_sweep/bitmask_sweep_available`,
  `kernel.metadata["BitmaskSweep"]`, `full.toml` wired)
- parity: `numfast-native/results/parity_bitmask_sweep.json`
  (WGSL == numpy bit-exact, n=512/p=8 seed 42; full path == fallback;
  popcnt exact vs `bin().count`)
- bench (n=512/p=8, seed 42): WGSL p50 2.721ms vs numpy p50 1.533ms
  (512x8 dispatch-bound; GPU wins at scale, numbers honest)
- tests: `tests/fast/test_ops_pairinsert_bitmasksweep.py` (4 passed)

## 4. Python API entry points

- `numfast.native_info()` → `{'disabled','dll','dll_exists'}`
  (2026-09-21: `dll=src/numfast/_native/numfast_native.dll`, `dll_exists=True`)
- `src/Drivers/CPU/_lib/native_cpu.py` (standalone, numpy/ctypes only):
  `available()`, `why()`, `fused_sum_count()`, `multi_sum_count()`,
  `sum_count_i32/i64()`, `mixed_sum_count()`, `mixed_sum_only_count()`,
  `count_only()`, `pack_i32_direct()`, `pack_sum_count_i32/f64()`,
  `pattern_encode_buffers()`, `sorted_run()`, `select_scatter()`,
  `mask_and/or/not()`, `shift_scatter()`, `cumsum_scatter()`,
  `map_scatter()`, `owner_shard_into()`, plus `*_available()` probes
  (`select/shift/map/cumsum/variant/mixed/fused/owner_available`).
- `src/Drivers/CPU/_lib/rowwise_kway.py` (canonical, standalone):
  `rowwise_kway_time_argmin_gather()`, `kway_available()`, numpy fallback.
- `src/Drivers/CPU/_lib/rowwise_min4_time.py` (thin compat wrapper over K-way):
  `rowwise_min4_time_argmin_gather()`, `time_rowmin_available()`.
- Numba (`src/Drivers/CPU/_lib/grouped_hash.py`, `cache=False`) stays the
  grouped-distinct fast lane (`pair_hash` first, `sorted_dedup` fallback);
  untouched by this build.

## 5. Fallback matrix

| condition | effect |
|---|---|
| `NUMFAST_NATIVE_DISABLE=1` | `available()` False; every wrapper runs the NumPy reference verbatim |
| DLL file missing / load error | same as above (`why()` reports `unavailable:…`) |
| new-lane symbols missing (older DLL) | only that lane's `*_available()` False; proven groupby/pack lanes unaffected (no whole-backend cascade) |
| int32 gate fail / int overflow / out-of-range key | explicit `OverflowError`/`RuntimeError` or `None` → caller runs proven path (never wraps, never silent lossy) |
| dtype without native lane (int64/bool/object…) | NumPy reference owns it |

Verified 2026-09-21: native `fused_sum_count` == fallback `fused_sum_count`
`([4.,7.,4.],[2,2,1])`; `select/shift/map/cumsum` fallbacks bit-identical
on probe vectors.

## 6. External consumer connect (no benchmark copies)

From any workdir outside the repo:

```bat
set PYTHONPATH=C:\App\numfast\numfast\src;C:\App\numfast\app-builder
REM optional override (default already resolves to the packaged DLL):
REM set NUMFAST_NATIVE_DLL=C:\App\numfast\numfast\src\numfast\_native\numfast_native.dll
C:\App\numfast\.venv\Scripts\python.exe -c "import numfast; print(numfast.native_info())"
```

Smoke (fast import + native probe, <1s):

```bat
C:\App\numfast\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'C:/App/numfast/numfast/src'); import numfast, numpy as np, importlib.util; s=importlib.util.spec_from_file_location('nc','C:/App/numfast/numfast/src/Drivers/CPU/_lib/native_cpu.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('available=',m.available()); print(m.fused_sum_count(np.array([0,1,0],dtype=np.int32),np.array([1.,2.,3.]),2))"
```

Expected: `available= True`, sums `[4. 2.]`, counts `[2 1]`.

## 7. Verification protocol + results (2026-09-21)

- `cargo build --release --target x86_64-pc-windows-gnu`: `Finished release profile`, 75/75 symbols resolve in both DLL slots.
- Fast import: `import numfast` 194ms; `available()` True; all 9 lane probes True
  (`select/shift/map/cumsum/variant/mixed/fused/owner`).
- Q9 exact spot + adjacent (disposable script, same engine entry as
  `tests/heavy/bench_clickbench_43.py`, DuckDB cross-check on
  `C:/App/competitions/ClickBench/data/hits_1m.parquet`, n=999978):
  Q7 EXACT, Q8 EXACT (5 groups, full-list), Q9 EXACT (full 1242-group map
  bit-identical + top-10 tie-gate EXACT). Q9 cold execute 4911ms
  (first-process numba compile inside; warm path unchanged, code untouched).
- 66-suite rerun:
  `pytest tests/fast/test_ops_text.py test_ops_text_affix.py
  test_ops_regexp_deferred.py test_ops_groupby.py test_ops_dictionary.py
  test_ops_dict_domain.py test_ops_dict_int64.py test_ops_boolmask_chain.py
  test_e2e_golden.py` (with `PYTHONPATH=C:/App/numfast/app-builder`) →
  **66 passed in 44.37s**.
- Full 43-query ClickBench matrix NOT rerun (honest subset: Q7/Q8/Q9 via the
  same `ir_*` engine path; benchmark file itself unmodified).
