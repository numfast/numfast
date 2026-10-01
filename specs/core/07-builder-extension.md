# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 07 — Builder + Extension (normative)

## PURPOSE

The only way to add behavior is an Extension. The Builder is frozen unless proven otherwise.

## INPUT / OUTPUT

- INPUT: folder `Ext/{Ext.toml, Ext.py, _lib/*.py, [shaders/wgsl/]}`, `_main/_main.toml` or `full.toml`.
- OUTPUT: `setup(kernel)` fills `kernel.metadata` (+ `alias/mods` in kernel).

## OWNER / DEPENDENCIES

- OWNER: Builder (assembly) + Guardian (compliance). DEPENDENCIES: `00`.

## INVARIANTS

- TOML: `name,version,depends,alias,mods,[metadata]`; `alias==mods` (length+names); `depends` lists Extension names, not packages.
- `Ext.py`: only `from _lib... import` + `setup(kernel)` (metadata only, no logic).
- `_lib/`: one file — one responsibility; `__init__` exports only.
- HIERARCHY (normative invariant — dependency tree):
  - Global/core helpers live at the ROOT (or in an explicitly shared `Core/Common` Extension): `dtypes/uniforms/dispatch-math/buffer-utils/error-contract`.
  - Nested Extensions are deeper: `Compute→{Reduce,Map,Scan}`, `Storage→{NFS,Encoding}`, `Drivers→{GPU,CPU}`, `Semantic→{TableExpr,Filter,GroupBy}`. Depth = specialization.
  - Dependency rule: an Extension depends ONLY on (a) the parent/shared level of the hierarchy or (b) explicitly declared dependencies through the Builder (`depends` + `alias/mods` + `setup(kernel)`). Examples: `Reduce→Compute OK`; `NFS→Storage OK`; a private `GroupBy→Compute._lib` import is FORBIDDEN (GroupBy is a sibling/consumer, not a child of Compute; reuse goes through the Builder mechanism, i.e. `depends=["Compute"]` + calls via the `kernel.metadata` alias, not `from Compute._lib import`).
  - A shared helper lives ONCE at the appropriate shared level and is reused through the Builder mechanism (`depends/alias/setup`); it is NOT copied and NOT privately imported. `NO INTERNAL IMPORTS` stands with this clarification: private cross-Extension imports (`from <Ext>._lib import`, `import <Ext>._lib` — REJECT) are forbidden; allowed are only (1) imports inside the Extension's own `_lib` and (2) reuse through the Builder (`depends/alias/setup`).
  - Copying a helper to avoid an import is also REJECT (duplication instead of promotion to the shared level). The correct fix for a duplicate: promote the helper to the shared level + declare `depends`.
  - Absolute `from <SameExt>._lib.<private>` for 1–2 helpers is discouraged (prefer a local function in the same file or promotion one level up, not a shared utility inside someone else's Extension).
- Disposable code (`develop/tests/tmp`, `demo/gpu_test/benchmark/notebook/tmp`) may ship without a manifest, but never as a shipped feature.
- `sys.path/mount/import-hook` is Builder tooling only, never in `_lib`.
- Closure: a change that requires editing `builder.py/kernel.py` must first prove impossibility through `add_to/mount/setup`.

## PUBLIC / PRIVATE

- PUBLIC: `alias` surface. PRIVATE: all of `_lib`.

## WHAT MUST NEVER HAPPEN

- Standalone `.py` features; code in `__init__`; `setup()` with logic; editing an existing Extension for new behavior (new Extension only); editing the Builder without proof.
- Private cross-Extension imports; copying a helper instead of promoting it to the shared level; relative-import shortcuts replacing `depends/alias/setup`.

## REQUIRED EXTENSION SURFACE — rewritability (normative)

The spec is detailed down to function level: the library can be reimplemented from it (functionality, not code quality). Each Extension MUST provide the listed functions/operations with the stated signatures and semantics (names are normative; further internal decomposition is free):

- `Core/dtypes`: `canonical_dtype(name: str) -> {itemsize, logical, accum}`; `is_scaled(dtype) -> bool`; `accum_dtype(logical) -> int64|f64-rule`; `check_overflow(value, contract: lossless|bounded|wrap) -> value|raises`.
- `Compute/Map`: `map_unary(x: Array, fn: str, dtype_rule) -> Array` (elementwise, `chunkable=true`); `map_binary(a, b|scalar, op: add|sub|mul|div|pow-scalar, dtype_rule) -> Array`; `compare(a, b|scalar, op: ==|!=|<|<=|>|>=) -> BoolMask`.
- `Compute/Reduce`: `reduce(x: Array, op: sum|count|mean|min|max, skipna: bool) -> scalar` (associative, `chunkable=true`, incremental scalar D2H); `var/std(x, ddof) -> scalar` (via two-pass or Welford, NumPy-compatible `ddof` semantics).
- `Compute/Scan`: `scan(x: Array, op: cumsum|cumprod|cummin|cummax, inclusive: bool) -> Array` (`chunkable=false` currently); `shift(x, n: int, fill) -> Array`.
- `Compute/Rolling`: `rolling_sum(x, window: int, min_periods) -> Array`; `rolling_mean(x, window, min_periods) -> Array` (`=RollingSum+MapBinary(div)`, not monolithic).
- `Compute/Sort`: `sort(x: Array, ascending: bool) -> Array` (power-of-two GPU, else observable CPU fallback); `topk(x, k) -> (values, indices)`; `rank(x, method) -> Array` (CPU-only with observable reason).
- `Compute/MatMul+FFT+Hist`: `matmul(a: rank1|batch, b) -> Array` (`chunkable=false` currently); `fft(x) -> ComplexArray`; `histogram(x, bins, range) -> (counts, edges)`.
- `Compute/Gather`: `gather(x: Array, idx: int32 Array) -> Array` (GPU int32, else exact CPU fallback); `filter_mask(x, mask: BoolMask) -> Array` (CPU pre-mask in v1; GPU `Compare+Scan+Gather` is future work with preserved semantics).
- `Storage/Encoding`: `plan(schema, n, reuse_hint) -> PackingPlan`; `pack_rows(rows, plan) -> ColumnLayout` (Disk/Code→Host only); `analyze_range(block: bytes[16B-meta]) -> {min,max,bits}`; `block_encode(physical_block[B=256], base=min(block)) -> {base, deltas, bits}`; `block_decode(header+packed) -> physical_block` (lossless for int/scaled-int); `schema_transform(logical, scale, offset) -> physical`; `schema_untransform(physical, scale, offset) -> logical`.
- `Storage/NFS`: `ingest_csv(path, schema, chunk_n) -> Table` (two-pass or append-only per `09`); `ingest_dzst(path) -> (int64[N,6], multiplier, power)`; `persist(table, path) -> metadata`; `optimize(table) -> PackingPlan` (canonical repack through Storage).
- `Drivers/CPU`: `execute(packets) -> buffers` (numpy oracle, exact on small N); `capability() -> {ops, max_dispatch, max_buffer_bytes, chunkable_hints}`.
- `Drivers/GPU`: `execute/execute_fused/execute_wave(packets) -> resident buffers`; `check_limit(packet, max_dispatch) -> ok|ValueError`; `pack_bits∈{8,16,32}` only, else observable fallback.
- `Runtime/Planner`: `compile(jobs) -> ExecutionGraph`; `optimize(graph) -> graph` (CSE/DCE/folding/fusion, pure); `select_backend(graph, n, profile) -> {backend, reason, cost_estimate}`; `chunk_plan(op, n, backend_limit) -> num_chunks|CPU-fallback`; `explain/explain_analyze/trace(graph) -> report` (without/with exec, levels semantic→timing).
- `Semantic/TableExpr`: `query()->filter()->derive()->window()->group()->reduce()` vocabulary (see `02`); `jobs() -> jobs[] {op,inputs,params,out}`; `compile() -> ExecutionGraph` (the single `Task→IR` path, 1 planner call per chain).
- A missing mandatory function = incomplete implementation (Guardian REJECT).
