# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 01 — Public API (normative)

## PURPOSE

The single public surface: `import numfast as nf`.

## INPUT / OUTPUT

- INPUT: `nf.series/nf.zeros|ones|full|arange|linspace|index|tile|repeat`, `nf.random.*(seed required)`, `nf.table/read_csv`, `Table/ColumnView`, `Query/LazyExpr`, `nf.mean/total/std/var/min/max/count/scan/sort/histogram/matmul/fft/topk/groupby`, `nf.mods.*`, `nf.optimize`, `nf.explain/trace/profile/calibrate/device_info/set_backend`.
- OUTPUT: lazy DAG → `.compute()/.evaluate()` → `NumericSeries/Table/dict/scalar`; final readback only via `.data()/.to_numpy()/to_list()`.

## OWNER / DEPENDENCIES

- OWNER: Semantic + TableExpr + Storage. DEPENDENCIES: `02` (IR), `03` (planner).

## INVARIANTS

- Rank-1 arrays only (v1.0); `int64` dtype unsupported as raw input (scaled only); `index` int32 WRAP; `log(0)→ValueError`; `**` scalar-exponent only.
- `Table.join → NotImplementedError` (deferred, not a blocker).
- `Table.filter(str)→lazy Query`; `filter(bool-mask)→eager compat`; `Table.sort` eager compat, lazy only via `query().sort()`; `to_list` for final readback only.
- `nf.mods.*` is a CPU-oracle reference (GPU is future work, API is stable).
- `tan` is unavailable in JS; Gather int32 on GPU — exact CPU fallback.

## API SIMPLICITY (normative)

- NumPy/Pandas-like syntax (normative): `nf.mean(x)`, `t["col"]`, `t.filter("price > 10")`, `t.groupby("sym").mean()`, `x.rolling_mean(20)`, `df["ret"] = df["close"].returns()`. A non-NumPy idiom as the only path is REJECT.
- Three equivalent styles for each covered operation (normative — all three must work equivalently): (a) row-wise/imperative (`for row in t.rows(): ...` / `t["c"][i]` — slow but valid for small N and learning); (b) array/vectorized (`nf.mean(t["c"])`, `t["a"] + t["b"]`); (c) chained (`t.query().filter(...).derive(...).groupby(...).reduce().compute()`). A missing style for a covered operation = incomplete API.
- Composable vocabulary (normative): all verbs `select/filter/where/derive/map/window/rolling_mean/returns/groupby/sort/rank/lookup/reduce` combine in any semantically meaningful order; `LazyExpr` is reusable (`e = t["c"].as_expr(); nf.mean(e+1)` and `nf.mean(e*2)` are both valid, CSE in the optimizer).
- Explicit, instructive errors: every error carries `what + how-to-fix + doc-link` (`"log(0)→ValueError: use t.filter('x>0') before nf.log"`), never a bare `AssertionError` on the public path.

## PUBLIC / PRIVATE

- PUBLIC: everything above. PRIVATE (`nf._internal`): `TABLE_REGISTRY`, `PackingPlan/ColumnLayout`, `pack_rows/extract_column/compute_plan`, `buffer/generation/gpu_handle/bucket`, `GpuBufferPool` internals, `_table_id`.

## WHAT MUST NEVER HAPPEN

- A new public path bypassing `Query→LazyExpr→jobs→ExecutionGraph`; `to_list` in the hot path; silent precision change; a second IR.
