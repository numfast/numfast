# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# 02 — Semantic + IR (normative)

## PURPOSE

One language (`load→filter→derive→rolling→group→reduce`) → one IR.

## INPUT

- `Table.query()/filter/derive/window/returns/groupby/sort/rank/lookup/reduce/select`, `ColumnView.as_expr()/to_ir()/returns()/rolling_mean()`, `LazyExpr/LazyColumn/LazyGroupBy`, predicate `str "col OP val" | (col,op,val)`, derive `str | LazyExpr`.

## OUTPUT

- `jobs[] {op,inputs,params,out}` → `compiler.compile → ExecutionGraph {nodes: Node(kernel_id,PortRefs,params), inputs: DataPort, outputs: DataPort, metadata}` → Optimizer (CSE/DCE/folding/fusion, pure) → Planner.

## OWNER / DEPENDENCIES

- OWNER: TableExpr + Semantic (thin wrappers, reuse Compute/Filter/GroupBy). DEPENDENCIES: Runtime (the single `Task→IR` path in `planner.py`).

## INVARIANTS

- `LazyExpr/Query` is not a second IR: only `jobs`; 1 planner call per chain; `_clone()` copies without H2D until `execute/collect/to_table`.
- Indicator composition: `returns=Shift+MapBinary(sub/div)`; `rolling=RollingSum+MapBinary(div)`; `SMA/EMA/RSI` are DAGs, not monolithic loops.
- `derive` string is a restricted eval; complex expressions use `col_expr/LazyExpr` only.
- `filter` executes as a CPU pre-mask in v1; a GPU `Compare+Scan+Gather` path is future work with preserved semantics, observable in trace.
- `groupby→mean/sum/min/max/count` executes on CPU in v1 (`GroupedQuery` on gpu→`RuntimeError`, `auto→cpu` observable).

## PUBLIC / PRIVATE

- PUBLIC: vocabulary `select/filter/where/derive/map/window/rolling_mean/returns/groupby/sort/rank/lookup/reduce` + `explain/trace/compile` (delegating to LazyExpr). PRIVATE: `_preds/_derive_specs/_active_expr`, `_apply_preds/_apply_mask`, `_eager_filter/_eager_sort`.

## WHAT MUST NEVER HAPPEN

- Duplicate lowering (`expression→jobs` clones); `filter` bypassing Query; a second Runtime; `if N>threshold` branching in the frontend.
