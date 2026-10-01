# NumFast Language (HLL prototype)

Prototype lives in `develop/HLL/` (Extension: `HLL.toml` + `HLL.py` +
`_lib/`). Core (`Planner`, `IR`, drivers, ABI) untouched. Proof bench:
`develop/bench_hll_mvp.py` (+`.toml`); Q30 JSON: `develop/bench_hll_q30.json`.

## 1. Философия

Язык — тонкая запись намерения, не новый движок. Любой запрос обязан
опуститься до существующих семантических `ir_*` и исполниться
существующим backend без новых kernels. Правило замкнутости: новое
поведение = Extension, never правка frozen-компонентов.

## 2. Syntax (минимальный DSL)

```
SUM(v+k) k=0..89     # 90 сумм, Q30-shape
SUM(v+7)             # одиночная сумма со сдвигом
```

Fluent-эквивалент: `scan / map / expr / filter / filter_and / compare /
group_by / agg / agg_multi / sort / top / select / distinct / conditional /
like / regex_replace / rolling / shifted / cumsum / join(inner,lookup)` —
real (lowering только существующими `ir_*`); `window` (partitioned OVER) —
stub: фиксирует намерение, jobs не эмитирует, `explain()` честно сообщает.

## 3. AST

`Field / Const / Binary / Call / Conditional`. `parse_dsl` покрывает
только `SUM(f+k)`-формы; остальное — `ValueError`, не догадки.

## 4. IR и boundary

Lowering (`lower`) эмитирует ТОЛЬКО `series / map / reduce / groupby /
sort / slice / gather` через `alias` существующего backend. Граница
жёсткая: HLL не знает kernels, драйверов, GPU. Зависимости (включая
`affine_fn` из reference `develop/OptimizerAffine`) — инъекцией, never
импортом чужого `_lib`.

## 5. Optimizer passes

`cse → pushdown → prune → affine → fusion-plan → dce → residency-plan`.
Переписывают граф только `cse / affine / dce` (первые и последний —
существующий `optimize`, средний — reference `affine_sum_lift`);
`pushdown / prune / fusion-plan / residency-plan` — plan-аннотации
(`annotate-only` в `meta.plan`), без новых kernels.

## 6. Execution

`execute` = `cpu_execute` существующего backend + host-derive скаляров
(`base + N*k`). Новых GPU kernels нет; `explain_device()` возвращает
`cpu` и причину.

## 7. Result

`ColumnCarry / ResultBuffer`. Материализации по умолчанию нет:
только явные `.collect() / .to_dict() / .to_numpy()`.
Интроспекция: `explain() / explain_plan() / explain_optimized() /
explain_cost() / explain_device()`.

## 8. GPU / ABI / embedding / dispatcher

GPU-путь не добавлен (прототип CPU-only поверх существующего backend).
ABI (`compile → optimize → cpu_execute`, граф `{nodes, inputs, outputs,
metadata}`) не менялась. Встраивание: `MAIN["build"](app_dir)` +
`alias`-словарь. Диспетчер не менялся; RoadGraph не используется.

## 9. Future: RoadGraph как primitive

Маршрутизация/placement — будущий отдельный primitive, не часть HLL.
Требует GO: дизайн RoadGraph вне HLL, GPU kernels по приоритету
`CCI → MEDIAN → STOCH → STDDEV → RSI`, перенос affine-семантики в core.

## 10. Q30 proof (факт, не цель)

DSL `SUM(v+k) k=0..89` → lowering (181 job, побайтово равен Path A) →
optimizer auto → 2 nodes + `meta.affine` → EXACT vs Path A и DuckDB,
`max_diff 0`, rows `999978`. См. `develop/bench_hll_q30.json`.

## 11. MVP-матрица (оператор → ir_* → nodes → proof EXACT)

MVP = 7 пунктов, CPU-only (`explain_device()=cpu`), lowering только
существующими `ir_*` из `src/Semantic/IR/_lib/nodes.py`, без правок
Planner/IR/ABI, без новых GPU kernels, без RoadGraph/Dispatcher.
Каноны review: `filter` арность 3 `(out,values,mask)`,
`groupby` позиционные `values/keys` + `op/result`,
`sort` позиционные `*keys` + `descending`,
`slice` позиционные `values` + `limit/offset`.

| оператор | ir_* | nodes (canonical call) | proof |
|---|---|---|---|
| scan | `ir_series` | `ir_series(name, arr)` zero-copy by-ref | Q30 + micros EXACT |
| expressions+map | `ir_map` | `ir_map(out, inp, fn, value)` scalar→`[inp]`, str→`[inp,value]` | Q30 181→2 EXACT same-graph |
| filter | `ir_compare→[ir_mask]→ir_filter` | `ir_filter(out, values, mask)` arity 3, 3VL invalid exclude | micro EXACT `max_diff 0` (N=20 seed 42) |
| group_by+agg | `ir_groupby / ir_groupby_multi + ir_pack_keys` | `ir_groupby(out, values, keys, op, result="carry")`, composite `ir_pack_keys(pk, *keys, mode="hash")`, mean=`sum+count+div` | micro EXACT sum/mean/composite |
| sort | `ir_sort` | `ir_sort(perm, *keys, descending=...)` perm-only | micro EXACT vs stable sort |
| top | `ir_sort→ir_slice→ir_gather` | `ir_slice(sli, perm, limit, offset)` on perm + `ir_gather(out, values, sli)`; `filter(top)` = та же сахар-композиция | micro EXACT + hits_1m spot top-3 EXACT |
| explain | — | `explain/explain_plan/explain_optimized/explain_cost/explain_device` (Table + ResultBuffer), jobs не эмитит | present, `device=cpu` |

Stubs (честные, `NotImplementedError` + `explain`, jobs не эмитить):
`window` (partitioned OVER; unpartitioned `rolling / shifted / cumsum`
real). `join` PARTIAL: `inner/lookup` key-lookup real
(`ir_series(build)+ir_lookup` → positions + `#hit/#k`; dupe build = error);
payload gather + left/outer nulls — stage-5 execution-contract boundary,
этап 5 НЕ начат. `like` честные границы: `_` и interior multi-`%` —
`NotImplementedError` (stage-5, never guessed). `multi-column` PARTIAL:
int one-traversal EXACT; float64 scale/offset — stage-5 contract.
Proof: `develop/bench_hll_stage4.py` (`.toml`) → `bench_hll_stage4.json`,
verdict GO (seed 42): 10/10 DONE-EXACT + 3/3 PARTIAL-EXACT
(join/window/multi-column), `immutability/explain/zero-copy` EXACT,
Q30 181→2 intact EXACT vs Path A и DuckDB (`max_diff 0`, rows `999978`).
Разблокированы ClickBench Q5/Q6/Q9 (distinct), Q21 (LIKE), Q29 (regex),
Q40 (conditional); PARTIAL Q10-Q12 (per-key distinct), H2O Q3-Q5 (int).
Осталось stubs/GO: всё вне stage-4 — только по GO Coordinator.
