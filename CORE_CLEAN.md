# CORE_CLEAN

Ядро = generic primitives (граф/веса/таблицы/редукции). Без дорог/диспетчеров/демок.

## Классификация

### CORE (осталось, не тронуто)

Rust `numfast-native/src/`:
- core/, groupby/, join.rs, join_i64.rs, sort.rs, unique.rs, segmented.rs
- series/ (scan/reduce/select/shift/cumsum/map), transform/ (pack/pattern)
- text.rs, result/, rng.rs
- cost.rs — generic integer costing core (веса), без RoadGraph-семантики
- adjacency.rs — generic CSR slice+gather
- sssp.rs — canonical single-source shortest path, generic CSR u32
- rowmin.rs, rowmin_kway.rs — generic rowwise MIN
- bounded_select.rs — generic bounded top-K
- lib.rs — FFI-символы не переименовывались

Python `src/` (extensions по full.toml): Core, Schema, Dictionary,
IR, Planner, GroupedHash, GroupedHashMT, CPU, GPU, Fused, Join,
GroupBy, Filter, Sort, Aggregate, Segmented (segmented_reduce +
adjacency_slice/flat — generic CSR), Runtime, NFS, NfsBlocks,
NfsStream, NfsZview, SortedLookup, Sssp (generic sssp_csr/sssp_batch).

### ROADGRAPH (происхождение — в ядре только generic-срез)

- `numfast-native/src/router.rs` (`router_route_csr`, FFI `nf_router_route`):
  порт `roadgraph/ClusterRouter/_lib/router.py::route_nodes`
  (portal graph, early-exit на target set). Физически НЕ перенесён:
  символ FFI заморожен, удаление/перенос сломало бы ABI (запрет задачи).
  Потребителей в `src/` нет. Полный road-контекст живёт в
  `C:/App/numfast/roadgraph/` (ClusterRouter, RouteApi, PortalStitch).
  Рекомендация координатору: вынести следующим шагом как shim/re-export,
  не в этом бюджете.
- `cost.rs`/`Segmented/_lib/cost.py`: паритет с
  `roadgraph/RouteApi/_lib/weights.py`, но реализация generic
  (u32 lanes, u64 middle, INF guard) — оставлена как веса ядра.

### TASK (competitions/transport — внутри ядра отсутствует)

- Диспетчеров, worker/order-ограничений, TBRS, truck-модели в `src/`
  и в `numfast-native/src/` нет (в cost.rs только код ошибки truck-слота).
- Бенч-харнессы задач лежат вне ядра: `numfast-native/tools/`
  (bench_*.py, parity_*.py, wasm_*.mjs), `scratch/` (j1/j2/j4, sort_m*,
  unique_m*, showdown_*, bench_p*.json) — не тронуты, переносу в
  `task3/` нет (каталога `task3/` в окружении нет; competitions — вне scope).

### DEMO (уехало в examples/)

Из корня `numfast/` в `numfast/examples/` (с починкой импортов):
- bench_composite.py, bench_stages.py (пути `../src`, `../../app-builder`)
- bench_segmented.py (путь `parent.parent / "src"`)
- bench_join_i64.py (+ bench_join_i64_results.json), bench_lexsort.py,
  bench_newgroup.py, bench_ukeys.py, bench_unique.py, bench_verify.py,
  _measure_text_encode.py (абсолютные пути — как было)
- Статус живо/заброшено: см. `examples/README.md`.

Удалено файлов: 0. Всё удалённое — в отчёт: ничего не удалялось.

## Проверка сборки

- `cargo check` в `numfast-native/`: см. Worker Report (ок/нет + хвост лога).
- FFI-символы ядра не переименовывались; потребители в `src/` не менялись
  (у bench-скриптов потребители отсутствуют, тесты их не импортируют —
  проверено grep по tests/src/tools/develop: 0 ссылок).
