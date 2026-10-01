# COST_TRAVEL_AUDIT (TRACK-A) — 2026-09-22

Sources: `numfast-native/src/cost.rs` (`cost_travel_batch`, `cost_intern_rows`, `travel_ms_from_mm`), `numfast-native/src/lib.rs:2159-2243` (FFI `nf_cost_travel_batch`, `nf_cost_intern`), `RUNTIME_MANIFEST.md` §2-3 (series/misc lane), `numfast-native/tools/`, `numfast-native/results/`, `numfast-native/src/rowmin.rs`, `numfast-native/src/rowmin_kway.rs`, `numfast-native/src/lib.rs:2535-2683` (FIX-1 + K-way FFI).

## 1. Умеет (контракт буквально)

- `nf_cost_travel_batch(dist:u32[n], speed:u32[n], k:u16[n], n, out:u32[n]) -> i32`: `out[i]=(dist[i]*k[i]+speed[i]/2)/speed[i]` (u64 middle, INF=`0xFFFFFFFF` passthrough без арифметики, `speed==0->INF`, `v<1&&len>0->1`, `v>=UINT32_MAX->INF`, `k==0->BAD_RANGE(-2)` с частичной записью до аборта; `n==0->0` без дерефа; `null->-1`). Ряд-независимый, детерминированный, directed/parallel семантика — строки вызывателя (без merge/MIN). Вход `dist` traffic-independent (bucket влияет только через `K`).
- `nf_cost_intern(vecs:u32[n*width], n, width, ids:u32[n], uniq:u32[total]) -> i64`: stateless dedup идентичных строк (first-appearance `cost_id`, compact table в том же порядке; parallel — без MIN-collapse; directed — отдельные строки; `n==0->0`, `width==0&&n>0->-2`, короткий `uniq->-3`).
- `RUNTIME_MANIFEST §cost`: строка `series/misc: nf_cost_intern, nf_cost_travel_batch` (без §cost-бенча/parity; в `tools/` нет `parity_cost.py`/`bench_cost.py`, в `results/` нет `parity_cost.json`/`bench_cost.json` — проверено листингом 69/47 файлов).
- Где/как выбирается argmin: НЕ в cost-примитиве (в `cost.rs`/`nf_cost_travel_batch` argmin нет вообще). Выбор — в отдельных rowwise-ядрах (см. п.2).

## 2. Мешало раньше (старый NO-GO буквально)

- Старый `rowwise_min4_argmin_gather` (`rowmin.rs:31-82`, FFI `lib.rs:2486-2533`): `m[i]=argmin_k(d[k][i])` — выбор по D-лейнам (float32), strict `<` от lane 0, tie=наименьший индекс. Для travel-time это неверный селектор (выбирает по дистанции/нагрузке-D, а не по времени-T) — отсюда NO-GO `argmin(D) vs argmin(T)`.
- FIX-1 `rowwise_min4_time_argmin_gather` (`rowmin.rs:103-154`, FFI `lib.rs:2560-2621`): `m[i]=argmin_k(t[k][i])` — выбор по T-лейнам (int32), strict `<` от lane 0, tie=наименьший индекс; D — чистый payload (float32 bit-exact gather, NaN/Inf/-0.0 ride). Плюс time-only путь (все D null).
- Канонический `rowwise_kway_time_argmin_gather` (`rowmin_kway.rs:26-56`, FFI `lib.rs:2647-2683`): то же правило для generic `K=1..=256` (`m=argmin_k(t[k])`, strict `<` от lane 0, tie=наименьший индекс, `t_best` exact, `d_best` bit-exact; `k==0||k>256->-2`).

## 3. NO-GO жив/снят

- СНЯТ для селектора: T-argmin с нужным tie-правилом существует в двух символах (`nf_rowwise_min4_time_argmin_gather` FIX-1 + канонический `nf_rowwise_kway_time_argmin_gather`), контракты буквально совпадают по правилу (strict `<` от lane 0, tie=smallest index, D никогда не выбирает). K4 согласован бит-в-бит с K-way (см. комментарий `rowmin.rs:84-86` + K-way parity 1025 кейсов).
- Cost-ядро к NO-GO непричастно: оно argmin не содержит, менять его не требуется. Замеров производительности в этом аудите нет.

## 4. Нужен ли новый примитив

- Нет. `nf_cost_travel_batch` + `nf_cost_intern` покрывают costing/interning; T-argmin покрыт `nf_rowwise_kway_time_argmin_gather` (канонический) / FIX-1 K4-совместимым символом. Композиция — вне ядра (caller rows + вызов двух существующих символов). Новых FFI не требуется.

## Тест символа (вне продакшена, ctypes, руками)

- Да, выполнен 2026-09-22 против `src/numfast/_native/numfast_native.dll` (оба cost-символа + оба T-argmin символа резолвятся).
- `nf_cost_travel_batch`: 20/20 parity с `travel_ms_from_mm` (включая INF/speed0/v<1->1/overflow->INF), `K==0->-2` с записью до аборта, `n==0->0`; `nf_cost_intern`: smoke `ng=2, ids=[0,1]` на двух разных строках. Итог: символ работает.
