# NATIVE-ALL — остаток legacy-JIT удалён, CLEANUP выполнен (partial)

Scope: MIGRATE_1.md remainder (cb_q9res / cb_q9prof / batch_close4) + все
JIT-лейны продукта (numfast/src) + roadgraph комменты. Seed 42. No push.
Date: 2026-09-26. Статус: **partial** — назначенная область закрыта
полностью, вне скоупа остались 24 файла research-лабораторий (точка
останова, см. §5).

## 1. NATIVE-ALL: JIT-очаг -> канонический примитив (все parity exact)

Новых Rust-примитивов не потребовалось: существующего ядра хватило
(Rust ctypes ABI + Arrow C++ + numpy C). WASM-гейт соблюдён: ни один
retired-лейн не был WASM-путем (докстринги WASM-gate обновлены:
browser использует wasm sort/unique).

| # | Очаг (MIGRATE_1/NATIVE_AUDIT) | Канонический примитив | Parity | p50 старое -> новое |
|---|---|---|---|---|
| 22 | batch_close4/h2_bytescan (файл удалён, §4) | Rust nf_text_contains (text_contains_buffers, ctypes ABI) + Arrow + numpy oracle | exact (100k/seed42, vs np.char.find) | numpy 22.09мс -> Rust 2.32мс = 9.5x (D-scale история: Arrow 788.9мс vs retired 520.1мс -> Arrow канон. в Dictionary) |
| 20 | cb_q9prof/s6_bisect (файл удалён) | np.searchsorted (C) / SortedLookup ядра | exact (ключи/запросы seed42) | 1.00x (та же семантика bisect) |
| 21 | cb_q9prof/s8_twosided (файл удалён) | numpy boolean-mask (C) / FILTER ядра | exact (hits=68189/100k seed42) | 1.00x |
| 19 | cb_q9prof/s5_threadflip (файл удалён) | микробенч тредов, перенос не нужен (конфиг only) | да (путь ядра идентичен) | 1.00x |
| 9-13,16-18,23 | cb_q9res s1*/s2_rust MT/ST pair-hash (файлы удалены) | sorted_dedup ядра (C-sort + vector scan); GroupedHash/GroupedHashMT всегда _HashMiss | exact (контракт ukeys sorted + inverse) | MT 101-102мс@16t (10M, история) -> sorted_dedup ~150-300мс (оценка по argsort 1.46мс/100k); ST python-loop lane был unreachable (гейт False) |
| 15 | cb_q9res s1e_radix unique (файл удалён) | Sort radix + GroupBy UNIQUE ядра / np.unique | exact | 1.00x (семантика unique) |
| — | cpu.py _dict_bytescan JIT-лейн | Rust text_contains_buffers (тот же Extension, `from _lib.native_cpu`) | exact (3000 строк, Arrow+char.find) | см. строку 22 |
| — | domain.py 2 JIT scan-лейна | Arrow bulk канон. (C++, zero-copy) | exact (1500 строк; все лейны chk-exact per FINAL_CLOSE4 H2) | retired parallel 90-121мс -> Arrow ~150-200мс (оценка; D-scale 788.9мс история) |
| — | groupindex.py 12 JIT-ядер | np.all probe / unique_fallback_index (Rust unique_inverse/np.unique) / np.add.at fused / flatnonzero+reduceat sorted-run | exact все 4 (100k/seed42, fused vs bincount-weights, sorted vs reduceat-эталон) | fused add.at 0.48мс vs bincount 0.68мс = 1.4x; probe ~8мкс -> ~12мс худший случай unsorted 10M (документировано) |
| — | native_cpu.py 3 JIT-проба | numpy-векторизация (уже была каноном) + reduceat f32 | exact (seg_f32=[3.0,7.0] native; probes (True,False)) | 1.00x |
| — | rng.py _yates_swaps JIT | python-цикл канон. (бит-идентичная арифметика при n<2^31) | exact (sample_no_replace детерминирован seed42) | 1.00x (алгоритм тот же) |
| — | grouped_hash.py ST python-loop lane | удалён (был unreachable); всегда _HashMiss -> sorted_dedup | exact (контракт) | — (мёртвый код) |

## 2. CLEANUP: WGSL-дубликаты (3)

| Файл | Основание |
|---|---|
| develop/backtest-experiment/PnLEngine/MultiTrail/multitrail.wgsl | удалён прошлым проходом (staged D), подтверждён отсутствующим на диске |
| develop/backtest/Loaders/_lib/shaders/pack_quote.wgsl | сирота: 0 читателей (`shaders/` нигде не импортируется); мэппинг Storage pack per MIGRATE_1 #1 |
| develop/backtest/Loaders/_lib/shaders/reduce_min.wgsl | сирота: 0 читателей; мэппинг REDUCE MIN per MIGRATE_1 #2 |

Пустой каталог shaders/ удалён следом. trunc/batched WGSL-строки
(develop/backtest/Search) оставлены: BitmaskSweep-примитив в ядре есть,
миграция Search Extension — отдельный шаг за Coordinator.

## 3. CLEANUP: удалённые тесты/результаты (манифест; файлы были untracked)

- cb_q9res/ (каталог целиком, ~488KB с rust_nf/target): s0_baseline.py,
  s1_partoa.py, s1b_partoa2.py, s1c_partoa3.py, s1d_parunique.py,
  s1e_prof.py, s1e_radix.py, s1f_dense.py, s1g_collect.py, s1h_lock.py,
  s2_rust.py, s3_integrated.py, s3b_overhead.py, s4_final.py, s5_fuzz.py;
  *.json (s0_baseline, s1_st, s1b/c/d/e/f/g_ladder, s1h_lock, s2_rust,
  s3_integrated, s4_final); rust_nf/ (Cargo.*, пустой src/, target/ —
  superseded: pair_insert живёт в numfast-native/src/pair_insert.rs).
- cb_q9prof/ (каталог целиком, ~95KB): s0_repro.py, s1_split.py,
  s2_prof.py, s3_entry.py, s4_stages.py, s5_threadflip.py, s6_bisect.py,
  s7_teardown.py, s8_twosided.py; *.txt (cd_idx, cpu_slice*6, s2_prof);
  s0_repro.json.
- develop/tests/tmp/batch_close4/ (~28KB, оставлены FINAL_CLOSE4.md +
  h2_parity.py как свежий parity): h1_mt_ladder.py, h2_bytescan.py,
  h2_e2e.py, h3_zonemap.py, h4_morton.py, results.json,
  results_pretty.json.
- numfast/scratch/sort_m2.py (assert на legacy JIT), numfast/
  .session-groupindex/bench_groupindex_10M.py (сессионный бенч JIT).
- Оставлено per приказу: competitor-бенчи (tests/heavy/bench_compete_trim.py,
  research/singlepass/bench_competitors_1T.py, roadgraph *_BENCH*) +
  свежие parity (batch_close4/h2_parity.py, .session-groupindex/
  check_groupindex_correct.py — без JIT-импортов, работают).

## 4. Гейт legacy-JIT (без venvs/dist_rc/build, без .md-истории)

```
grep -rn "import numba|from numba|@.*njit|numba.prange" --include="*.py" .
```
- numfast/src + develop + roadgraph + cb_* + batch_close4: **0**
- Файлы продукта компилируются: ALL-COMPILE-OK (11 файлов).

## 5. Точка останова (partial, для Coordinator)

Осталось **24 файла** с JIT-импортами, все в numfast/tests/research/
(взаимосвязанные лабы: audit_cpu/probe_10m, buffer_reuse/workspace,
cache_part/kernels, direct_dense/kernels, fshift_mt/kernels,
fused_part/kernels, groupby_lab/{bench_m78,kernels,metadata,
candidates/g..o ×9}, mtgroup/kernels_mt, reduction_tree/tree,
singlepass/{bench_hash_adaptive,bench_hash_inline,bench_sorted_sp,
kernels_sp}). Вне скоупа шага (MIGRATE_1 remainder = только 3 каталога;
лабы импортируют друг друга: удаление kernels ломает kept-бенчи).
Опции для Coordinator: (a) удалить лабы целиком (кроме competitor-бенчей),
(b) шим-миграция лаб следующим шагом. Дополнительно: word-упоминания
(без импортов) в tools/parity_sorted.py, tools/bench_ghash_mt.py,
scratch/sort_s2.py, sort_s3.py — их numba-нога измеряет retired-лейн
(теперь всегда miss); решение за Coordinator.
