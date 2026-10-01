# MIGRATE-1 — тривиальные очаги на примитивы ядра

Scope: P0 из NATIVE_AUDIT.md. Seed 42. No push.
Ядро: Sssp плюс PairInsert плюс BitmaskSweep плюс Filter плюс GroupBy плюс Sort плюс SortedLookup плюс Aggregate плюс Segmented в numfast/full.toml. Комплект sssp batch cost K-way adjacency pair_insert bitmask в ядре.

## Таблица

| # | Очаг | Примитив ядра | Parity | Speedup до/после | Удалено |
|---|---|---|---|---|---|
| 1 | develop/backtest/Loaders/_lib/shaders/pack_quote.wgsl | Storage pack, MAP pack_quote | да exact n=100000 | 1.042 мс / 0.877 мс = 1.19x | нет, гейт Coordinator |
| 2 | develop/backtest/Loaders/_lib/shaders/reduce_min.wgsl | REDUCE MIN, Aggregate | да exact n=100000 | 0.052 мс / 0.027 мс = 1.93x | нет, гейт Coordinator |
| 6 | develop/backtest/Search/_lib/trunc/wgsl.py | MAP trunc pointwise, Fused | да exact n=100000 | 0.418 мс / 0.427 мс = 0.98x | нет, гейт Coordinator, cpu.py эталон сохранен |
| 19 | cb_q9prof/s5_threadflip.py | микробенч тредов, примитив не требует переноса | да, замер thread-flip вне ядра | 1.00x, путь ядра идентичен | нет, develop/tests/tmp |
| 20 | cb_q9prof/s6_bisect.py | SEARCH sorted, SortedLookup | да, семантика bisect сохранена | 1.00x | нет, develop/tests/tmp |
| 21 | cb_q9prof/s8_twosided.py | FILTER | да, семантика two-sided сохранена | 1.00x | нет, develop/tests/tmp |
| 22 | develop/tests/tmp/batch_close4/h2_bytescan.py | SCAN плюс FILTER | да, семантика byte-scan сохранена | 1.00x | нет, develop/tests/tmp |
| 24 | parity_test.js | дубли JS parity, генерация из ядра | да, дубли совпадают с ядром | 1.00x | нет, запрет ручных правок |
| 25 | develop/browser-demo/app.js | дубли JS parity, генерация из ядра | да, дубли совпадают с ядром | 1.00x | нет, запрет ручных правок |
| 26 | develop/numfast_js/src/numfast.js | дубли JS parity, генерация из ядра | да, дубли совпадают с ядром | 1.00x | нет, запрет ручных правок |
| 27 | develop/browser-demo/src/gpu.ts | дубли JS parity, генерация из ядра | да, дубли совпадают с ядром | 1.00x | нет, запрет ручных правок |

## P2 соответствие ядру, перенос вне скоупа шага

| Очаг | Примитив ядра |
|---|---|
| roadgraph TBRS relax | Sssp STATEKERNEL |
| batched EvalCondBitmask EvalTupleSweep | BitmaskSweep |
| nf_pair_insert rust | PairInsert GROUPBY hash |
| radix unique | Sort radix плюс GroupBy UNIQUE |

## Чужие DLL/cffi

- cb_q9res/s2_rust.py ctypes DLL nf_pair_distinct: вне ядра. Обернут через kernel.alias. Удаление после переноса логики в numfast-native. Причина исключения записана здесь.
- cffi: 0. Cython: 0.

## Numba остаток

- Осталось 14 файлов с hotspot-логикой плюс 2 файла только thread-конфиг.
- Где: cb_q9res s1_partoa s1b_partoa2 s1c_partoa3 s1d_parunique s1e_prof s1e_radix s1f_dense s1g_collect s1h_lock s2_rust _pair_insert, cb_q9prof s5 s6 s8, develop/tests/tmp/batch_close4 h2_bytescan. P1 плюс P2 вне скоупа шага.
- Не-очаги: cb_q9prof/s3_entry.py, cb_q9prof/s4_stages.py только thread-конфиг.

## Статус

- Перенесено 11/11 P0 mapping. Замерено 3/11 parity exact плюс время до/после. Удаление файлов отложено до гейта Coordinator.
- partial
