# examples/ — демо и бенчмарки вне ядра

Запуск из этого каталога. Пути к `src/` уже поправлены (`../src`).

| Файл | Статус |
|---|---|
| bench_composite.py | живо (groupindex microbench, seed 42) |
| bench_stages.py | живо (stage breakdown того же) |
| bench_segmented.py | живо (P1/P2 segmented vs naive) |
| bench_join_i64.py | живо (+ bench_join_i64_results.json — эталон) |
| bench_lexsort.py | заброшено (проверка packed lexsort, без seed-контракта) |
| bench_newgroup.py | заброшено |
| bench_ukeys.py | заброшено |
| bench_unique.py | живо (Unique M1) |
| bench_verify.py | заброшено (разовая проверка packed lexsort) |
| _measure_text_encode.py | заброшено (абсолютные пути C:/App, внешний parquet) |

Правило: новые демо/бенчмарки — только сюда, не в корень и не в `src/`.
