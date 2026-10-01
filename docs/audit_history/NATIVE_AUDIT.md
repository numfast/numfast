# NATIVE-AUDIT — натив вне ядра numfast

Scope: все вне `C:\App\numfast\numfast\`. Инвентарь. Переноса нет.
Date: 2026-09-26. Seed: 42. No push.

## Итог

- Сырые файлы `.rs/.wgsl/.wat/.wasm/build.rs/Cargo.toml` вне ядра: 4
- WGSL-строки в Python `@group`: 4 файла
- `ctypes` + своя DLL: 1 файл
- `cffi`: 0
- Cython: 0
- `numba @jit/@njit` с hotspot-логикой: 14 файлов (+2 файла только thread-конфиг)
- CUDA / OpenCL / WAT / WASM-refs в Python вне ядра: 0
- JS/TS WGSL-дубли вне ядра: 4 файла
- Всего нативных очагов вне ядра: 27

## Таблица

| # | Путь | Тип | Что делает | Примитивы ядра | Перенос |
|---|---|---|---|---|---|
| 1 | develop/backtest/Loaders/_lib/shaders/pack_quote.wgsl | wgsl файл | Pack OHLC: low minus offset, копия d_open/d_high/d_close/buy_vol/sell_vol в u32 | да: Storage pack, MAP pack_quote | тривиально |
| 2 | develop/backtest/Loaders/_lib/shaders/reduce_min.wgsl | wgsl файл | Reduction min по low i32 через shared + atomicMin | да: REDUCE MIN | тривиально |
| 3 | cb_q9res/rust_nf/Cargo.toml | cargo манифест | Манифест cdylib nf_pair_distinct, release lto thin | частично: сборка натива в ядре numfast-native | сложно |
| 4 | cb_q9res/rust_nf/src/lib.rs | rust | nf_pair_insert: open-addressing insert пары i64, mul-xor-shift, linear probing | частично: GROUPBY hash, JOIN hash | нужен новый примитив ядра |
| 5 | develop/backtest/Search/_lib/batched/wgsl.py | wgsl строка 105 | EvalCondBitmask: f32 сравнения в бимаску; EvalTupleSweep: bucket sums ret/ret2, popcount, transitions per-bar | частично: FILTER + SCAN + REDUCE | нужен новый примитив ядра |
| 6 | develop/backtest/Search/_lib/trunc/wgsl.py | wgsl строка 2 | trunc f32 к нулю | да: MAP trunc, pointwise | тривиально |
| 7 | roadgraph/RoadGraph/_lib/tbrs_gpu.py RELAX_WGSL | wgsl строка 6 | Bellman-Ford relax: atomicMin dist, changed flag, saturating add | частично: SSSP, STATEKERNEL | нужен новый примитив ядра |
| 8 | roadgraph/scratch_assign_probe/probe.py EVAL_WGSL | wgsl строка 12 | Assign probe: time_ok, cap_ok, skill_ok, feas, cost travel+wait | частично: MAPBINARY + FILTER | сложно |
| 9 | cb_q9res/s2_rust.py | ctypes DLL + njit | Грузит nf_pair_distinct.dll, сверяет rust против njit _pair_insert на ClickBench hits_10m | частично: GROUPBY hash | сложно |
| 10 | cb_q9res/s1_partoa.py | njit 4 | _fp_fill, _scatter, _part_insert prange, _part_collect_st | частично: PARTITION + GROUPBY hash | сложно |
| 11 | cb_q9res/s1b_partoa2.py | njit | Партиционирование вариант 2 | частично: PARTITION | сложно |
| 12 | cb_q9res/s1c_partoa3.py | njit | Партиционирование вариант 3 | частично: PARTITION | сложно |
| 13 | cb_q9res/s1d_parunique.py | njit | Параллельный unique | частично: UNIQUE, GROUPBY | сложно |
| 14 | cb_q9res/s1e_prof.py | njit | Проф disproportionate стадии unique | частично: UNIQUE | сложно |
| 15 | cb_q9res/s1e_radix.py | njit 7 | radix unique: _fp_mt, _count_mt, _scatter_mt, _pins_mt, _rcount_mt, _rscatter_mt, _dedup_count | частично: SORT radix, UNIQUE | нужен новый примитив ядра |
| 16 | cb_q9res/s1f_dense.py | njit 6 | dense groupby: fp, count, scatter, pins, msort_seg | частично: GROUPBY dense | сложно |
| 17 | cb_q9res/s1g_collect.py | njit | collect стадия groupby | частично: GROUPBY collect | сложно |
| 18 | cb_q9res/s1h_lock.py | njit | lock-контеншн эксперимент | нет: микробенч синхронизации | сложно |
| 19 | cb_q9prof/s5_threadflip.py | njit | thread-flip замер | нет: микробенч тредов | тривиально |
| 20 | cb_q9prof/s6_bisect.py | njit | bisect замер | частично: SEARCH sorted | тривиально |
| 21 | cb_q9prof/s8_twosided.py | njit | two-sided замер | частично: FILTER | тривиально |
| 22 | develop/tests/tmp/batch_close4/h2_bytescan.py | njit 1 | byte-scan needle в u8 | частично: SCAN, FILTER | тривиально |
| 23 | cb_q9res/s2_rust.py _pair_insert | njit 1 | ST insert эталон для rust falsifier | частично: GROUPBY hash | сложно |
| 24 | parity_test.js | js wgsl | JS паритет шейдеров | да: дубли JS parity | тривиально |
| 25 | develop/browser-demo/app.js | js wgsl | Демо вызов GPU | да: дубли JS parity | тривиально |
| 26 | develop/numfast_js/src/numfast.js | js wgsl | JS runtime parity | да: дубли JS parity | тривиально |
| 27 | develop/browser-demo/src/gpu.ts | ts wgsl | TS обертка GPU | да: дубли JS parity | тривиально |

Не-очаги: cb_q9prof/s3_entry.py, cb_q9prof/s4_stages.py — только numba.set_num_threads, jit-логики нет.
Тест-обвязка WGPU_API: develop/wgpu_smoke_test.py, tests/test_gpu_math.py, develop/backtest/Tests/gpu_audit/_lib/audit.py, create_indexes.py — smoke/audit/docs, продукт-логики нет.

## План переноса

P0 тривиально, без новых примитивов:
- 1 pack_quote.wgsl → Storage pack Extension, затем удалить файл
- 2 reduce_min.wgsl → REDUCE MIN, затем удалить файл
- 6 trunc WGSL → MAP trunc pointwise, затем удалить модуль
- 24–27 JS дубли → сгенерировать из ядра, запретить ручные правки
- 19–22 микробенчи s5/s6/s8/h2 → переписать через SCAN/FILTER/SEARCH ядра в develop/tests/tmp

P1 сложно, требует адаптер Extension:
- 8 assign probe → MAPBINARY+FILTER Extension в develop, следом гейт паритета CPU/GPU
- 9–13,16–18,23 ClickBench партиция/groupby/collect → GROUPBY/PARTITION Extension, single-thread сначала
- 3 Cargo cdylib → удалить после переноса логики в numfast-native, общий билд ядра

P2 новый примитив ядра, решение за Coordinator:
- 4 nf_pair_insert → новый примитив hash pair-insert ядра
- 5 batched EvalCondBitmask/EvalTupleSweep → новый примитив bitmask-sweep ядра
- 7 TBRS relax → новый примитив SSSP/STATEKERNEL ядра
- 14–15 radix unique → новый примитив SORT radix + UNIQUE ядра

Порядок: P0, затем P1, затем P2. Каждый перенос: Extension в develop, паритет-тест, удаление исходного очага, запись в .project_index.

## Гейт: запрет натива вне ядра

Правило: файлы `*.rs`, `*.wgsl`, `*.wat`, `*.wasm`, `build.rs`, `Cargo.toml` живут только в `numfast/numfast-native/` плюс `numfast/scratch/*.wasm` артефакты. WGSL-строки `@group` в Python живут только в `numfast/**/ _lib/*.py`. Свои DLL через ctypes вне ядра запрещены.

pre-commit `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: no-native-outside-kernel
        name: no native outside kernel
        entry: scripts/no_native_outside_kernel.sh
        language: script
        pass_filenames: false
```

`scripts/no_native_outside_kernel.sh`:
```sh
#!/bin/sh
set -eu
bad=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.rs$|\.wgsl$|\.wat$|\.wasm$|build\.rs$|Cargo\.toml$' | grep -v '^numfast/numfast-native/' | grep -v '^numfast/scratch/' || true)
if [ -n "$bad" ]; then echo "$bad"; echo "NATIVE OUTSIDE KERNEL BLOCKED"; exit 1; fi
pats=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' | grep -v '^numfast/' || true)
if [ -n "$pats" ]; then
  hit=$(echo "$pats" | xargs grep -l '@group\|ctypes\.CDLL\|WinDLL' || true)
  if [ -n "$hit" ]; then echo "$hit"; echo "WGSL CTYPES OUTSIDE KERNEL BLOCKED"; exit 1; fi
fi
exit 0
```

Grep-check для CI:
```
git ls-files | grep -E '\.rs$|\.wgsl$|Cargo\.toml$' | grep -v '^numfast/numfast-native/' | grep -v '^numfast/scratch/'
git ls-files '*.py' | grep -v '^numfast/' | xargs grep -l '@group\|ctypes\.CDLL' || true
```
