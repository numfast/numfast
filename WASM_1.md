# WASM-1 — TS-bridge + Q-lookup в браузере (2026-09-26)

## 1. Архитектура моста

- Сырой cdylib ABI, без wasm-bindgen (решение записано: ноль зависимостей,
  `usize` → i32 на wasm32, ABI стабилен — биндген мешает, а не помогает).
- Один ESM-модуль `numfast-native/ts/bridge.ts`: класс `Bridge` (bump-аллокатор
  поверх linear memory, layout из `tools/wasm_mem.mjs`: `STACK_TOP=0x100000`,
  `GUARD`, канарейка, `BASE=0x101000`) + zero-copy виды (`subarray`, не `slice`;
  `put()` — единственная копия: blit входов в память wasm).
- Покрыты 8 экспортов: `nf_sssp_csr`, `nf_sssp_csr_pred`, `nf_sssp_batch`,
  `nf_cost_travel_batch`, `nf_cost_intern` (возврат i64 → BigInt),
  `nf_rowwise_kway_time_argmin_gather` (таблица указателей lane в памяти wasm),
  `nf_adjacency_slice`, `nf_adjacency_gather`.
- K-way ограничение `1 <= k <= 256`, batch на wasm однопоточен (см. §2).
- Исходник — erasable TS: Node ≥22 исполняет напрямую; браузеру нужен
  `ts/dist/bridge.js` (чистый ESM, `tsc --module es2022 --moduleResolution node`).
  `ts/qlookup.ts` импортирует `./bridge.ts` (Node) — для сборки dist правится
  в `./bridge.js` вручную, т.к. tsc без `allowImportingTsExtensions` их разводит.

## 2. WASM-сборка Rust

- Команда: `cargo build --release --target wasm32-unknown-unknown`
  (из `numfast-native/`; таргет доставлен через `rustup target add`).
- Единственное изменение логики — wasm-гейт в `src/sssp.rs::sssp_batch`:
  на `target_arch="wasm32"` те же шарды исполняются инлайн вместо
  `std::thread::scope` (порядок = порядок источников, форка логики нет).
- Артефакт `target/wasm32-unknown-unknown/release/numfast_native.wasm`,
  196149 Б (был 120282 Б, stale от 2026-09-12 без sssp/cost/kway/adjacency).
  Экспортов `nf_*`: 85 (было 61). Старый `tools/numfast_native.wasm` не тронут.
- Проверка: все 8 целевых символов — `function` в инстансе.

## 3. Замер Q-lookup

- Q-lookup = `nf_cost_travel_batch` по квантованным артефактам пары
  (dist u32 мм + speed u32 мм/с + k u16 промилле).
- Артефакты: `ts/artifacts/{dist,speed,k}.bin`, 1M пар, 9.54 МиБ, seed 42
  (`node ts/qlookup.ts --gen 1000000`).
- Node 26 (V8 — тот же движок, что в Chrome): паритет с JS-эталоном 0/10000,
  медиана ядра 14.489 мс → **69 019 305 lookup/s** против native 171k/s
  (~400×; native-базлайн — ctypes per-call overhead, не ядро).
- Браузер: `python -m http.server 8080` из `numfast-native/`, открыть
  `ts/demo.html` — fetch wasm + 3 .bin, времена пар и lookup/s в `<pre>`
  и консоли. Прямой замер в Chrome на этом хосте не выполнялся (нет
  headless-браузера в бюджете) — страница готова, результат ожидается
  порядка десятков M lookup/s (тот же V8 + WebAssembly).

## 4. GAP (Dijkstra/solver следующие)

- `nf_sssp_*` в браузере не гонялись (только наличие экспортов); нужен
  Dijkstra-партишн-бенч на графе из `develop/` + сравнение с Python
  `src/Relational/Sssp/_lib/sssp.py`.
- Solver (K-way + cost_intern связка) в TS не сведён — только атомарные обёртки.
- dist-сборка: унифицировать импорт `./bridge.js` (сейчас ручная правка),
  добавить `tsconfig.json` + `package.json` в `ts/`.
- `tools/numfast_native.wasm` (копия для harness) не обновлена намеренно.

## Пути

- `numfast-native/ts/bridge.ts`, `numfast-native/ts/qlookup.ts`,
  `numfast-native/ts/dist/bridge.js`, `numfast-native/ts/demo.html`,
  `numfast-native/ts/artifacts/{dist,speed,k}.bin`,
  `numfast-native/src/sssp.rs` (wasm-гейт),
  `numfast-native/target/wasm32-unknown-unknown/release/numfast_native.wasm`
