# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# tests/ — конвенция (v0.2.1)

Реализации пока нет, только раскладка. Код библиотеки пойдёт поверх следующих миссий.

```
tests/
  README.md        ← эта конвенция
  conftest.py      ← маркеры fast/heavy + heavy по умолчанию пропускаются
  fast/            ← секунды, претят быстро; запуск по умолчанию
  heavy/           ← память/нагрузка; ТОЛЬКО явно
```

## Правила

- `tests/fast/` — юнит-конвенанс секунды-масштаба. Маркер `fast` (или без маркера —
  conftest считает немаркированное обычным fast). Запуск по умолчанию: `pytest tests/`
  выполняет только fast (heavy auto-skip, см. `conftest.py`).
- `tests/heavy/` — нагрузка/память (многогигабайтные буфера, multi-chunk GPU,
  долгие эквивалентности). Каждый файл/тест обязан иметь маркер `heavy`.
  Запуск только явно: `pytest -m heavy tests/` (или конкретный файл).
- Маркеры pytest: `fast`, `heavy` (зарегистрированы в `conftest.py`, без warnings).
- Фиксированный seed `42` для всех воспроизводимых тестов/бенчмарков.
- Числовые пороги — только из `../specs/conformance-profile.toml`
  (atol/rtol/ULP, `max_dict_entries`), не хардкод в тестах.
- heavy-тесты обязаны документировать в docstring: сколько памяти/времени и что доказывают.

## Команды

```bash
pytest tests/            # только fast (default)
pytest -m heavy tests/   # только heavy (явно)
pytest tests/ -m "fast"  # то же, что default
```
