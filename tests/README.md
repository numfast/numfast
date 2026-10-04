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

## Два режима: checkout и установленный пакет

Один и тот же набор тестов работает в двух режимах, и `conftest.py` выбирает
режим сам — по тому, откуда резолвится `numfast`.

**Checkout** (режим разработчика): `numfast` резолвится в `src/` этого репозитория,
`builder` — настоящий, из app-builder. Собирается всё, ничего не пропускается.

```bash
export PYTHONPATH="C:/App/numfast/numfast/src;C:/App/numfast/app-builder"
pytest tests/
```

**Установленный пакет** (то, что уходит в релиз): `numfast` резолвится в
site-packages, движка-дерева нет. Kernel берётся из `numfast.get_kernel()` —
установленный пакет сам находит свой корень по `numfast/full.toml`, поэтому
`APP_DIR`/`FORK` не входные данные. Нужны `tests/` и `specs-rebuilt/`
(их кладёт в sdist `MANIFEST.in`), но **не** `src/`, `full.toml` и
`numfast-native/`:

```bash
python -m venv .venv-wheel
.\.venv-wheel\Scripts\python.exe -m pip install dist_rc\numfast-0.2.1-py3-none-win_amd64.whl[pandas,test]
# распаковать sdist (или скопировать tests/ + specs-rebuilt/) ВНЕ дерева репозитория
.\.venv-wheel\Scripts\python.exe -m pytest <that-dir>/tests -q
```

Модули, которым всё равно нужен исходник движка, в этом режиме не собираются, и
`conftest.py` печатает в конце прогона имя каждого и причину (список — в
`conftest.py:SOURCE_ONLY`). Молча пропускать их нельзя: именно на них держатся
text-guard-ы по исходникам движка, и без дерева их нечем проверять.

