# Tables — работа с таблицами

## Описание

Модуль `Tables` предоставляет утилиты для работы с таблицами на CuPy (GPU).
Вместо Pandas DataFrame используется `dict[str, cupy.ndarray]` — каждая колонка
это отдельный 1D-массив на GPU.

## GPU-first

Все данные через CuPy уходят на GPU. При отсутствии GPU — прозрачный CPU-режим.
Функции возвращают `dict[str, xp.ndarray]` без копирования на CPU.

## Функции

### `table_main(data) -> dict`

Создаёт таблицу из словаря. Каждое значение конвертируется в cupy-массив.

```python
t = exts.Tables.table_main({"a": [1, 2, 3], "b": [4, 5, 6]})
# {"a": cupy.ndarray([1, 2, 3]), "b": cupy.ndarray([4, 5, 6])}
```

### `merge_tables(left, right, on=None) -> dict`

Объединяет две таблицы по ключу на GPU.

```python
t = exts.Tables.merge_tables(
    {"id": [1, 2], "x": [10, 20]},
    {"id": [1, 2], "y": [100, 200]},
    on="id"
)
```

### `aggregate_table(data, group_by, agg="sum") -> dict`

Агрегирует данные по колонке на GPU.

```python
t = exts.Tables.aggregate_table(
    {"cat": [1, 1, 2], "val": [1, 2, 3]},
    group_by="cat",
    agg="sum"
)
```

Поддерживаемые агрегации: `sum`, `mean`, `min`, `max`, `std`, `var`.

## Внутренняя реализация

Вся логика в `Tables/_lib/tables_lib.py`:

- `_to_dataframe()` — конвертация `dict[list]` → `dict[xp.ndarray]`
- `_merge()` — объединение через broadcasting + xp.where
- `_aggregate()` — groupby через xp.unique + маски
