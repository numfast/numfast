# Series — работа с рядами

## Описание

Модуль `Series` предоставляет утилиты для работы с одномерными числовыми рядами на CuPy (GPU).

## GPU-first

Все данные живут на GPU через CuPy. При отсутствии GPU — автоматический CPU-режим.
Функции возвращают массив, совместимый с CuPy-API, без копирования на CPU.

## Функции

### `series_main(data) -> ndarray`

Создаёт массив из переданных данных на GPU.

```python
arr = exts.Series.series_main([1, 2, 3, 4, 5])
# cupy.ndarray([1., 2., 3., 4., 5.])
```

### `rolling_window(data, window=3) -> ndarray`

Скользящее окно по ряду. Возвращает 2D-массив на GPU.

```python
windows = exts.Series.rolling_window([1, 2, 3, 4, 5], window=3)
# cupy.ndarray shape (3, 3)
```

### `normalize_series(data) -> ndarray`

Нормализация ряда в [0, 1]. Результат на GPU.

```python
norm = exts.Series.normalize_series([10, 20, 30, 40, 50])
# cupy.ndarray([0.0, 0.25, 0.5, 0.75, 1.0])
```

## Внутренняя реализация

Вся логика в `Series/_lib/series_lib.py`:

- `_to_array()` — конвертация в `xp.ndarray` (CuPy; CPU — автоматически)
- `_rolling()` — скользящее окно через broadcasting индексов
- `_normalize()` — min-max нормализация
