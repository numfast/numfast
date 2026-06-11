# AGENTS.md

## Роли и ответственность

Ты — ИИ-ассистент по разработке numfast.
Задача — помогать писать код, тесты и документацию для GPU-фреймворка `numfast`.

## Архитектура проекта

Проект построен на концепции **Extensions** (расширений):

```
Extension/
├── Extension.toml       # Манифест
├── Extension.py         # Публичное API (возвращает xp.ndarray)
└── _lib/
    └── extension_lib.py # Внутренние реализации на xp (CuPy; CPU — автоматически)
```

### GPU-first правило

CuPy — единственный вычислительный бэкенд. Все `_lib` внутри используют `xp`:

```python
try:
    import cupy as xp
except ImportError:
    import numpy as xp
```

Этот механизм обеспечивает прозрачный CPU-режим: если NVIDIA GPU нет,
фреймворк автоматически исполняет тот же код на CPU.
Пользователю не нужно ничего менять — ни зависимости, ни код.

**Запрещено** в публичном API:
- `.tolist()` — копирует на CPU
- `.get()` — копирует на CPU
- `.item()` — копирует на CPU
- Возвращать `list`, `tuple` или `dict` из числовых данных

**Разрешено** возвращать только `xp.ndarray`.

### CPU-режим

При отсутствии NVIDIA GPU `import cupy` не удаётся,
и срабатывает встроенный механизм CPU-совместимости.
Все операции остаются корректными — разница лишь в том, на чём выполняется код.

### Загрузка через framework-builder

```python
from framework_builder import ExtensionsLoader
for ext in ExtensionsLoader('./numfast'):
    print(ext.name)
```

### Структура проекта

```
numfast/
├── AGENTS_ru.md
├── README.md
├── pyproject.toml
├── _main/          # Ядро
├── Memory/         # Кэш (чистый Python)
├── Series/         # Ряды на CuPy
├── Tables/         # Таблицы на CuPy (structured arrays)
├── Mods/           # Декораторы (чистый Python)
├── Proxy/          # Ленивая загрузка (чистый Python)
└── docs_ru/        # Документация
```

### _main (ядро)
- `info()` — версия фреймворка
- `status()` — состояние расширений

### Series (ряды на GPU)
- `series_main(data)` → `xp.ndarray` — создание ряда
- `rolling_window(data, window)` → `xp.ndarray` — скользящее окно
- `normalize_series(data)` → `xp.ndarray` — нормализация [0,1]

### Tables (таблицы на GPU)
Вместо Pandas — `dict[str, xp.ndarray]` (каждая колонка = 1D-массив на GPU).
- `table_main(data)` → `dict[str, xp.ndarray]`
- `merge_tables(left, right, on)` → `dict[str, xp.ndarray]`
- `aggregate_table(data, group_by, agg)` → `dict[str, xp.ndarray]`

### Memory (кэш, чистый Python)
- `cached_calculation(key, func, ttl)`
- `clear_expired()`
- `stats()`

### Mods (декораторы, чистый Python)
- `@register("name")`
- `get_registered()`

### Proxy (ленивая загрузка, чистый Python)
- `load(module_name)`
- `info()`

## Зависимости

```toml
[project]
dependencies = [
    "cupy>=13.0",
    "rich>=13.0",
    "framework-builder>=0.1.0",
]
```

## Тестирование

```bash
python -m pytest tests/
```

## Соглашения

1. **Имя модуля** = имя папки
2. **Имя файла** = имя модуля
3. **lib-файл** = `_lib/{lower}_lib.py`
4. **toml-манифест** = `{Module}.toml`
5. **Публичное API** — только в `{Module}.py`, возвращает `xp.ndarray`
6. **Внутренняя логика** — в `_lib/`, использует `xp` (CuPy; CPU — автоматически)
7. **Никакого CPU-transfer** в публичном API
8. **Документация** на русском в `docs_ru/`

---

*Последнее обновление: 2026-06-10*
