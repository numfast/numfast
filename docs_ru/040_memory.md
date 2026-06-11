# Memory — кэширование

## Описание

Модуль `Memory` предоставляет простой кэш с TTL для результатов вычислений.
Не зависит от CuPy — чистый Python.

## Функции

### `cached_calculation(key, func, ttl=60) -> object`

Кэширует результат вызова `func` на `ttl` секунд.

```python
exts.Memory.cached_calculation("heavy", lambda: sum(range(10**6)), ttl=30)
```

### `clear_expired()`

Очищает истёкшие записи (старше 60 секунд).

### `stats() -> dict`

Возвращает статистику кэша.

```python
exts.Memory.stats()
# {"total_entries": 1, "keys": ["heavy"]}
```

## Внутренняя реализация

Вся логика в `Memory/_lib/memory_lib.py`:

- `_CacheEntry` — dataclass с `value` и `timestamp`
- `_CacheStore` — хранилище с методами `get`, `put`, `delete`
