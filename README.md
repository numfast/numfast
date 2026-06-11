# NumFast

NumFast — фреймворк для GPU-численных расчётов на CuPy с модульной архитектурой на расширениях.

## Установка

```bash
pip install numfast
```

## Быстрый старт

```python
from framework_builder import ExtensionsLoader

loader = ExtensionsLoader('./numfast')
exts = loader.load()

exts._main.info()
exts.Series.series_main([1, 2, 3, 4, 5])
exts.Tables.table_main({"a": [1, 2, 3], "b": [4, 5, 6]})
exts.Memory.cached_calculation("key", lambda: 42)
exts.Proxy.load("cupy")
```

## Расширения

| Модуль   | Назначение                          |
|----------|-------------------------------------|
| `_main`  | Ядро фреймворка                     |
| `Series` | Ряды на CuPy (GPU)                  |
| `Tables` | Таблицы на CuPy (GPU)               |
| `Memory` | Кэш с TTL                           |
| `Mods`   | Декораторы / Registry               |
| `Proxy`  | Ленивая загрузка                    |

## GPU-first

Все численные операции по умолчанию выполняются на GPU через CuPy.
Если среда не имеет NVIDIA GPU — фреймворк автоматически,
без вмешательства пользователя, исполняет тот же код на CPU.

## Разработка

```bash
numfast test    # запуск тестов
numfast lint    # проверка кода
numfast build   # сборка
```

## Лицензия

MIT
