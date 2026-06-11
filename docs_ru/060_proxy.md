# Proxy — ленивая загрузка

## Описание

Модуль `Proxy` предоставляет механизм ленивой загрузки модулей.
Не зависит от CuPy — чистый Python.

## Функции

### `load(module_name) -> module`

Загружает модуль по имени. При повторном вызове возвращает закэшированный модуль.

```python
cp = exts.Proxy.load("cupy")
cp.array([1, 2, 3])
```

### `info() -> dict`

Возвращает информацию о загруженных модулях.

```python
exts.Proxy.info()
# {"loaded_modules": ["cupy"]}
```

## Внутренняя реализация

Вся логика в `Proxy/_lib/proxy_lib.py`:

- `_LazyLoader` — класс с кэшем модулей и `importlib.import_module`
