# Mods — декораторы и Registry

## Описание

Модуль `Mods` предоставляет декораторы для регистрации функций в реестре.
Не зависит от CuPy — чистый Python.

## Функции

### `register(name=None)`

Декоратор, регистрирующий функцию в реестре.

```python
@exts.Mods.register("my_func")
def my_function():
    return 42
```

### `get_registered() -> dict`

Возвращает словарь зарегистрированных функций.

```python
exts.Mods.get_registered()
# {"my_func": <function my_function>}
```

## Внутренняя реализация

Вся логика в `Mods/_lib/mods_lib.py`:

- `_Registry` — класс с методами `add`, `get`, `items`
