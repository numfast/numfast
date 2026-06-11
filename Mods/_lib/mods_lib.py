from collections.abc import Callable


class _Registry:
    def __init__(self):
        self._items: dict[str, Callable] = {}

    def add(self, name: str, func: Callable):
        self._items[name] = func

    def get(self, name: str) -> Callable | None:
        return self._items.get(name)

    def items(self):
        return self._items.items()
