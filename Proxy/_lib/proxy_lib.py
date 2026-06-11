import importlib

class _LazyLoader:
    def __init__(self):
        self._cache: dict[str, object] = {}

    def load(self, module_name: str):
        if module_name not in self._cache:
            self._cache[module_name] = importlib.import_module(module_name)
        return self._cache[module_name]
