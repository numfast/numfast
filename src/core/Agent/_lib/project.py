"""Project — дескриптор проекта.

Stateless. Не хранит состояние. Только идентифицирует проект.
"""

import os


class Project:
    """Проект — директория с кодом NumFast.

    Project — stateless дескриптор. Не хранит Runtime, не кэширует.
    """

    def __init__(self, path: str):
        self._path = os.path.abspath(path)

    @property
    def path(self) -> str:
        """Абсолютный путь к проекту."""
        return self._path

    @property
    def name(self) -> str:
        """Имя проекта (последний компонент пути)."""
        return os.path.basename(self._path)

    @property
    def exists(self) -> bool:
        """Существует ли директория проекта."""
        return os.path.isdir(self._path)

    def __repr__(self) -> str:
        return f"Project('{self.name}')"


__all__ = ["Project"]
