"""Workspace — рабочее пространство.

Stateless. Не хранит состояние. Только фабрика для проектов.
"""

import os


class Workspace:
    """Рабочее пространство — корневая директория с проектами.

    Workspace — stateless фабрика для Project.
    """

    def __init__(self, path: str = "."):
        self._path = os.path.abspath(path)

    @property
    def path(self) -> str:
        """Абсолютный путь к рабочему пространству."""
        return self._path

    def open(self, name: str) -> "Project":
        """Открыть проект в workspace.

        Args:
            name: имя проекта (директория)

        Returns:
            Project — дескриптор проекта
        """
        from .project import Project
        project_path = os.path.join(self._path, name)
        return Project(project_path)

    def labs(self) -> list[dict]:
        """Найти все Lab в workspace.

        Returns:
            список {number, name, path}
        """
        labs = []
        labs_dir = os.path.join(self._path, "Labs")
        if not os.path.isdir(labs_dir):
            return labs
        for entry in sorted(os.listdir(labs_dir)):
            if entry.startswith("_"):
                parts = entry.split("_", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    labs.append({
                        "number": int(parts[1]),
                        "name": entry,
                        "path": os.path.join(labs_dir, entry),
                    })
        return labs

    def examples(self) -> list[dict]:
        """Найти все примеры в workspace.

        Returns:
            список {name, path}
        """
        examples = []
        examples_dir = os.path.join(self._path, "examples")
        if not os.path.isdir(examples_dir):
            return examples
        for entry in sorted(os.listdir(examples_dir)):
            if entry.endswith(".py") and not entry.startswith("__"):
                examples.append({
                    "name": entry[:-3],
                    "path": os.path.join(examples_dir, entry),
                })
        return examples


__all__ = ["Workspace"]
