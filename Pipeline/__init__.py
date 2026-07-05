"""Pipeline — список задач для последовательного выполнения.

Pipeline хранит Task'и (операция + параметры).
Dispatcher выполняет их по порядку.
Результат — одна Table.
"""

from ._lib.pipeline import Pipeline, Task

__all__ = ["Pipeline", "Task"]
