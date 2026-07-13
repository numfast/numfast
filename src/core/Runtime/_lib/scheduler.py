"""Scheduler — анализ зависимостей, группировка в волны.

Scheduler получает список Task (после компиляции) и группирует их в волны.
Волна — набор независимых задач, которые можно выполнять параллельно.

Wave 1: задачи, которые читают только source data
Wave N: задачи, которые читают source data + выходы задач из волн < N
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from .task import Task, ResourceRef


@dataclass
class ScheduleWave:
    """Группа независимых ExecutionPackets.

    Все пакеты в волне не имеют зависимостей друг от друга.
    Driver может выполнять их параллельно.
    """
    packets: list["ExecutionPacket"] = field(default_factory=list)


class Scheduler:
    """Планировщик — группирует Task в волны по зависимостям.

    Использует топологический анализ: задача может войти в волну N,
    если все её зависимости (task refs) находятся в волнах < N.
    """

    def schedule(self, tasks: list[Task]) -> list[ScheduleWave]:
        """Сгруппировать задачи в волны.

        Args:
            tasks: список Task (результат компиляции)

        Returns:
            list[ScheduleWave] — волны для последовательного выполнения
        """
        if not tasks:
            return []

        task_wave: dict[int, int] = {}
        waves: list[ScheduleWave] = []

        for task in tasks:
            max_dep_wave = -1
            for ref in task.inputs:
                if ref.type == "task":
                    dep_wave = task_wave.get(ref.task_id, -1)
                    if dep_wave > max_dep_wave:
                        max_dep_wave = dep_wave

            target_wave = max_dep_wave + 1

            while len(waves) <= target_wave:
                waves.append(ScheduleWave())

            waves[target_wave].packets.append(task)
            task_wave[task.id] = target_wave

        return waves
