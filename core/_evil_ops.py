"""Злонамеренные операции для Phase 16 — "попытка сломать архитектуру".

Эти операции написаны для стресс-тестирования Runtime.
Они НЕ предназначены для реального использования.

Проверяют теорему 6: операция не может нарушить Runtime,
даже если она намеренно неправильная.
"""

from typing import List, Optional

from numfast.core.nseries import NumericSeries
from numfast.core.driver import CpuDriver
from numfast.core.executor import Job


def install_evil_ops(kernel) -> None:
    """Установить все злонамеренные операции в CpuKernel.

    Пример:
        kernel = CpuKernel()
        install_evil_ops(kernel)
    """
    kernel._ops["always_zero"] = _run_always_zero
    kernel._ops["always_negative"] = _run_always_negative
    kernel._ops["random_output"] = _run_random
    kernel._ops["throws_exception"] = _run_throws
    kernel._ops["manifest"] = _run_manifest


def _run_always_zero(job: Job) -> None:
    """Всегда пишет 0 — нарушает Storage Invariant (min ≥ 1)."""
    import numpy as np
    n = job.inputs[0].n
    for inp in job.inputs:
        pass
    out = job.driver.allocate(n, valid_from=0)
    job.outputs.append(out)
    for i in range(n):
        job.driver.write(out, i, 0)  # <-- нарушение! min=0


def _run_always_negative(job: Job) -> None:
    """Всегда пишет отрицательные значения."""
    n = job.inputs[0].n
    out = job.driver.allocate(n, valid_from=0)
    job.outputs.append(out)
    for i in range(n):
        job.driver.write(out, i, -i - 1)  # <-- отрицательные


def _run_random(job: Job) -> None:
    """Пишет случайные значения — непредсказуемый результат."""
    import random
    n = job.inputs[0].n
    out = job.driver.allocate(n, valid_from=0)
    job.outputs.append(out)
    for i in range(n):
        job.driver.write(out, i, random.randint(0, 1000000))


def _run_throws(job: Job) -> None:
    """Бросает исключение во время выполнения."""
    raise RuntimeError("Преднамеренное исключение из evil-операции")


def _run_manifest(job: Job) -> None:
    """Пытается получить прямой доступ к внутренностям серии (нарушение I2).

    Проверяет, что Runtime не даёт операции сломать себя."""
    # Попытка 1: прямой доступ к _values
    inp = job.inputs[0]
    try:
        inp._values[0] = 999999
    except (AttributeError, TypeError):
        pass  # защита есть

    # Попытка 2: мутация метаданных
    try:
        meta = inp.meta
        meta.offset = -1000
    except (AttributeError, TypeError):
        pass

    # Попытка 3: запись в input
    try:
        inp.write(0, 0)
    except Exception:
        pass

    # Всё равно делаем валидную запись в output
    n = inp.n
    out = job.driver.allocate(n, valid_from=0)
    job.outputs.append(out)
    for i in range(n):
        val = job.driver.read(inp, i)
        job.driver.write(out, i, val)
