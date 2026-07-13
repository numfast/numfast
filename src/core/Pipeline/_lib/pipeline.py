"""Pipeline — список задач (Task) для последовательного выполнения.

Pipeline НЕ строит граф. Нет DAG, нет узлов, нет топологической сортировки.
Pipeline — это просто список задач, которые Dispatcher выполняет по порядку.

Два API:
    1. pipe.add(op, *args, **kwargs) — интерактивное добавление
    2. Pipeline(jobs) — пакетное создание из списка кортежей
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Task:
    """Одна задача в пайплайне.
    
    op: callable с методами .compute(source, ...) и .make_name(...)
    params: словарь параметров (args, kwargs)
    name: имя колонки результата (напр. "EMA_20")
    """
    op: Callable
    params: dict = field(default_factory=dict)
    name: str = ""


class Pipeline:
    """Пайплайн — список задач для последовательного выполнения.
    
    Примеры:
        # Сахар: добавляем по одному
        pipe = Pipeline()
        pipe.add(EMA, 20)
        pipe.add(RSI, 14)
        result = pipe.compute(source_data)
        
        # Пакетный: список кортежей
        jobs = [(EMA, 20), (RSI, 14)]
        pipe = Pipeline(jobs)
        result = pipe.compute(source_data)
    """
    
    def __init__(self, jobs=None):
        self._tasks = []  # list of Task
        if jobs:
            for job in jobs:
                if isinstance(job, tuple):
                    op, *args = job
                    params = {"args": args, "kwargs": {}}
                elif isinstance(job, dict):
                    op = job["op"]
                    params = job.get("params", {})
                else:
                    raise TypeError(f"Expected tuple or dict, got {type(job)}")
                self._add_task(op, params)
    
    def add(self, op, *args, **kwargs):
        """Добавить задачу.
        
        Args:
            op: callable с .compute(source, ...) и .make_name(...)
            *args: позиционные параметры (period и т.д.)
            **kwargs: именованные параметры
        """
        params = {"args": args, "kwargs": kwargs}
        self._add_task(op, params)
    
    def _add_task(self, op, params):
        args = params.get("args", ())
        kwargs = params.get("kwargs", {})
        name = op.make_name(*args, **kwargs)
        self._tasks.append(Task(op=op, params=params, name=name))
    
    @property
    def tasks(self):
        return list(self._tasks)
    
    def __len__(self):
        return len(self._tasks)
    
    def __repr__(self):
        return f"Pipeline({len(self._tasks)} tasks)"
    
    def compute(self, source):
        """Выполнить все задачи через Dispatcher.
        
        Args:
            source: SeriesProxy, Table, или dict с данными
        
        Returns:
            Table с колонками результатов
        """
        from .dispatcher import Dispatcher
        dispatcher = Dispatcher()
        return dispatcher.execute(self._tasks, source)
