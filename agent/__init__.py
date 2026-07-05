"""Agent SDK — программный слой между пользователем и платформой NumFast.

AI-agnostic. Не зависит от LLM, API ключей, промптов.

Usage:
    from agent import AgentSDK, Result

    sdk = AgentSDK()

    with sdk.session() as session:
        result = session.run("examples/moving_average.py")
        print(result.status, result.duration)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from .session import Session
from .workspace import Workspace
from .project import Project


@dataclass
class Result:
    """Результат выполнения инструмента.

    Все инструменты возвращают Result. Никаких исключений — ошибки в поле error.
    """
    status: str                    # "ok" | "error" | "skipped"
    data: Any = None               # произвольные данные результата
    error: str = ""                # сообщение об ошибке (если status == "error")
    duration: float = 0.0          # время выполнения в секундах
    metadata: dict = field(default_factory=dict)  # driver, compile_time, dispatch_time, ...


class AgentSDK:
    """Точка входа в Agent SDK.

    Stateless — не хранит состояние проекта.
    Всё состояние — только в Session.
    """

    @staticmethod
    def workspace(path: str = ".") -> Workspace:
        """Открыть рабочее пространство.

        Args:
            path: корневая директория (проект NumFast или любой проект)

        Returns:
            Workspace — фабрика для открытия проектов
        """
        return Workspace(path)

    @staticmethod
    def session(use_gpu: bool = False) -> Session:
        """Создать новую сессию.

        Сессия — эксклюзивный владелец Runtime.
        Один Runtime принадлежит только одной сессии в один момент времени.

        Args:
            use_gpu: использовать WebGPU если True

        Returns:
            Session — контекстный менеджер, освобождает Runtime при выходе
        """
        return Session(use_gpu=use_gpu)


__all__ = ["AgentSDK", "Result", "Session", "Workspace", "Project"]
