"""Agent SDK — программный слой между пользователем и платформой NumFast.

AI-agnostic. Не зависит от LLM, API ключей, промптов.
"""

from dataclasses import dataclass, field
from typing import Any

from .session import Session
from .workspace import Workspace
from .project import Project


@dataclass
class Result:
    """Результат выполнения инструмента.

    Все инструменты возвращают Result. Никаких исключений — ошибки в поле error.
    """
    status: str
    data: Any = None
    error: str = ""
    duration: float = 0.0
    metadata: dict = field(default_factory=dict)


class AgentSDK:
    """Точка входа в Agent SDK.

    Stateless — не хранит состояние проекта.
    Всё состояние — только в Session.
    """

    @staticmethod
    def workspace(path: str) -> Workspace:
        return Workspace(path)

    @staticmethod
    def session(use_gpu: bool = False) -> Session:
        return Session(use_gpu=use_gpu)


__all__ = ['AgentSDK', 'Result', 'Session', 'Workspace', 'Project']
