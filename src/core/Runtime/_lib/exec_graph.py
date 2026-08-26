"""Execution Graph IR — чистое промежуточное представление.

IR = данные, не поведение.
Все операции над IR — чистые функции вне dataclass'ов.

Execution Graph НЕ знает про AST, Driver, Buffer.
Node не содержит методов execute() / compile() / optimize().

Правила:
- Dataclass'ы содержат ТОЛЬКО поля
- Никаких методов ни на одном dataclass'е
- Все операции: отдельные чистые функции
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PortRef:
    """Ссылка на входные данные Node.

    source:
        "@input" — внешний вход графа (сырая колонка)
        int      — node_id другого Node в графе
    port:
        int      — индекс выхода Node (для source=int)
        str      — имя колонки (для source="@input")
    dtype:
        "f32", "i32" — тип данных
    """
    source: str | int
    port: int | str
    dtype: str = "f32"


@dataclass
class OutputSpec:
    """Описание одного выхода Node.

    Поддерживает multi-output:
    - ArgMinMax -> val + idx (2 выхода)
    - Scan -> prefix + workgroup_totals (2 выхода)

    size_fn: формула размера выхода (str или None).
        None = same as input (default)
        "input_div_workgroup" = input_size / workgroup_size
    """
    name: str
    dtype: str = "f32"
    size_fn: str | None = None


@dataclass
class DataPort:
    """Внешний порт графа — вход или выход.

    Для inputs: достаточно name + dtype.
    Для outputs: node_id + port указывают на производителя.
    """
    name: str
    dtype: str = "f32"
    node_id: int | None = None
    port: int | None = None


@dataclass
class Node:
    """Один узел графа — один kernel dispatch.

    kernel_id: alias в kernel_table ("RollingSum", "StateKernel", ...)
    params: dict[str, int | float | bool | str] — только сериализуемые типы
    dispatch: (dx, dy, dz) или None (auto)
    """
    id: int
    kernel_id: str
    inputs: list[PortRef]
    outputs: list[OutputSpec]
    params: dict = field(default_factory=dict)
    dispatch: tuple | None = None


@dataclass
class ExecutionGraph:
    """Корневой контейнер Execution Graph IR.

    Содержит:
    - nodes: все операции DAG (topological order)
    - inputs: внешние входы (сырые колонки)
    - outputs: внешние выходы (результаты)
    - metadata: version, source, build_time

    IR не имеет методов.
    Все операции над ExecutionGraph — чистые функции.
    """
    nodes: list[Node]
    inputs: list[DataPort]
    outputs: list[DataPort]
    metadata: dict = field(default_factory=dict)
