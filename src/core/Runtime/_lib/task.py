"""Task — датакласс результата компиляции.

После компиляции никаких строк зависимостей не остаётся.
Dispatcher видит только Task.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .mod_iface import BufferSpec


@dataclass
class ResourceRef:
    type: str
    column: str = ""
    task_id: int = -1
    output_idx: int = 0


@dataclass
class Task:
    id: int
    op: str
    params: dict
    inputs: list[ResourceRef]
    resource: str
    num_outputs: int = 1
    out_names: list[str] = field(default_factory=list)
    workspace: list = field(default_factory=list)
    uniforms: dict[str, Any] = field(default_factory=dict)
    output_size_fn: Optional[Callable] = None  # callable[[list[int]], list[int]] or None
    dispatch: Optional[tuple] = None          # (dx, dy, dz) — override dispatch size
