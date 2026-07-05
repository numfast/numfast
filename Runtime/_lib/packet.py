"""ExecutionPacket — готовый пакет для Driver. Аналог GPU Command.

Dispatcher (Runtime) превращает Task (логический) в ExecutionPacket (физический):
- ResourceRef → готовые BufferView
- resource/out_names → output_buffers
- params → uniforms (с проверкой типов)

Driver получает ExecutionPacket и вызывает kernel["drivers"]["cpu"](packet).
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BufferView:
    """Ссылка на готовый блок памяти.

    Содержит только то, что нужно Driver:
    - view: BlockView (доступ read/write/length)
    - dtype: str
    - size: int
    """
    view: Any       # BlockView
    dtype: str = "float"
    size: int = 0


@dataclass
class ExecutionPacket:
    """Полностью подготовленный пакет для выполнения ядром.

    Driver не знает Task, ResourceRef, имена выходов, ресурсы.
    Driver получает только ExecutionPacket.
    """
    kernel: str                        # alias ("EMA")
    input_buffers: list[BufferView]    # готовые входы
    output_buffers: list[BufferView]   # готовые выходы
    workspace_buffers: list[BufferView] = field(default_factory=list)
    uniforms: dict = field(default_factory=dict)          # только int/float/bool
    bindings: list[BufferView] = field(default_factory=list)  # все ресурсы единым списком
    dispatch: tuple = (1, 1, 1)        # для WebGPU (workgroup count)
    abi_version: int = 1               # версия ABI ядра

    def validate(self):
        """Валидация packet перед выполнением."""
        from .mod_iface import validate_uniforms
        validate_uniforms(self.uniforms)
