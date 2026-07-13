"""Mod interface — контракт между модами и Runtime.

Никаких классов с методами.
Мод — это функция describe(params) → ExecutionPlan.
Ядро (kernel) — функция cpu(ctx).

Всё регистрируется в context.kernel_table как alias → {describe, drivers: {cpu, wgsl, cuda}}.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class BlockView:
    """Абстрактный блок памяти. ABI: read(i), write(i, v), length().

    Ядро НЕ использует квадратные скобки.
    Только view.read(i), view.write(i, v), view.length().
    Это позволяет транслировать ядро в WGSL/CUDA без парсинга Python.
    """
    __slots__ = ('_data', '_dtype', '_size')

    def __init__(self, data, dtype="float", size=0):
        self._data = data
        self._dtype = dtype
        self._size = size if size > 0 else (len(data) if hasattr(data, '__len__') else 0)

    def read(self, idx) -> float:
        """Прочитать элемент по индексу."""
        return self._data[idx]

    def write(self, idx, value):
        """Записать элемент по индексу."""
        self._data[idx] = value

    def length(self) -> int:
        """Количество элементов."""
        return self._size

    # Внутреннее (не для ядер):
    @property
    def _raw(self):
        return self._data


@dataclass
class InputSlot:
    """Описание одного входа."""
    name: str       # логическое имя ("Close", "Volume")
    dtype: str      # "float", "int"


@dataclass
class OutputSlot:
    """Описание одного выхода."""
    dtype: str      # "float", "int"
    template: str   # "ema_{period}" — шаблон имени


@dataclass
class BufferSpec:
    """Спецификация буфера памяти для ядра."""
    dtype: str = "float"
    elements: int = 0


@dataclass
class ExecutionPlan:
    """План выполнения — результат вызова describe(params).

    Живёт только внутри компиляции.
    После формирования Task — уничтожается.
    """
    inputs: list[InputSlot] = field(default_factory=list)
    outputs: list[OutputSlot] = field(default_factory=list)
    workspace: list[BufferSpec] = field(default_factory=list)
    uniforms: dict = field(default_factory=dict)        # конкретные параметры
    output_size_fn: Optional[Callable] = None
    dispatch: Optional[tuple] = None          # (dx, dy, dz) — override workgroup count
    # output_size_fn: Callable[[list[int]], list[int]]
    #   Принимает input_sizes, возвращает output_sizes.
    #   Если None — дефолт: все output = input_sizes[0]


@dataclass
class Buffer:
    """Буфер данных для ExecutionContext.

    Ядро получает доступ через buf.view.read(i) / buf.view.write(i, v).
    """
    view: BlockView
    dtype: str = "float"
    size: int = 0

    def __post_init__(self):
        if isinstance(self.view, BlockView):
            if self.size == 0:
                self.size = self.view.length()
        else:
            self.view = BlockView(self.view, self.dtype, self.size)


@dataclass
class ExecutionContext:
    """Контекст выполнения одной операции.

    Предоставляется Driver'ом ядру.
    Все буферы уже выделены и готовы к чтению/записи.
    """
    inputs: list[Buffer] = field(default_factory=list)
    outputs: list[Buffer] = field(default_factory=list)
    workspace: list[Buffer] = field(default_factory=list)
    uniforms: dict = field(default_factory=dict)  # неизменяемые параметры


# Типы для kernel_table
DescribeFn = Callable[[dict], ExecutionPlan]    # (params) → ExecutionPlan
KernelFn = Callable[[ExecutionContext], None]    # (ctx) → None

# Структура записи в kernel_table
# {describe: DescribeFn, drivers: {cpu: KernelFn, wgsl: str | None, cuda: str | None}}
KernelRecord = dict  # с ключами describe, drivers


_UNIFORM_TYPES = (int, float, bool)


def validate_uniforms(uniforms: dict):
    """Проверить, что все uniforms — int, float или bool.

    Raises:
        TypeError: если найден недопустимый тип.
    """
    for k, v in uniforms.items():
        if type(v) not in _UNIFORM_TYPES:
            raise TypeError(
                f"Uniform '{k}' has type '{type(v).__name__}', "
                f"expected int, float, or bool"
            )


class KernelValidator:
    """Проверяет ExecutionPacket против ожиданий ядра (describe).

    Вызывается Driver перед выполнением.
    Аналог Vulkan Validation Layers.
    """

    @staticmethod
    def validate(packet: "ExecutionPacket", plan: ExecutionPlan):
        """Проверить packet на соответствие плану.

        Args:
            packet: ExecutionPacket — готовый пакет
            plan: ExecutionPlan — результат describe()

        Raises:
            AssertionError: если проверка не пройдена
        """
        # Input count
        assert len(packet.input_buffers) == len(plan.inputs), \
            f"Expected {len(plan.inputs)} inputs, got {len(packet.input_buffers)}"

        # Output count
        assert len(packet.output_buffers) == len(plan.outputs), \
            f"Expected {len(plan.outputs)} outputs, got {len(packet.output_buffers)}"

        # Workspace count & types
        assert len(packet.workspace_buffers) == len(plan.workspace), \
            f"Expected {len(plan.workspace)} workspace buffers, got {len(packet.workspace_buffers)}"

        for i, spec in enumerate(plan.workspace):
            assert packet.workspace_buffers[i].dtype == spec.dtype, \
                f"Workspace[{i}] dtype: expected {spec.dtype}, got {packet.workspace_buffers[i].dtype}"

        # Uniforms types
        validate_uniforms(packet.uniforms)
