"""Reduce descriptor — свёртка всех элементов: sum / min / max.

Параметр op:
  - строка: "sum" / "min" / "max" (default "sum") — CPU-путь;
  - числовой код: 0 = sum, 1 = min, 2 = max — GPU-путь.

Uniform "op" всегда числовой: WebGPU pack_uniforms пакует uniforms
как f32 и падает на строках (frozen builder передаёт node.params в
uniforms без нормализации). Строковый op принимается для имён выходов
(template) и CPU-диспетчеризации.

Выход — массив из одного элемента: output_size_fn = [1].
Dispatch задаётся builder'ом (дефолт), потоки id.x >= n отсечены guard
в WGSL, поэтому минимальный и максимальный коды корректны при любом n.

S62 (унифицировано, решение пользователя 2026-08-20): N=0 — ВСЕ op
(sum/min/max) → ValueError (guard здесь, до dispatch — output_size_fn
вызывается builder'ом для обоих бэкендов до создания пакетов; GPU
физически не может создать 0-byte buffer на frozen wgpu_driver, sum
0.0 нейтраль не реализуется — единая предсказуемая ошибка).
"""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan

_OP_NAMES = {0: "sum", 1: "min", 2: "max"}
_OP_CODES = {"sum": 0.0, "min": 1.0, "max": 2.0}

_EMPTY_MSG = (
    "Reduce on empty input (n=0) is undefined; "
    "use Fill or check input length")


def _output_size(op, sizes):
    """S62 guard: N=0 -> ValueError for ALL ops BEFORE dispatch.

    op может быть строкой ("sum"/"min"/"max") или числовым кодом
    (0/1/2, int или float); для числа используется _OP_NAMES.
    Вызывается builder'ом (builder.py output_size_fn) на ОБОИХ бэкендах
    — единый путь, до создания пакетов (GPU не сабмитится).
    """
    n = sizes[0]
    if n == 0:
        raise ValueError(_EMPTY_MSG)
    return [1]


def describe(params: dict) -> ExecutionPlan:
    op = params.get("op", "sum")
    if isinstance(op, (int, float)):
        name = _OP_NAMES.get(int(op), "sum")
        code = float(op)
    else:
        name = op if op in _OP_CODES else "sum"
        code = _OP_CODES.get(name, 0.0)

    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template=name)],
        workspace=[
            BufferSpec(dtype="float", elements=0),  # partial per thread
        ],
        uniforms={"op": code},
        output_size_fn=lambda sizes: _output_size(op, sizes),
    )
