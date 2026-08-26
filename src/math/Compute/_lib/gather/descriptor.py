"""Gather descriptor.

out[i] = src[index[i]]; оба входа — float (index хранится как f32,
WGSL делает i32(index[i])). CPU-оракул принимает любые числовые индексы
(int(idx.read(i))).

D-5 GAP (int32 индексный буфер, решение владельца — зафиксировано):
GPU-путь не поддерживает нативный array<i32> индексный буфер. Причина:
драйвер (FROZEN wgpu_driver._get_packed_info) сериализует ВСЕ входы ядра
единым caps["input_dtype"] — смешение float-src + int32-index в одном
kernel невозможно без разморозки Drivers. Отдельный gather_i32 тоже не
решает: у него src должен остаться f32. Следствие: GPU-gather по индексам
>= 2^24 неточен (f32); на CPU точен до 2^31. Устранение — после unfreeze
драйвера (per-slot dtype).
"""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="src", dtype="float"),
            InputSlot(name="index", dtype="float"),  # D-5: int32 planned
        ],
        outputs=[OutputSlot(dtype="float", template="gather")],
        workspace=[],
        uniforms={},
    )
