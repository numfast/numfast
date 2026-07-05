"""Reduce descriptor — сумма всех элементов."""

from numfast.Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="sum")],
        workspace=[
            BufferSpec(dtype="float", elements=0),  # partial sums per workgroup
        ],
        uniforms=dict(params),
    )
