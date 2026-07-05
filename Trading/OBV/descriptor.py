"""OBV descriptor."""

from numfast.Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="Close", dtype="float"),
            InputSlot(name="Volume", dtype="float"),
        ],
        outputs=[OutputSlot(dtype="float", template="obv")],
        workspace=[BufferSpec(dtype="float", elements=0)],  # cumulative OBV
        uniforms=dict(params),
    )
