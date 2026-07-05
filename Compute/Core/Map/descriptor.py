"""Map descriptor — apply element-wise function."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="map_{func}")],
        workspace=[],
        uniforms=dict(params),
    )
