"""TrueRange descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="high", dtype="float"),
            InputSlot(name="low", dtype="float"),
            InputSlot(name="close", dtype="float"),
        ],
        outputs=[OutputSlot(dtype="float", template="true_range")],
        workspace=[],
        uniforms={},
    )