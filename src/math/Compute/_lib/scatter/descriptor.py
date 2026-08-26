"""Scatter descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="value", dtype="float"),
            InputSlot(name="index", dtype="float"),
        ],
        outputs=[OutputSlot(dtype="float", template="scatter")],
        workspace=[],
        uniforms={},
    )
