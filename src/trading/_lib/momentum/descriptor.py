"""Momentum descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[InputSlot(name="Close", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="mom_{period}")],
        workspace=[],
        uniforms=dict(params),
    )
