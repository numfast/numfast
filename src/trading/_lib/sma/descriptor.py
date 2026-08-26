"""SMA descriptor — декларация требований ядра."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[InputSlot(name="Close", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="sma_{period}")],
        workspace=[],
        uniforms=dict(params),
    )
