"""RSI descriptor — декларация требований ядра."""

from numfast.Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[InputSlot(name="Close", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="rsi_{period}")],
        workspace=[],
        uniforms=dict(params),
    )
