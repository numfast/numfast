"""MACD descriptor — декларация требований ядра."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params):
    return ExecutionPlan(
        inputs=[InputSlot(name="Close", dtype="float")],
        outputs=[
            OutputSlot(dtype="float", template="macd_{fast}_{slow}"),
            OutputSlot(dtype="float", template="signal_{fast}_{slow}"),
            OutputSlot(dtype="float", template="hist_{fast}_{slow}"),
        ],
        workspace=[
            BufferSpec(dtype="float", elements=0),
            BufferSpec(dtype="float", elements=0),
        ],
        uniforms=dict(params),
    )
