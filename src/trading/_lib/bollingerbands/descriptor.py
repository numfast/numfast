"""Bollinger Bands descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[InputSlot(name="Close", dtype="float")],
        outputs=[
            OutputSlot(dtype="float", template="bb_middle_{period}"),
            OutputSlot(dtype="float", template="bb_upper_{period}"),
            OutputSlot(dtype="float", template="bb_lower_{period}"),
        ],
        workspace=[
            BufferSpec(dtype="float", elements=0),  # SMA buffer
            BufferSpec(dtype="float", elements=0),  # sum buffer
        ],
        uniforms=dict(params),
    )
