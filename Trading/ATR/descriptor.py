"""ATR descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="High", dtype="float"),
            InputSlot(name="Low", dtype="float"),
            InputSlot(name="Close", dtype="float"),
        ],
        outputs=[OutputSlot(dtype="float", template="atr_{period}")],
        workspace=[BufferSpec(dtype="float", elements=0)],  # prev_close buffer
        uniforms=dict(params),
    )
