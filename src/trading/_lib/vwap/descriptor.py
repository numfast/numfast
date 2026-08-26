"""VWAP descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="High", dtype="float"),
            InputSlot(name="Low", dtype="float"),
            InputSlot(name="Close", dtype="float"),
            InputSlot(name="Volume", dtype="float"),
        ],
        outputs=[OutputSlot(dtype="float", template="vwap")],
        workspace=[BufferSpec(dtype="float", elements=0)],  # cum_vol buffer
        uniforms=dict(params),
    )
