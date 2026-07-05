"""CCI descriptor."""

from numfast.Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="High", dtype="float"),
            InputSlot(name="Low", dtype="float"),
            InputSlot(name="Close", dtype="float"),
        ],
        outputs=[OutputSlot(dtype="float", template="cci_{period}")],
        workspace=[
            BufferSpec(dtype="float", elements=0),  # TP buffer
            BufferSpec(dtype="float", elements=0),  # SMA buffer
        ],
        uniforms=dict(params),
    )
