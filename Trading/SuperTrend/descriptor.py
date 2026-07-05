"""SuperTrend descriptor."""

from numfast.Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="High", dtype="float"),
            InputSlot(name="Low", dtype="float"),
            InputSlot(name="Close", dtype="float"),
        ],
        outputs=[
            OutputSlot(dtype="float", template="supertrend_{period}_{mult}"),
            OutputSlot(dtype="float", template="supertrend_dir_{period}_{mult}"),
        ],
        workspace=[
            BufferSpec(dtype="float", elements=0),  # prev_close for ATR
            BufferSpec(dtype="float", elements=0),  # ATR buffer
            BufferSpec(dtype="float", elements=0),  # prev final upper band
            BufferSpec(dtype="float", elements=0),  # prev final lower band
            BufferSpec(dtype="float", elements=0),  # prev direction (1/-1)
        ],
        uniforms=dict(params),
    )
