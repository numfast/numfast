"""Keltner Channels descriptor."""

from numfast.Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="High", dtype="float"),
            InputSlot(name="Low", dtype="float"),
            InputSlot(name="Close", dtype="float"),
        ],
        outputs=[
            OutputSlot(dtype="float", template="kc_middle_{period_ema}"),
            OutputSlot(dtype="float", template="kc_upper_{period_ema}_{period_atr}"),
            OutputSlot(dtype="float", template="kc_lower_{period_ema}_{period_atr}"),
        ],
        workspace=[
            BufferSpec(dtype="float", elements=0),  # prev_EMA
            BufferSpec(dtype="float", elements=0),  # prev_ATR
            BufferSpec(dtype="float", elements=0),  # prev_close for ATR
        ],
        uniforms=dict(params),
    )
