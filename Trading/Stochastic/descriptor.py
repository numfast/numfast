"""Stochastic descriptor."""

from numfast.Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="High", dtype="float"),
            InputSlot(name="Low", dtype="float"),
            InputSlot(name="Close", dtype="float"),
        ],
        outputs=[
            OutputSlot(dtype="float", template="stoch_k_{period_k}"),
            OutputSlot(dtype="float", template="stoch_d_{period_k}_{period_d}"),
        ],
        workspace=[
            BufferSpec(dtype="float", elements=0),  # raw %K buffer for %D smoothing
        ],
        uniforms=dict(params),
    )
