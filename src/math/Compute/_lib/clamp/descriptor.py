"""Clamp descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="clamp")],
        workspace=[],
        uniforms={
            "min_val": float(params.get("min_val", 0.0)),
            "max_val": float(params.get("max_val", 1.0)),
        },
    )
