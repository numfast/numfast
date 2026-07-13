"""MapScalar descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="map_scalar_{op}")],
        workspace=[],
        uniforms={
            "op": float(params.get("op", 0)),
            "scalar": float(params.get("scalar", 0.0)),
        },
    )
