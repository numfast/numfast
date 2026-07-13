"""MapBinary descriptor — element-wise binary ops.

Input count depends on scalar mode: 0, 1, or 2 array inputs.
Supports: + - * / max min
"""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    inputs = []
    if not float(params.get("use_scalar_a", 0)):
        inputs.append(InputSlot(name="a", dtype="float"))
    if not float(params.get("use_scalar_b", 0)):
        inputs.append(InputSlot(name="b", dtype="float"))

    return ExecutionPlan(
        inputs=inputs,
        outputs=[OutputSlot(dtype="float", template="map_binary_{op}")],
        workspace=[],
        uniforms={
            "op": float(params.get("op", 0)),
            "scalar_a": float(params.get("scalar_a", 0.0)),
            "scalar_b": float(params.get("scalar_b", 0.0)),
            "use_scalar_a": float(params.get("use_scalar_a", 0)),
            "use_scalar_b": float(params.get("use_scalar_b", 0)),
        },
    )
