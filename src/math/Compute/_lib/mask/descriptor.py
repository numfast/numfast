"""Mask descriptor.

cond is always array (uint32). a and b support scalar mode.
"""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    inputs = [InputSlot(name="cond", dtype="uint32")]
    if not float(params.get("use_scalar_a", 0)):
        inputs.append(InputSlot(name="a", dtype="float"))
    if not float(params.get("use_scalar_b", 0)):
        inputs.append(InputSlot(name="b", dtype="float"))

    return ExecutionPlan(
        inputs=inputs,
        outputs=[OutputSlot(dtype="float", template="mask")],
        workspace=[],
        uniforms={
            "scalar_a": float(params.get("scalar_a", 0.0)),
            "scalar_b": float(params.get("scalar_b", 0.0)),
            "use_scalar_a": float(params.get("use_scalar_a", 0)),
            "use_scalar_b": float(params.get("use_scalar_b", 0)),
        },
    )