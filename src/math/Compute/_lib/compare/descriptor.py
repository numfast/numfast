"""Compare descriptor.

Input count depends on scalar mode:
  0 inputs: both operands are scalar (degenerate, always compares uniforms)
  1 input:  one operand is scalar (use_scalar_a or use_scalar_b)
  2 inputs: both operands are arrays
"""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    op = params.get("op", "gt")
    inputs = []
    if not float(params.get("use_scalar_a", 0)):
        inputs.append(InputSlot(name="a", dtype="float"))
    if not float(params.get("use_scalar_b", 0)):
        inputs.append(InputSlot(name="b", dtype="float"))

    return ExecutionPlan(
        inputs=inputs,
        outputs=[OutputSlot(dtype="uint32", template="compare")],
        workspace=[],
        uniforms={
            "op": op,
            "scalar_a": float(params.get("scalar_a", 0.0)),
            "scalar_b": float(params.get("scalar_b", 0.0)),
            "use_scalar_a": float(params.get("use_scalar_a", 0)),
            "use_scalar_b": float(params.get("use_scalar_b", 0)),
        },
    )
