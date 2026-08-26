"""Combine descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    num_inputs = int(params.get("num_inputs", 2))
    inputs = [InputSlot(name=f"in{i}", dtype="float") for i in range(num_inputs)]
    return ExecutionPlan(
        inputs=inputs,
        outputs=[OutputSlot(dtype="float", template="combine_{num_inputs}")],
        workspace=[],
        uniforms={
            "num_inputs": float(num_inputs),
            "w0": float(params.get("w0", 1.0)),
            "w1": float(params.get("w1", 0.0)),
            "w2": float(params.get("w2", 0.0)),
            "w3": float(params.get("w3", 0.0)),
            "bias": float(params.get("bias", 0.0)),
        },
    )
