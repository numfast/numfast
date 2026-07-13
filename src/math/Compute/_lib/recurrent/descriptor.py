"""Recurrent descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    mode = int(params.get("mode", 0))
    if mode == 0:
        # Single: 1 input, 1 output, 1 workspace (state)
        return ExecutionPlan(
            inputs=[InputSlot(name="data", dtype="float")],
            outputs=[OutputSlot(dtype="float", template="recurrent")],
            workspace=[BufferSpec(dtype="float", elements=0)],
            uniforms={
                "mode": float(mode),
                "a": float(params.get("a", 1.0)),
                "b": float(params.get("b", 0.0)),
            },
        )
    elif mode == 1:
        # Dual: 2 inputs, 2 outputs, 2 workspace (states)
        return ExecutionPlan(
            inputs=[
                InputSlot(name="in0", dtype="float"),
                InputSlot(name="in1", dtype="float"),
            ],
            outputs=[
                OutputSlot(dtype="float", template="recurrent_0"),
                OutputSlot(dtype="float", template="recurrent_1"),
            ],
            workspace=[
                BufferSpec(dtype="float", elements=0),
                BufferSpec(dtype="float", elements=0),
            ],
            uniforms={
                "mode": float(mode),
                "a": float(params.get("a", 1.0)),
                "b": float(params.get("b", 0.0)),
            },
        )
    else:
        raise ValueError(f"Unknown Recurrent mode: {mode}")
