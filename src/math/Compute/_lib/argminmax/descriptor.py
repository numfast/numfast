"""ArgMin/ArgMax descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    op = params.get("op", "min")  # "min" or "max"
    return ExecutionPlan(
        inputs=[
            InputSlot(name="src", dtype="float"),
        ],
        outputs=[
            OutputSlot(dtype="float", template="arg_val"),
            OutputSlot(dtype="float", template="arg_idx"),
        ],
        workspace=[],
        uniforms={"op": op},
    )
