"""LogicalOr descriptor -- bitwise OR for uint32."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="a", dtype="uint32"),
            InputSlot(name="b", dtype="uint32"),
        ],
        outputs=[OutputSlot(dtype="uint32", template="logical_or")],
        workspace=[],
        uniforms={},
    )
