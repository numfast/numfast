"""Williams %R descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    return ExecutionPlan(
        inputs=[
            InputSlot(name="High", dtype="float"),
            InputSlot(name="Low", dtype="float"),
            InputSlot(name="Close", dtype="float"),
        ],
        outputs=[OutputSlot(dtype="float", template="williams_r_{period}")],
        workspace=[],
        uniforms=dict(params),
    )
