"""Rolling StdDev descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    p = int(params["period"])
    if p < 1:
        raise ValueError(f"RollingStdDev period must be >= 1; got {p}")
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template=f"rolling_std_{p}")],
        workspace=[],
        uniforms={"period": float(p)},
    )
