"""StateKernel descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    mode = int(params.get("mode", 0))
    params_out = dict(params)
    params_out["mode"] = float(mode)
    
    if mode == 0:
        return ExecutionPlan(
            inputs=[InputSlot(name="data", dtype="float")],
            outputs=[OutputSlot(dtype="float", template="ema")],
            workspace=[BufferSpec(dtype="float", elements=0)],
            uniforms=params_out,
        )
    elif mode == 1:
        return ExecutionPlan(
            inputs=[
                InputSlot(name="gain", dtype="float"),
                InputSlot(name="loss", dtype="float"),
            ],
            outputs=[
                OutputSlot(dtype="float", template="ema_gain"),
                OutputSlot(dtype="float", template="ema_loss"),
            ],
            workspace=[
                BufferSpec(dtype="float", elements=0),
                BufferSpec(dtype="float", elements=0),
            ],
            uniforms=params_out,
        )
    elif mode == 2:
        return ExecutionPlan(
            inputs=[
                InputSlot(name="close", dtype="float"),
                InputSlot(name="upper_band", dtype="float"),
                InputSlot(name="lower_band", dtype="float"),
            ],
            outputs=[OutputSlot(dtype="float", template="supertrend")],
            workspace=[
                BufferSpec(dtype="float", elements=0),  # prev_upper
                BufferSpec(dtype="float", elements=0),  # prev_lower
                BufferSpec(dtype="float", elements=0),  # prev_dir
            ],
            uniforms=params_out,
        )
    else:
        raise ValueError(f"Unknown StateKernel mode: {mode}")