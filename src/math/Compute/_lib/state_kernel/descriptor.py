"""StateKernel descriptor."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, BufferSpec, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    mode = int(params.get("mode", 0))
    params_out = dict(params)
    params_out["mode"] = float(mode)
    # S87: float-normalization of a/b (defaults match cpu.py: a=1.0, b=0.0).
    # NaN/Inf a/b are NOT validated (IEEE propagate, contract sec.8).
    params_out["a"] = float(params.get("a", 1.0))
    params_out["b"] = float(params.get("b", 0.0))

    if mode == 0:
        return ExecutionPlan(
            inputs=[InputSlot(name="data", dtype="float")],
            outputs=[OutputSlot(dtype="float", template="ema")],
            workspace=[BufferSpec(dtype="float", elements=0)],
            uniforms=params_out,
            # S87: serial kernels (workgroup_size(1)) run the FULL-array
            # loop per invocation -> dispatch=(1,1,1) (plan.dispatch ABI
            # hook, mod_iface.py:79; builder.py:132-133 takes plan.dispatch
            # before the fallback dx=ceil(n/64)). Effect: O(dx*n) -> O(n),
            # serial NOT under _check_dispatch_limit (dx=1) -> n > 4_194_240
            # works. WGSL serial kernels unchanged (correct for one
            # invocation; n==0 guard in SINGLE, empty loop safe in
            # DUAL/SUPERTREND).
            dispatch=(1, 1, 1),
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
            dispatch=(1, 1, 1),
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
            dispatch=(1, 1, 1),
        )
    else:
        raise ValueError(f"Unknown StateKernel mode: {mode}")


# ---------------------------------------------------------------------------
# Parallel 3-pass scan (mode 0 SINGLE / mode 1 DUAL)
# ---------------------------------------------------------------------------

def _scan_uniforms(params: dict) -> dict:
    mode = int(params.get("mode", 0))
    return {
        "mode": float(mode),
        "a": float(params["a"]),
        "b": float(params["b"]),
        "chunk": float(params.get("chunk", 256.0)),
    }


def describe_scan_local(params: dict) -> ExecutionPlan:
    mode = int(params.get("mode", 0))
    uniforms = _scan_uniforms(params)
    chunk = int(uniforms["chunk"])
    if chunk < 1:
        chunk = 1

    def _nblocks(n):
        return (n + chunk - 1) // chunk

    if mode == 0:
        def _sizes(input_sizes):
            n = input_sizes[0]
            return [n, _nblocks(n)]

        return ExecutionPlan(
            inputs=[InputSlot(name="data", dtype="float")],
            outputs=[
                OutputSlot(dtype="float", template="skl_tmp"),
                OutputSlot(dtype="float", template="skl_last"),
            ],
            workspace=[],
            uniforms=uniforms,
            output_size_fn=_sizes,
        )
    elif mode == 1:
        def _sizes(input_sizes):
            n = input_sizes[0]
            return [n, n, _nblocks(n), _nblocks(n)]

        return ExecutionPlan(
            inputs=[
                InputSlot(name="gain", dtype="float"),
                InputSlot(name="loss", dtype="float"),
            ],
            outputs=[
                OutputSlot(dtype="float", template="skl_tmp0"),
                OutputSlot(dtype="float", template="skl_tmp1"),
                OutputSlot(dtype="float", template="skl_last0"),
                OutputSlot(dtype="float", template="skl_last1"),
            ],
            workspace=[],
            uniforms=uniforms,
            output_size_fn=_sizes,
        )
    else:
        raise ValueError(f"Unknown StateKernelScanLocal mode: {mode}")


def describe_scan_totals(params: dict) -> ExecutionPlan:
    mode = int(params.get("mode", 0))
    uniforms = _scan_uniforms(params)

    if mode == 0:
        def _sizes(input_sizes):
            return [input_sizes[0]]  # same as input (len(last) = nblocks)

        return ExecutionPlan(
            inputs=[InputSlot(name="last", dtype="float")],
            outputs=[OutputSlot(dtype="float", template="skl_sp")],
            workspace=[],
            uniforms=uniforms,
            output_size_fn=_sizes,
        )
    elif mode == 1:
        def _sizes(input_sizes):
            return [input_sizes[0], input_sizes[1]]

        return ExecutionPlan(
            inputs=[
                InputSlot(name="last0", dtype="float"),
                InputSlot(name="last1", dtype="float"),
            ],
            outputs=[
                OutputSlot(dtype="float", template="skl_sp0"),
                OutputSlot(dtype="float", template="skl_sp1"),
            ],
            workspace=[],
            uniforms=uniforms,
            output_size_fn=_sizes,
        )
    else:
        raise ValueError(f"Unknown StateKernelScanTotals mode: {mode}")


def describe_scan_final(params: dict) -> ExecutionPlan:
    mode = int(params.get("mode", 0))
    uniforms = _scan_uniforms(params)

    if mode == 0:
        def _sizes(input_sizes):
            return [input_sizes[0]]  # same as tmp (n)

        return ExecutionPlan(
            inputs=[
                InputSlot(name="tmp", dtype="float"),
                InputSlot(name="sp", dtype="float"),
            ],
            outputs=[OutputSlot(dtype="float", template="skl_out")],
            workspace=[],
            uniforms=uniforms,
            output_size_fn=_sizes,
        )
    elif mode == 1:
        def _sizes(input_sizes):
            return [input_sizes[0], input_sizes[1]]

        return ExecutionPlan(
            inputs=[
                InputSlot(name="tmp0", dtype="float"),
                InputSlot(name="tmp1", dtype="float"),
                InputSlot(name="sp0", dtype="float"),
                InputSlot(name="sp1", dtype="float"),
            ],
            outputs=[
                OutputSlot(dtype="float", template="skl_out0"),
                OutputSlot(dtype="float", template="skl_out1"),
            ],
            workspace=[],
            uniforms=uniforms,
            output_size_fn=_sizes,
        )
    else:
        raise ValueError(f"Unknown StateKernelScanFinal mode: {mode}")