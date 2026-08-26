import numpy as np

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan

_INT_OPS = (0, 1, 2, 6)


def describe(params: dict) -> ExecutionPlan:
    # S49: validate op BEFORE compile/dispatch (both backends — one path).
    op = int(params.get("op", 0))
    if op not in (0, 1, 2, 3, 4, 5, 6):
        raise ValueError(
            f"Unknown MapBinary op code: {op}; expected 0..6 "
            "(0=add, 1=sub, 2=mul, 3=div, 4=max, 5=min, 6=mod)"
        )

    # dtype-семантика (решение владельца): params.dtype
    #   "float32" (default) -> f32 буферы, все ops;
    #   "int32"             -> i32 буферы, ops {0=add,1=sub,2=mul} WRAP,
    #                          {6=mod} C-trunc % guard b==0 -> 0.
    # Строковый "dtype" удаляется из params -> numeric uniform int_mode
    # (uniforms обязаны быть numeric для pack_uniforms). Idempotent:
    # describe вызывается повторно (compile/planner/builder).
    if "int_mode" not in params:
        dtype = params.pop("dtype", "float32")
        if dtype not in ("float32", "int32"):
            raise ValueError(
                f"MapBinary: dtype {dtype!r} not supported "
                f"(allowed: float32|int32)")
        params["int_mode"] = 1 if dtype == "int32" else 0
    int_mode = int(params["int_mode"])

    if int_mode and op not in _INT_OPS:
        raise ValueError(
            f"MapBinary int32: op {op} not supported "
            f"(allowed: 0=add, 1=sub, 2=mul WRAP, 6=mod C-trunc)"
        )

    # Stabilize output template: op=1.0 -> "map_binary_1", not "map_binary_1.0".
    # FIX: builder copies node.params into uniforms (builder.py:119 uniforms[k]=v)
    # and pack_uniforms packs int as uint32 bits (garbage as f32, FTZ->0).
    # Uniform "op" must be float for WGSL(f32), but template needs int "1".
    # Use np.int32 for params["op"]: formats as "1" (template), but
    # isinstance(..., int) is False, so pack_uniforms packs as float 1.0.
    # int32 mode: plain Python int (NOT np.int32!) -> builder keeps it over
    # the int plan.uniform -> pack_uniforms '<I' u32 bits -> WGSL i32 op.
    params["op"] = int(op) if int_mode else np.int32(op)
    slot_dtype = "int32" if int_mode else "float"
    if int_mode:
        params["scalar_a"] = int(params.get("scalar_a", 0))
        params["scalar_b"] = int(params.get("scalar_b", 0))
        params["use_scalar_a"] = int(params.get("use_scalar_a", 0))
        params["use_scalar_b"] = int(params.get("use_scalar_b", 0))
    else:
        params["scalar_a"] = float(params.get("scalar_a", 0.0))
        params["scalar_b"] = float(params.get("scalar_b", 0.0))
        params["use_scalar_a"] = float(params.get("use_scalar_a", 0))
        params["use_scalar_b"] = float(params.get("use_scalar_b", 0))

    # S53: both-scalars unreachable mode. Rejection used to be an accident of
    # builder's blanket "no inputs" ValueError; S204 legitimized 0-input plans
    # (creation kernels), so the contract moves here explicitly.
    if params["use_scalar_a"] and params["use_scalar_b"]:
        raise ValueError(
            "MapBinary: use_scalar_a=1 + use_scalar_b=1 -> no inputs "
            "(at least one operand must be a series input)"
        )

    inputs = []
    if not params["use_scalar_a"]:
        inputs.append(InputSlot(name="a", dtype=slot_dtype))
    if not params["use_scalar_b"]:
        inputs.append(InputSlot(name="b", dtype=slot_dtype))

    return ExecutionPlan(
        inputs=inputs,
        outputs=[OutputSlot(dtype=slot_dtype, template="map_binary_{op}")],
        workspace=[],
        uniforms={
            "op": int(op) if int_mode else float(op),
            "scalar_a": params["scalar_a"],
            "scalar_b": params["scalar_b"],
            "use_scalar_a": params["use_scalar_a"],
            "use_scalar_b": params["use_scalar_b"],
            "int_mode": int_mode,
            "_pad0": 0,
            "_pad1": 0,
            "_pad2": 0,
        },
    )
