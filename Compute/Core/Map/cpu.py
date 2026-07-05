"""Map kernel — CPU reference implementation."""

import math
from numfast.Runtime._lib.mod_iface import ExecutionContext


_FUNCS = {
    0: math.sin,
    1: math.cos,
    2: math.exp,
    3: math.sqrt,
    4: math.log,
    5: abs,
    6: lambda x: -x,
    7: lambda x: x * x,
}


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    func_code = int(ctx.uniforms.get("func", 7))
    func = _FUNCS.get(func_code)
    if func is None:
        raise ValueError(f"Unknown Map function code: {func_code}")
    for i in range(src.length()):
        dst.write(i, func(src.read(i)))
