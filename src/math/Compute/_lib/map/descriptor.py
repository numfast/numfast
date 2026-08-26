"""Map descriptor — apply element-wise function.

S39: the func code is validated HERE — the single validation point for
both backends, executed by the compiler (describe() is called for every
job) BEFORE any dispatch. An unknown code raises ValueError with the
same message on CPU and GPU.
"""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan

# np.int32 (NOT plain int) for the uniform — same FIX as MapBinary descriptor:
# builder copies node.params into uniforms and pack_uniforms packs a plain
# int as uint32 bits (WGSL f32 then reads 1e-45 -> FTZ -> 0.0 -> every op
# executes the sin branch). np.int32 formats as "6" (template "map_6"),
# but isinstance(np.int32(6), int) is False, so pack_uniforms packs it as
# f32 6.0. Idempotent: int() accepts np.int32 on repeated describe() calls.
import numpy as np

_FUNC_CODES = (0, 1, 2, 3, 4, 5, 6, 7)
_FUNC_NAMES = "0=sin,1=cos,2=exp,3=sqrt,4=log,5=abs,6=neg,7=square"


def describe(params: dict) -> ExecutionPlan:
    func = int(params.get("func", 7))
    if func not in _FUNC_CODES:
        raise ValueError(
            f"Unknown Map function code: {func}; expected 0..7 ({_FUNC_NAMES})"
        )
    params["func"] = np.int32(func)
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="map_{func}")],
        workspace=[],
        uniforms=dict(params),
    )