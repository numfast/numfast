"""RandomKernel descriptor — counter-based random creation without input buffers."""

import math

from Runtime._lib.mod_iface import ExecutionPlan, OutputSlot

WORKGROUP = 64


def describe(params: dict) -> ExecutionPlan:
    """RandomKernel plan: inputs=[], outputs=[rnd_{n}], uniforms {n, mode, seed, p0..p1}.

    Modes: 0=uniform(low=p0, high=p1), 1=int_range[p0,p1), 2=normal(loc=p0, scale=p1).
    """
    n = int(params.get("n", 0))
    mode = int(params.get("mode", 0))
    seed = int(params.get("seed", 42))

    if n < 1:
        raise ValueError(f"RandomKernel: n must be >= 1, got {n}")
    if mode not in (0, 1, 2):
        raise ValueError(f"RandomKernel: unknown mode {mode}, expected 0|1|2")

    p0 = float(params.get("p0", 0.0))
    p1 = float(params.get("p1", 1.0))

    def _sizes(_input_sizes):
        return [n]

    return ExecutionPlan(
        inputs=[],
        outputs=[OutputSlot(dtype="float", template=f"rnd_{n}")],
        uniforms={
            "n": n,
            "mode": mode,
            "seed": seed & 0xFFFFFFFF,
            "p0": p0,
            "p1": p1,
        },
        output_size_fn=_sizes,
        dispatch=(math.ceil(n / WORKGROUP), 1, 1),
    )
