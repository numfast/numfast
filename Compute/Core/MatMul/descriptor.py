"""MatMul descriptor — tiled matrix multiply C = A × B."""

from numfast.Runtime._lib.mod_iface import ExecutionPlan, InputSlot, OutputSlot

TILE = 16


def describe_matmul(params: dict) -> ExecutionPlan:
    """Matrix multiply C = A × B.

    Inputs: A (float, M×K flatten), B (float, K×N flatten)
    Params: M, N, K — matrix dimensions
    Outputs: C (float, M×N flatten)
    """
    M = params.get("M", 64)
    N = params.get("N", 64)
    K = params.get("K", 64)

    def _sizes(input_sizes):
        return [M * N]  # C = M×N flattened

    dx = (N + TILE - 1) // TILE
    dy = (M + TILE - 1) // TILE

    return ExecutionPlan(
        inputs=[
            InputSlot(name="A", dtype="float"),
            InputSlot(name="B", dtype="float"),
        ],
        outputs=[OutputSlot(dtype="float", template="C")],
        uniforms={"M": M, "N": N, "K": K},
        output_size_fn=_sizes,
        dispatch=(dx, dy, 1),
    )
