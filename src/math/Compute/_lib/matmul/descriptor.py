"""MatMul descriptor — tiled matrix multiply C = A × B."""

from Runtime._lib.mod_iface import ExecutionPlan, InputSlot, OutputSlot

TILE = 16


def describe_matmul(params: dict) -> ExecutionPlan:
    """Matrix multiply C = A × B.

    Inputs: A (float, M×K flatten), B (float, K×N flatten)
    Params: M, N, K — matrix dimensions
    Outputs: C (float, M×N flatten)
    """
    M, N, K = int(params.get("M", 64)), int(params.get("N", 64)), int(params.get("K", 64))
    if M < 1 or N < 1 or K < 1:
        raise ValueError(f"MatMul M/N/K must be >= 1; got M={M}, N={N}, K={K}")

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
        uniforms={"M": float(M), "N": float(N), "K": float(K)},
        output_size_fn=_sizes,
        dispatch=(dx, dy, 1),
    )
