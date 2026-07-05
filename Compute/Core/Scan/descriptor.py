"""Scan descriptors — ExecutionPlans for three chained Scan kernels."""

from numfast.Runtime._lib.mod_iface import ExecutionPlan, InputSlot, OutputSlot

BLOCK = 64

def describe_local(params: dict) -> ExecutionPlan:
    """Local prefix sum per block of 64.

    Inputs: data (float)
    Outputs: scan_local (float) — local prefix sums, same size as input
             block_sum (float) — total per block, size = ceil(N/64)
    """
    def _sizes(input_sizes):
        n = input_sizes[0]
        num_blocks = max(1, (n + BLOCK - 1) // BLOCK)
        return [n, num_blocks]

    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[
            OutputSlot(dtype="float", template="scan_local"),
            OutputSlot(dtype="float", template="block_sum"),
        ],
        workspace=[],
        uniforms={},
        output_size_fn=_sizes,
    )


def describe_totals(params: dict) -> ExecutionPlan:
    """Prefix sum of block totals.

    Inputs: block_sums (float)
    Outputs: block_prefix (float) — same size as input
    """
    def _sizes(input_sizes):
        return [input_sizes[0]]  # same as input

    return ExecutionPlan(
        inputs=[InputSlot(name="block_sums", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="block_prefix")],
        workspace=[],
        uniforms={},
        output_size_fn=_sizes,
    )


def describe_final(params: dict) -> ExecutionPlan:
    """Add accumulated block offset to local prefix sums.

    Inputs: data (float) — original input
            local (float) — local prefix sums
            prefix (float) — accumulated block prefix sums
    Outputs: scan (float) — final inclusive prefix sum, same as data
    """
    def _sizes(input_sizes):
        return [input_sizes[0]]  # same as "data" input

    return ExecutionPlan(
        inputs=[
            InputSlot(name="data", dtype="float"),
            InputSlot(name="local", dtype="float"),
            InputSlot(name="prefix", dtype="float"),
        ],
        outputs=[OutputSlot(dtype="float", template="scan")],
        workspace=[],
        uniforms={},
        output_size_fn=_sizes,
    )
