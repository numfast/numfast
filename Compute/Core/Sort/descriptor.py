"""Bitonic Sort descriptor — one substage pass."""

from Runtime._lib.mod_iface import ExecutionPlan, InputSlot, OutputSlot


def describe_sort(params: dict) -> ExecutionPlan:
    """One substage of bitonic sort.

    Inputs: data (float) — array to sort (must be power of 2)
    Params: stage (int) — current stage (1..log2N)
            substage (int) — current substage (stage..1)
    Outputs: data (float) — partially sorted array
    """
    stage = params.get("stage", 1)
    substage = params.get("substage", 1)

    def _sizes(input_sizes):
        return [input_sizes[0]]  # output same size as input

    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="data")],
        uniforms={
            "stage": stage,
            "substage": substage,
        },
        output_size_fn=_sizes,
    )
