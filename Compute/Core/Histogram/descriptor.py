"""Histogram descriptor."""

from numfast.Runtime._lib.mod_iface import ExecutionPlan, InputSlot, OutputSlot


def describe(params: dict) -> ExecutionPlan:
    """Histogram: count elements per bin.

    Inputs: data (float)
    Params: min_val, max_val, num_bins (default 10)
    Outputs: bins (float) — array of num_bins counts
    """
    num_bins = params.get("num_bins", 10)
    min_val = params.get("min_val", 0.0)
    max_val = params.get("max_val", 1.0)

    def _sizes(input_sizes):
        return [num_bins]

    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="bins")],
        uniforms={
            "min_val": min_val,
            "max_val": max_val,
            "num_bins": num_bins,
        },
        output_size_fn=_sizes,
    )
