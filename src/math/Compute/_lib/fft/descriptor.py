"""FFT descriptors — one butterfly stage + bit-reversal."""

from Runtime._lib.mod_iface import ExecutionPlan, InputSlot, OutputSlot


def describe_stage(params: dict) -> ExecutionPlan:
    """One stage of Cooley-Tukey radix-2 FFT.

    Inputs: complex (float, 2N) — interleaved real/imag
    Params: stage (int) — 0..log2(N)-1
            N (int) — total complex points (power of 2)
    Outputs: complex (float, 2N)
    """
    N = params.get("N", 1024)

    def _sizes(input_sizes):
        return [2 * N]  # real + imag per point

    return ExecutionPlan(
        inputs=[InputSlot(name="complex", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="complex")],
        uniforms={"stage": params.get("stage", 0), "N": N},
        output_size_fn=_sizes,
    )


def describe_bitreverse(params: dict) -> ExecutionPlan:
    """Bit-reversal permutation.

    Inputs: complex (float, 2N)
    Params: N (int) — power of 2
    Outputs: complex (float, 2N) — bit-reversed order
    """
    N = params.get("N", 1024)

    def _sizes(input_sizes):
        return [2 * N]

    return ExecutionPlan(
        inputs=[InputSlot(name="complex", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="complex")],
        uniforms={"N": N},
        output_size_fn=_sizes,
    )
