"""Histogram CPU reference."""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu_histogram(ctx: ExecutionContext):
    """Compute histogram: count elements per bin.

    Input: data (float)
    Output: bins (float) — histogram counts
    Uniforms: min_val, max_val, num_bins
    """
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    n = src.length()
    min_val = ctx.uniforms.get("min_val", 0.0)
    max_val = ctx.uniforms.get("max_val", 1.0)
    num_bins = ctx.uniforms.get("num_bins", 10)

    for b in range(num_bins):
        dst.write(b, 0.0)

    for i in range(n):
        val = src.read(i)
        clamped = max(min_val, min(val, max_val))
        if max_val > min_val:
            norm = (clamped - min_val) / (max_val - min_val)
            bin_idx = int(norm * num_bins)
            bin_idx = min(bin_idx, num_bins - 1)
        else:
            bin_idx = 0
        dst.write(bin_idx, dst.read(bin_idx) + 1.0)
