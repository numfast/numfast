"""Bollinger Bands kernel.

Middle = SMA(period)
Upper = Middle + mult * StdDev
Lower = Middle - mult * StdDev
"""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    middle = ctx.outputs[0].view
    upper = ctx.outputs[1].view
    lower = ctx.outputs[2].view
    sma_buf = ctx.workspace[0].view  # rolling SMA
    sum_buf = ctx.workspace[1].view   # rolling sum of squares
    period = ctx.uniforms["period"]
    mult = ctx.uniforms.get("mult", 2.0)
    n = src.length()

    rolling_sum = 0.0
    rolling_sum_sq = 0.0

    for i in range(n):
        val = src.read(i)
        rolling_sum += val
        rolling_sum_sq += val * val

        if i >= period:
            prev = src.read(i - period)
            rolling_sum -= prev
            rolling_sum_sq -= prev * prev

        if i >= period - 1:
            avg = rolling_sum / period
            variance = rolling_sum_sq / period - avg * avg
            if variance < 0:
                variance = 0.0
            std = variance ** 0.5

            sma_buf.write(i, avg)
            sum_buf.write(i, rolling_sum_sq)

            middle.write(i, avg)
            upper.write(i, avg + mult * std)
            lower.write(i, avg - mult * std)
        else:
            middle.write(i, val)
            upper.write(i, val)
            lower.write(i, val)
