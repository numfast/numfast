"""TrueRange — CPU reference.

TR = max(high - low, |high - prev_close|, |low - prev_close|)
where prev_close = close[i-1] (previous period's close)
"""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    high = ctx.inputs[0].view
    low = ctx.inputs[1].view
    close = ctx.inputs[2].view
    dst = ctx.outputs[0].view
    n = high.length()

    prev_close = close.read(0)
    for i in range(n):
        if i == 0:
            dst.write(i, 0.0)
        else:
            hl = high.read(i) - low.read(i)
            hc = abs(high.read(i) - prev_close)
            lc = abs(low.read(i) - prev_close)
            tr = max(hl, hc, lc)
            dst.write(i, tr)
        prev_close = close.read(i)