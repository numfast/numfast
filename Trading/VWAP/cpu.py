"""VWAP kernel — Volume Weighted Average Price.

VWAP = sum(TP * Volume) / sum(Volume)
TP = (High + Low + Close) / 3
"""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    high = ctx.inputs[0].view
    low = ctx.inputs[1].view
    close = ctx.inputs[2].view
    volume = ctx.inputs[3].view
    dst = ctx.outputs[0].view
    cum_vol_buf = ctx.workspace[0].view
    n = high.length()

    cum_tpv = 0.0
    cum_vol = 0.0

    for i in range(n):
        tp = (high.read(i) + low.read(i) + close.read(i)) / 3.0
        vol = volume.read(i)
        cum_tpv += tp * vol
        cum_vol += vol

        cum_vol_buf.write(i, cum_vol)

        if cum_vol > 0:
            dst.write(i, cum_tpv / cum_vol)
        else:
            dst.write(i, tp)
