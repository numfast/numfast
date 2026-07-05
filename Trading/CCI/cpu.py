"""CCI kernel — Commodity Channel Index.

TP = (High + Low + Close) / 3
CCI = (TP - SMA(TP, period)) / (0.015 * MeanDeviation)
where MeanDeviation = sum(|TP - SMA(TP)|) / period
"""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    high = ctx.inputs[0].view
    low = ctx.inputs[1].view
    close = ctx.inputs[2].view
    dst = ctx.outputs[0].view
    tp_buf = ctx.workspace[0].view
    sma_buf = ctx.workspace[1].view
    period = ctx.uniforms["period"]
    n = high.length()
    
    # First pass: compute TP and SMA
    rolling_sum = 0.0
    for i in range(n):
        tp = (high.read(i) + low.read(i) + close.read(i)) / 3.0
        tp_buf.write(i, tp)
        rolling_sum += tp
        
        if i >= period:
            rolling_sum -= tp_buf.read(i - period)
        
        if i >= period - 1:
            sma = rolling_sum / period
            sma_buf.write(i, sma)
        else:
            sma_buf.write(i, tp)
    
    # Second pass: compute mean deviation and CCI
    for i in range(n):
        if i >= period - 1:
            sma = sma_buf.read(i)
            # Mean deviation over period
            md_sum = 0.0
            for j in range(period):
                md_sum += abs(tp_buf.read(i - j) - sma)
            md = md_sum / period
            
            if md != 0:
                cci = (tp_buf.read(i) - sma) / (0.015 * md)
            else:
                cci = 0.0
            dst.write(i, cci)
        else:
            dst.write(i, 0.0)
