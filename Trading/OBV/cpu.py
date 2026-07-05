"""OBV kernel — On-Balance Volume."""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    close = ctx.inputs[0].view
    volume = ctx.inputs[1].view
    dst = ctx.outputs[0].view
    obv_buf = ctx.workspace[0].view
    n = close.length()
    
    cumulative = 0.0
    for i in range(n):
        if i == 0:
            cumulative = 0.0
        else:
            if close.read(i) > close.read(i - 1):
                cumulative += volume.read(i)
            elif close.read(i) < close.read(i - 1):
                cumulative -= volume.read(i)
        
        obv_buf.write(i, cumulative)
        dst.write(i, cumulative)
