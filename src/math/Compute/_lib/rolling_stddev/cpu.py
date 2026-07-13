"""Rolling StdDev — CPU reference.

Uses Welford-style rolling variance for numerical stability.
stddev = sqrt(E[X^2] - E[X]^2)
"""

import math
from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    period = int(ctx.uniforms["period"])
    n = src.length()
    
    running_sum = 0.0
    running_sum_sq = 0.0
    for i in range(n):
        x = src.read(i)
        running_sum += x
        running_sum_sq += x * x
        if i >= period:
            old = src.read(i - period)
            running_sum -= old
            running_sum_sq -= old * old
        if i >= period - 1:
            mean = running_sum / period
            variance = running_sum_sq / period - mean * mean
            if variance > 0:
                dst.write(i, math.sqrt(variance))
            else:
                dst.write(i, 0.0)
        else:
            dst.write(i, 0.0)
