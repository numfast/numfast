"""Stochastic Oscillator kernel.

%K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
%D = SMA(%K, period_d)
"""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    high = ctx.inputs[0].view
    low = ctx.inputs[1].view
    close = ctx.inputs[2].view
    k_out = ctx.outputs[0].view
    d_out = ctx.outputs[1].view
    raw_k_buf = ctx.workspace[0].view  # store raw %K values for %D smoothing
    period_k = ctx.uniforms["period_k"]
    period_d = ctx.uniforms.get("period_d", 3)
    n = high.length()

    for i in range(n):
        if i >= period_k - 1:
            hh = high.read(i)
            ll = low.read(i)
            for j in range(1, period_k):
                hh = max(hh, high.read(i - j))
                ll = min(ll, low.read(i - j))

            if hh != ll:
                raw_k = (close.read(i) - ll) / (hh - ll) * 100.0
            else:
                raw_k = 50.0
        else:
            raw_k = 50.0

        raw_k_buf.write(i, raw_k)
        k_out.write(i, raw_k)

    # %D = SMA of raw %K over period_d
    for i in range(n):
        if i >= period_d - 1:
            d_sum = 0.0
            for j in range(period_d):
                d_sum += raw_k_buf.read(i - j)
            d_out.write(i, d_sum / period_d)
        else:
            d_out.write(i, raw_k_buf.read(i))
