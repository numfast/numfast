"""ATR kernel — Average True Range (14-period Wilder's)."""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    high = ctx.inputs[0].view
    low = ctx.inputs[1].view
    close = ctx.inputs[2].view
    dst = ctx.outputs[0].view
    prev_close_buf = ctx.workspace[0].view
    period = ctx.uniforms["period"]
    n = high.length()

    for i in range(n):
        if i == 0:
            tr = high.read(i) - low.read(i)
            prev_close_buf.write(i, close.read(i))
            dst.write(i, tr)
        else:
            prev = prev_close_buf.read(i - 1)
            hl = high.read(i) - low.read(i)
            hc = abs(high.read(i) - prev)
            lc = abs(low.read(i) - prev)
            tr = max(hl, max(hc, lc))
            prev_close_buf.write(i, close.read(i))

            if i < period:
                # Simple average for first period bars
                prev_atr = dst.read(i - 1)
                dst.write(i, (prev_atr * i + tr) / (i + 1))
            else:
                # Wilder's smoothed ATR
                prev_atr = dst.read(i - 1)
                dst.write(i, (prev_atr * (period - 1) + tr) / period)
