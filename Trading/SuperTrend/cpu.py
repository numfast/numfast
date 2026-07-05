"""SuperTrend kernel.

Complex stateful trend-following indicator.
"""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    high = ctx.inputs[0].view
    low = ctx.inputs[1].view
    close = ctx.inputs[2].view
    st_out = ctx.outputs[0].view
    dir_out = ctx.outputs[1].view
    prev_close_buf = ctx.workspace[0].view
    atr_buf = ctx.workspace[1].view
    prev_final_upper = ctx.workspace[2].view
    prev_final_lower = ctx.workspace[3].view
    prev_dir_buf = ctx.workspace[4].view
    period = ctx.uniforms["period"]
    mult = ctx.uniforms.get("mult", 3.0)
    n = high.length()

    for i in range(n):
        if i == 0:
            tr = high.read(i) - low.read(i)
            atr = tr
            prev_close_buf.write(i, close.read(i))
        else:
            prev_c = prev_close_buf.read(i - 1)
            hl = high.read(i) - low.read(i)
            hc = abs(high.read(i) - prev_c)
            lc = abs(low.read(i) - prev_c)
            tr = max(hl, max(hc, lc))
            prev_close_buf.write(i, close.read(i))

            if i < period:
                prev_atr = atr_buf.read(i - 1) if i > 0 else tr
                atr = (prev_atr * i + tr) / (i + 1)
            else:
                prev_atr = atr_buf.read(i - 1)
                atr = (prev_atr * (period - 1) + tr) / period

        atr_buf.write(i, atr)

        hl2 = (high.read(i) + low.read(i)) / 2.0
        basic_upper = hl2 + mult * atr
        basic_lower = hl2 - mult * atr

        if i == 0:
            final_upper = basic_upper
            final_lower = basic_lower
            direction = 1
        else:
            prev_fu = prev_final_upper.read(i - 1)
            prev_fl = prev_final_lower.read(i - 1)

            final_upper = basic_upper if (basic_upper < prev_fu or close.read(i - 1) > prev_fu) else prev_fu
            final_lower = basic_lower if (basic_lower > prev_fl or close.read(i - 1) < prev_fl) else prev_fl

            prev_dir = prev_dir_buf.read(i - 1)
            if prev_dir == 1:
                direction = -1 if close.read(i) <= final_lower else 1
            else:
                direction = 1 if close.read(i) >= final_upper else -1

        prev_final_upper.write(i, final_upper)
        prev_final_lower.write(i, final_lower)
        prev_dir_buf.write(i, direction)

        st_out.write(i, final_upper if direction == -1 else final_lower)
        dir_out.write(i, float(direction))
