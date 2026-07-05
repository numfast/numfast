"""ADX kernel — Average Directional Index.

Pipeline per Bar i (i >= 1):
  TR = max(High-Low, |High-prevClose|, |Low-prevClose|)
  +DM = High - prevHigh  (if > prevLow - Low and > 0, else 0)
  -DM = prevLow - Low    (if > High - prevHigh and > 0, else 0)

  Smoothed TR  = rolling sum of last `period` TR values
  Smoothed +DM = rolling sum of last `period` +DM values
  Smoothed -DM = rolling sum of last `period` -DM values

  +DI = +DM / TR * 100
  -DI = -DM / TR * 100
  DX = |+DI - -DI| / (+DI + -DI) * 100
  ADX = SMA of DX over `period` bars
"""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    high = ctx.inputs[0].view
    low = ctx.inputs[1].view
    close = ctx.inputs[2].view
    plus_di_out = ctx.outputs[0].view
    minus_di_out = ctx.outputs[1].view
    adx_out = ctx.outputs[2].view
    tr_buf = ctx.workspace[0].view
    dm_plus_buf = ctx.workspace[1].view
    dm_minus_buf = ctx.workspace[2].view
    dx_buf = ctx.workspace[3].view
    period = ctx.uniforms["period"]
    n = high.length()

    for i in range(n):
        if i == 0:
            tr = high.read(i) - low.read(i)
            dm_plus = 0.0
            dm_minus = 0.0
        else:
            prev_c = close.read(i - 1)
            hl = high.read(i) - low.read(i)
            hc = abs(high.read(i) - prev_c)
            lc = abs(low.read(i) - prev_c)
            tr = max(hl, max(hc, lc))

            up_move = high.read(i) - high.read(i - 1)
            down_move = low.read(i - 1) - low.read(i)
            dm_plus = up_move if up_move > down_move and up_move > 0 else 0.0
            dm_minus = down_move if down_move > up_move and down_move > 0 else 0.0

        tr_buf.write(i, tr)
        dm_plus_buf.write(i, dm_plus)
        dm_minus_buf.write(i, dm_minus)

        if i < period:
            plus_di_out.write(i, 0.0)
            minus_di_out.write(i, 0.0)
            dx_buf.write(i, 0.0)
            adx_out.write(i, 0.0)
        else:
            s_tr = 0.0
            s_dmp = 0.0
            s_dmn = 0.0
            for j in range(period):
                idx = i - j
                s_tr += tr_buf.read(idx)
                s_dmp += dm_plus_buf.read(idx)
                s_dmn += dm_minus_buf.read(idx)

            if s_tr > 0:
                pdi = s_dmp / s_tr * 100.0
                ndi = s_dmn / s_tr * 100.0
            else:
                pdi = 0.0
                ndi = 0.0

            plus_di_out.write(i, pdi)
            minus_di_out.write(i, ndi)

            if pdi + ndi > 0:
                dx = abs(pdi - ndi) / (pdi + ndi) * 100.0
            else:
                dx = 0.0
            dx_buf.write(i, dx)

            dx_sum = 0.0
            for j in range(period):
                idx = i - j
                if idx >= period:
                    dx_sum += dx_buf.read(idx)
            adx_out.write(i, dx_sum / period)
