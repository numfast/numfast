"""Keltner Channels kernel.

Middle = EMA(close, period_ema)
Upper = Middle + ATR(period_atr) * mult
Lower = Middle - ATR(period_atr) * mult
"""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    high = ctx.inputs[0].view
    low = ctx.inputs[1].view
    close = ctx.inputs[2].view
    middle = ctx.outputs[0].view
    upper = ctx.outputs[1].view
    lower = ctx.outputs[2].view
    prev_ema_buf = ctx.workspace[0].view
    prev_atr_buf = ctx.workspace[1].view
    prev_close_buf = ctx.workspace[2].view
    period_ema = ctx.uniforms["period_ema"]
    period_atr = ctx.uniforms.get("period_atr", period_ema)
    mult = ctx.uniforms.get("mult", 2.0)
    n = high.length()
    
    # ATR-specific: Wilder's multiplier
    atr_mult = 2.0 / (period_atr + 1.0)
    ema_mult = 2.0 / (period_ema + 1.0)
    
    for i in range(n):
        # --- ATR ---
        if i == 0:
            tr = high.read(i) - low.read(i)
            prev_close_buf.write(i, close.read(i))
            prev_atr_buf.write(i, tr)
            atr = tr
        else:
            prev_c = prev_close_buf.read(i - 1)
            hl = high.read(i) - low.read(i)
            hc = abs(high.read(i) - prev_c)
            lc = abs(low.read(i) - prev_c)
            tr = max(hl, max(hc, lc))
            prev_close_buf.write(i, close.read(i))
            
            if i < period_atr:
                # Simple average for initial
                prev_atr = prev_atr_buf.read(i - 1)
                atr = (prev_atr * i + tr) / (i + 1)
            else:
                prev_atr = prev_atr_buf.read(i - 1)
                atr = (prev_atr * (period_atr - 1) + tr) / period_atr
            
            prev_atr_buf.write(i, atr)
        
        # --- EMA ---
        if i == 0:
            ema = close.read(i)
        else:
            prev_ema = prev_ema_buf.read(i - 1)
            ema = (close.read(i) - prev_ema) * ema_mult + prev_ema
        
        prev_ema_buf.write(i, ema)
        middle.write(i, ema)
        
        # --- Channels ---
        upper.write(i, ema + atr * mult)
        lower.write(i, ema - atr * mult)
