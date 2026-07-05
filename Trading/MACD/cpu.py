"""MACD kernel — CPU implementation."""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx):
    src = ctx.inputs[0].view
    fast = ctx.uniforms["fast"]
    slow = ctx.uniforms["slow"]
    signal = ctx.uniforms.get("signal", 9)

    macd_line = ctx.outputs[0].view
    signal_line = ctx.outputs[1].view
    histogram = ctx.outputs[2].view

    ema_fast = ctx.workspace[0].view
    ema_slow = ctx.workspace[1].view

    n = src.length()

    mult_fast = 2.0 / (fast + 1.0)
    for i in range(n):
        if i == 0:
            ema_fast.write(i, src.read(i))
        else:
            prev = ema_fast.read(i - 1)
            ema_fast.write(i, (src.read(i) - prev) * mult_fast + prev)

    mult_slow = 2.0 / (slow + 1.0)
    for i in range(n):
        if i == 0:
            ema_slow.write(i, src.read(i))
        else:
            prev = ema_slow.read(i - 1)
            ema_slow.write(i, (src.read(i) - prev) * mult_slow + prev)

    for i in range(n):
        macd_line.write(i, ema_fast.read(i) - ema_slow.read(i))

    mult_signal = 2.0 / (signal + 1.0)
    for i in range(n):
        if i == 0:
            signal_line.write(i, macd_line.read(i))
        else:
            prev = signal_line.read(i - 1)
            signal_line.write(i, (macd_line.read(i) - prev) * mult_signal + prev)

    for i in range(n):
        histogram.write(i, macd_line.read(i) - signal_line.read(i))
