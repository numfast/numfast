"""RSI kernel — CPU implementation."""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    period = ctx.uniforms["period"]

    n = src.length()
    if n < period + 1:
        for i in range(n):
            dst.write(i, 50.0)
        return

    deltas = [src.read(i) - src.read(i - 1) for i in range(1, n)]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period):
        dst.write(i, 50.0)

    for i in range(period, n):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            dst.write(i, 100.0)
        else:
            rs = avg_gain / avg_loss
            dst.write(i, 100.0 - 100.0 / (1.0 + rs))
