"""Williams %R kernel.

%R = (Highest High - Close) / (Highest High - Lowest Low) * -100
"""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    high = ctx.inputs[0].view
    low = ctx.inputs[1].view
    close = ctx.inputs[2].view
    dst = ctx.outputs[0].view
    period = ctx.uniforms["period"]
    n = high.length()

    for i in range(n):
        if i >= period - 1:
            hh = high.read(i)
            ll = low.read(i)
            for j in range(1, period):
                hh = max(hh, high.read(i - j))
                ll = min(ll, low.read(i - j))

            if hh != ll:
                dst.write(i, (hh - close.read(i)) / (hh - ll) * -100.0)
            else:
                dst.write(i, -50.0)
        else:
            dst.write(i, 0.0)
