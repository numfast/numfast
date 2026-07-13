"""Rolling Max -- O(n) monotonic deque."""

from collections import deque
from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    period = int(ctx.uniforms["period"])
    n = src.length()

    limit = min(period - 1, n)
    for i in range(limit):
        dst.write(i, 0.0)

    if n < period:
        return

    dq = deque()
    for i in range(n):
        while dq and dq[0] <= i - period:
            dq.popleft()

        val = src.read(i)

        # Remove indices with value <= current (strictly decreasing queue)
        while dq and src.read(dq[-1]) <= val:
            dq.pop()

        dq.append(i)

        if i >= period - 1:
            dst.write(i, src.read(dq[0]))
