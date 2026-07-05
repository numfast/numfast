"""Scan CPU references — one per kernel phase."""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu_local(ctx: ExecutionContext):
    """Local prefix sum per block of 64.

    Output 0: local prefix sums (same length as input)
    Output 1: block sums (one per full block of 64)
    """
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    sums = ctx.outputs[1].view
    n = src.length()
    BLOCK = 64

    num_full_blocks = n // BLOCK
    for b in range(num_full_blocks):
        start = b * BLOCK
        total = src.read(start)
        dst.write(start, total)
        for i in range(start + 1, start + BLOCK):
            total += src.read(i)
            dst.write(i, total)
        sums.write(b, total)

    # Partial last block
    remainder = n % BLOCK
    if remainder > 0:
        start = num_full_blocks * BLOCK
        total = src.read(start)
        dst.write(start, total)
        for i in range(start + 1, n):
            total += src.read(i)
            dst.write(i, total)
        sums.write(num_full_blocks, total)


def cpu_totals(ctx: ExecutionContext):
    """Prefix sum of block totals (single sequential pass)."""
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    n = src.length()
    if n == 0:
        return
    acc = src.read(0)
    dst.write(0, acc)
    for i in range(1, n):
        acc += src.read(i)
        dst.write(i, acc)


def cpu_final(ctx: ExecutionContext):
    """Add accumulated block offset to each local prefix sum.

    final[i] = local_scan[i] + prefix_sum(block_totals[0..block_id-1])
    """
    local = ctx.inputs[1].view
    prefix = ctx.inputs[2].view
    dst = ctx.outputs[0].view
    n = local.length()
    BLOCK = 64

    for i in range(n):
        block_id = i // BLOCK
        offset = prefix.read(block_id - 1) if block_id > 0 else 0.0
        dst.write(i, local.read(i) + offset)
