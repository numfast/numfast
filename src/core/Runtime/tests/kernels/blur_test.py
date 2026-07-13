"""Simple 3x3 box blur — 2D kernel test."""

from Runtime._lib.mod_iface import (
    InputSlot, OutputSlot, BufferSpec, ExecutionPlan, ExecutionContext
)


def describe(params):
    return ExecutionPlan(
        inputs=[InputSlot(name="image", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="blur_{size}")],
        workspace=[],
        uniforms=dict(params),
    )


def cpu(ctx):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    size = ctx.uniforms.get("size", 3)
    width = ctx.uniforms.get("width", 4)

    n = src.length()
    height = n // width
    half = size // 2

    for i in range(n):
        row = i // width
        col = i % width
        total = 0.0
        count = 0
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                r = row + dy
                c = col + dx
                if 0 <= r < height and 0 <= c < width:
                    total += src.read(r * width + c)
                    count += 1
        dst.write(i, total / count)
