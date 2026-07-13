"""Kernel Stress — максимально сложное ядро для проверки ABI.

Требования:
- 5 входов (float, float, float, int, float)
- 7 выходов (float, float, float, float, float, float, float)
- workspace: 3 буфера (float, float, int)
- uniforms: int, float, bool — все три типа
"""

from Runtime._lib.mod_iface import (
    InputSlot, OutputSlot, BufferSpec, ExecutionPlan, ExecutionContext
)


def describe(params):
    return ExecutionPlan(
        inputs=[
            InputSlot(name="a", dtype="float"),
            InputSlot(name="b", dtype="float"),
            InputSlot(name="c", dtype="float"),
            InputSlot(name="d", dtype="int"),
            InputSlot(name="e", dtype="float"),
        ],
        outputs=[
            OutputSlot(dtype="float", template="out_{name}_1"),
            OutputSlot(dtype="float", template="out_{name}_2"),
            OutputSlot(dtype="float", template="out_{name}_3"),
            OutputSlot(dtype="float", template="out_{name}_4"),
            OutputSlot(dtype="float", template="out_{name}_5"),
            OutputSlot(dtype="float", template="out_{name}_6"),
            OutputSlot(dtype="float", template="out_{name}_7"),
        ],
        workspace=[
            BufferSpec(dtype="float", elements=0),
            BufferSpec(dtype="float", elements=0),
            BufferSpec(dtype="int", elements=0),
        ],
        uniforms=dict(params),
    )


def cpu(ctx):
    # 5 inputs
    a = ctx.inputs[0].view
    b = ctx.inputs[1].view
    c = ctx.inputs[2].view
    d = ctx.inputs[3].view
    e = ctx.inputs[4].view

    # 7 outputs
    o1 = ctx.outputs[0].view
    o2 = ctx.outputs[1].view
    o3 = ctx.outputs[2].view
    o4 = ctx.outputs[3].view
    o5 = ctx.outputs[4].view
    o6 = ctx.outputs[5].view
    o7 = ctx.outputs[6].view

    # 3 workspace
    ws1 = ctx.workspace[0].view
    ws2 = ctx.workspace[1].view
    ws3 = ctx.workspace[2].view

    # Uniforms (int, float, bool)
    scale = ctx.uniforms.get("scale", 1.0)
    offset = ctx.uniforms.get("offset", 0.0)
    clamp = ctx.uniforms.get("clamp", False)

    n = a.length()

    for i in range(n):
        v = a.read(i) * scale + b.read(i) + c.read(i) + float(d.read(i)) + e.read(i) + offset
        ws1.write(i, v)
        ws2.write(i, v * 0.5)
        ws3.write(i, int(v) % 100)

    for i in range(n):
        raw = ws1.read(i)
        o1.write(i, raw)
        o2.write(i, raw * 2)
        o3.write(i, raw / 2)
        o4.write(i, ws2.read(i))
        o5.write(i, float(ws3.read(i)))
        o6.write(i, raw + float(ws3.read(i)))

    # Clamp logic (uniform bool)
    if clamp:
        for i in range(n):
            raw = o1.read(i)
            if raw < 0:
                o1.write(i, 0.0)
                o7.write(i, 0.0)
            elif raw > 100:
                o1.write(i, 100.0)
                o7.write(i, 100.0)
            else:
                o7.write(i, raw)
    else:
        for i in range(n):
            o7.write(i, o1.read(i))
