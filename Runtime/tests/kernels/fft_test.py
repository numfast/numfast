"""Simple FFT kernel — workspace test.

Реализует прямое DFT (не FFT) для проверки workspace.
В реальности будет CUDA/wgsl cufft.
"""

from numfast.Runtime._lib.mod_iface import (
    InputSlot, OutputSlot, BufferSpec, ExecutionPlan, ExecutionContext
)
import math


def describe(params):
    return ExecutionPlan(
        inputs=[InputSlot(name="signal", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="fft_{name}")],
        workspace=[
            BufferSpec(dtype="float", elements=0),
            BufferSpec(dtype="float", elements=0),
        ],
        uniforms=dict(params),
    )


def cpu(ctx):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    real_ws = ctx.workspace[0].view
    imag_ws = ctx.workspace[1].view

    n = src.length()
    for k in range(n):
        real_sum = 0.0
        imag_sum = 0.0
        for t in range(n):
            angle = 2 * math.pi * k * t / n
            real_sum += src.read(t) * math.cos(angle)
            imag_sum -= src.read(t) * math.sin(angle)
        real_ws.write(k, real_sum / n)
        imag_ws.write(k, imag_sum / n)
        dst.write(k, math.sqrt(real_sum**2 + imag_sum**2) / n)
