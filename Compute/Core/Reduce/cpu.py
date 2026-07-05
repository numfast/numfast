"""Reduce kernel — CPU реализация.

Суммирует все элементы входного массива.
Reference implementation — чистый Python, без оптимизаций.
"""

from numfast.Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    n = src.length()
    
    total = 0.0
    for i in range(n):
        total += src.read(i)
    
    dst.write(0, total)
