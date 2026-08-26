"""Expression — CPU reference.

Python eval с ограниченным namespace для безопасности.
Доступны только: переменные A-Z, функции abs/max/min, арифметика.
"""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    formula = ctx.uniforms.get("formula", "A")
    num_vars = int(ctx.uniforms.get("num_vars", 1))

    dst = ctx.outputs[0].view
    n = dst.length()

    # Подготовить входы: переменные A, B, C, ...
    views = {}
    for i in range(num_vars):
        letter = chr(65 + i)  # A=65, B=66, ...
        views[letter] = ctx.inputs[i].view

    # Безопасный namespace: только abs/max/min, без __builtins__
    safe_globals = {
        "__builtins__": {},
        "abs": abs,
        "max": max,
        "min": min,
    }

    for i in range(n):
        local_vars = {}
        for letter, view in views.items():
            local_vars[letter] = view.read(i)
        result = eval(formula, safe_globals, local_vars)
        dst.write(i, result)
