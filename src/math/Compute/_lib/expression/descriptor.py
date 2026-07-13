"""Expression descriptor — определяет кол-во входов по формуле."""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan


def describe(params: dict) -> ExecutionPlan:
    formula = params.get("formula", "A")
    # Найти уникальные заглавные буквы (A-Z) в формуле
    vars = sorted(set(c for c in formula if c.isupper() and c.isalpha()))
    num_vars = len(vars)
    inputs = [InputSlot(name=v, dtype="float") for v in vars]
    return ExecutionPlan(
        inputs=inputs,
        outputs=[OutputSlot(dtype="float", template="expr")],
        workspace=[],
        uniforms={
            "formula": formula,
            "num_vars": float(num_vars),
        },
    )
