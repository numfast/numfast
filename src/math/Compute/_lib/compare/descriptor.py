"""Compare descriptor.

Input count depends on scalar mode:
  0 inputs: both operands are scalar (degenerate, always compares uniforms)
  1 input:  one operand is scalar (use_scalar_a or use_scalar_b)
  2 inputs: both operands are arrays

Параметр op:
  - строка: "gt" / "ge" / "lt" / "le" / "eq" / "ne" (default "gt") — CPU-путь;
  - числовой код: 0=gt, 1=ge, 2=lt, 3=le, 4=eq, 5=ne — GPU-путь.

Uniform "op" всегда числовой: WebGPU pack_uniforms пакует uniforms
как f32 и падает на строках (frozen builder передаёт node.params в
uniforms без нормализации). Строковый op принимается для шаблонов
выходов и CPU-диспетчеризации (см. cpu.py).
"""

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan

_OP_NAMES = {0: "gt", 1: "ge", 2: "lt", 3: "le", 4: "eq", 5: "ne"}
_OP_CODES = {"gt": 0.0, "ge": 1.0, "lt": 2.0, "le": 3.0, "eq": 4.0, "ne": 5.0}


def describe(params: dict) -> ExecutionPlan:
    op = params.get("op", "gt")
    if isinstance(op, (int, float)):
        name = _OP_NAMES.get(int(op), "gt")
        code = float(op)
    else:
        name = op if op in _OP_CODES else "gt"
        code = _OP_CODES.get(name, 0.0)

    inputs = []
    if not float(params.get("use_scalar_a", 0)):
        inputs.append(InputSlot(name="a", dtype="float"))
    if not float(params.get("use_scalar_b", 0)):
        inputs.append(InputSlot(name="b", dtype="float"))

    return ExecutionPlan(
        inputs=inputs,
        outputs=[OutputSlot(dtype="uint32", template="compare")],
        workspace=[],
        uniforms={
            "op": code,
            "scalar_a": float(params.get("scalar_a", 0.0)),
            "scalar_b": float(params.get("scalar_b", 0.0)),
            "use_scalar_a": float(params.get("use_scalar_a", 0)),
            "use_scalar_b": float(params.get("use_scalar_b", 0)),
        },
    )
