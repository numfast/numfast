"""Compare — CPU reference: out[i] = a[i] op b[i], result 0/1 (uint32).

op принимается строкой ("gt"/"ge"/"lt"/"le"/"eq"/"ne") или числовым
кодом (0=gt, 1=ge, 2=lt, 3=le, 4=eq, 5=ne) — см. descriptor.
"""

from Runtime._lib.mod_iface import ExecutionContext

_CODES = {0: "gt", 1: "ge", 2: "lt", 3: "le", 4: "eq", 5: "ne"}


def _compare(a, b, op: str) -> int:
    if op == "gt":   return 1 if a > b else 0
    if op == "ge":   return 1 if a >= b else 0
    if op == "lt":   return 1 if a < b else 0
    if op == "le":   return 1 if a <= b else 0
    if op == "eq":   return 1 if a == b else 0
    if op == "ne":   return 1 if a != b else 0
    return 0


def _resolve_op(op):
    if isinstance(op, str):
        return op if op in ("gt", "ge", "lt", "le", "eq", "ne") else "gt"
    return _CODES.get(int(op), "gt")


def cpu(ctx: ExecutionContext):
    op = _resolve_op(ctx.uniforms.get("op", "gt"))
    dst = ctx.outputs[0].view
    use_sa = int(ctx.uniforms.get("use_scalar_a", 0))
    use_sb = int(ctx.uniforms.get("use_scalar_b", 0))
    scalar_a = ctx.uniforms.get("scalar_a", 0.0)
    scalar_b = ctx.uniforms.get("scalar_b", 0.0)

    # Map input index based on scalar flags
    input_idx = 0
    a_view = None if use_sa else ctx.inputs[input_idx].view
    if not use_sa:
        input_idx += 1
    b_view = None if use_sb else ctx.inputs[input_idx].view

    n = dst.length()
    for i in range(n):
        a_val = scalar_a if use_sa else a_view.read(i)
        b_val = scalar_b if use_sb else b_view.read(i)
        dst.write(i, _compare(a_val, b_val, op))
