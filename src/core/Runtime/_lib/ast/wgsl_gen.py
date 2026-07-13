"""AST -> WGSL string generator.

Usage:
    wgsl = generate(node, id_var="id.x")
    wgsl = generate(node, id_var="id.x", packed_meta={...})
"""

from .ast_nodes import (
    AstNode,
    Field, Const, Unary, UnaryOp, Binary, BinaryOp, Call, Conditional,
    TmpRef,
)


def generate(
    node: AstNode,
    id_var: str = "id.x",
    packed_meta: dict | None = None,
    array_prefix: str = "",
) -> str:
    unpack_lines: list[str] = []
    expr = _visit(node, id_var, packed_meta or {}, array_prefix, unpack_lines)
    if unpack_lines:
        return "\n".join(unpack_lines) + "\n" + expr
    return expr


_VISITORS: dict = {}


def _visit(
    node: AstNode,
    id_var: str,
    packed_meta: dict,
    array_prefix: str,
    unpack_lines: list[str],
) -> str:
    if not _VISITORS:
        _VISITORS.update({
            Field: _visit_Field,
            Const: _visit_Const,
            Unary: _visit_Unary,
            Binary: _visit_Binary,
            Call: _visit_Call,
            Conditional: _visit_Conditional,
            TmpRef: _visit_TmpRef,
        })
    visitor = _VISITORS.get(type(node))
    if visitor is None:
        raise NotImplementedError(f"No visitor for {type(node).__name__}")
    return visitor(node, id_var, packed_meta, array_prefix, unpack_lines)


def _visit_Field(
    node: Field,
    id_var: str,
    packed_meta: dict,
    array_prefix: str,
    unpack_lines: list[str],
) -> str:
    meta = packed_meta.get(node.name)
    if meta is not None:
        unpack_lines.extend(_gen_unpack(node.name, id_var, meta, array_prefix))
        return node.name
    return f"{node.name}[{id_var}]"


def _visit_Const(
    node: Const,
    id_var: str,
    packed_meta: dict,
    array_prefix: str,
    unpack_lines: list[str],
) -> str:
    v = node.value
    if v == int(v):
        return f"{int(v)}.0"
    return f"{v}"


def _visit_Unary(
    node: Unary,
    id_var: str,
    packed_meta: dict,
    array_prefix: str,
    unpack_lines: list[str],
) -> str:
    expr = _visit(node.expr, id_var, packed_meta, array_prefix, unpack_lines)
    op = node.op.to_wgsl()
    if node.op == UnaryOp.NEG and isinstance(node.expr, (Unary, Binary, Call)):
        return f"{op}({expr})"
    return f"{op}{expr}"


def _visit_Binary(
    node: Binary,
    id_var: str,
    packed_meta: dict,
    array_prefix: str,
    unpack_lines: list[str],
) -> str:
    left = _visit(node.left, id_var, packed_meta, array_prefix, unpack_lines)
    right = _visit(node.right, id_var, packed_meta, array_prefix, unpack_lines)
    op = node.op.to_wgsl()

    if _needs_parens(node.left, node.op):
        left = f"({left})"
    if _needs_parens(node.right, node.op):
        right = f"({right})"

    return f"{left} {op} {right}"


def _needs_parens(child: AstNode, parent_op: BinaryOp) -> bool:
    if not isinstance(child, Binary):
        return False
    child_op = child.op
    if parent_op in (BinaryOp.ADD, BinaryOp.MUL) and child_op == parent_op:
        return False
    return _prec(child_op) < _prec(parent_op)


def _prec(op: BinaryOp) -> int:
    return {
        BinaryOp.OR: 1,
        BinaryOp.AND: 2,
        BinaryOp.LT: 3, BinaryOp.GT: 3,
        BinaryOp.LE: 3, BinaryOp.GE: 3,
        BinaryOp.EQ: 3, BinaryOp.NE: 3,
        BinaryOp.ADD: 4, BinaryOp.SUB: 4,
        BinaryOp.MUL: 5, BinaryOp.DIV: 5,
    }.get(op, 0)


def _visit_Call(
    node: Call,
    id_var: str,
    packed_meta: dict,
    array_prefix: str,
    unpack_lines: list[str],
) -> str:
    args = ", ".join(
        _visit(a, id_var, packed_meta, array_prefix, unpack_lines)
        for a in node.args
    )
    return f"{node.name}({args})"


def _visit_Conditional(
    node: Conditional,
    id_var: str,
    packed_meta: dict,
    array_prefix: str,
    unpack_lines: list[str],
) -> str:
    cond = _visit(node.cond, id_var, packed_meta, array_prefix, unpack_lines)
    then_expr = _visit(node.then_expr, id_var, packed_meta, array_prefix, unpack_lines)
    else_expr = _visit(node.else_expr, id_var, packed_meta, array_prefix, unpack_lines)
    return f"select({else_expr}, {then_expr}, {cond})"


def _visit_TmpRef(
    node: TmpRef,
    id_var: str,
    packed_meta: dict,
    array_prefix: str,
    unpack_lines: list[str],
) -> str:
    return node.name


def _gen_unpack(field_name: str, id_var: str, meta: dict, array_prefix: str) -> list[str]:
    parts = meta["parts"]
    scale = meta["scale"]
    offset = meta["offset"]
    num_parts_row = meta["num_parts_row"]
    arr = array_prefix or "packed_quote"

    if len(parts) == 1:
        part_idx, bit_offset, chunk_size = parts[0]
        rname = f"{field_name}_raw"
        mask = _mask_str(chunk_size)
        return [
            f"let {rname} = ({arr}[{id_var} * {num_parts_row} + {part_idx}] >> {bit_offset}) & {mask};",
            f"let {field_name} = f32({rname}) * {scale} + {offset};",
        ]

    shifts = [
        sum(parts[j][2] for j in range(i + 1, len(parts)))
        for i in range(len(parts))
    ]

    lines = []
    for i, (part_idx, bit_offset, chunk_size) in enumerate(parts):
        mask = _mask_str(chunk_size)
        lines.append(
            f"let part{i} = ({arr}[{id_var} * {num_parts_row} + {part_idx}] >> {bit_offset}) & {mask};"
        )

    raw_parts = []
    for i in range(len(parts)):
        if shifts[i] > 0:
            raw_parts.append(f"(part{i} << {shifts[i]}u)")
        else:
            raw_parts.append(f"part{i}")
    raw_expr = " | ".join(raw_parts)

    lines.append(f"let {field_name}_raw = {raw_expr};")
    lines.append(f"let {field_name} = f32({field_name}_raw) * {scale} + {offset};")
    return lines


def _mask_str(bits: int) -> str:
    mask = (1 << bits) - 1
    return f"0x{mask:X}u"
