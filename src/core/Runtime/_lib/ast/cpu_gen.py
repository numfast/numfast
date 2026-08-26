"""CPU Generator — AST -> Python eval expression.

Generates a Python expression string from an AST node.
Safe for eval with restricted namespace (abs, max, min only).

Usage:
    gen = CpuGenerator()
    expr = gen.generate(node)
    # Result: "(low + dHigh)"
    
    # Evaluate with numpy arrays:
    import numpy as np
    result = eval(expr, {"__builtins__": {}, "abs": abs, "max": max, "min": min},
                  {"low": np.array([...]), "dHigh": np.array([...])})
"""

from .ast_nodes import (
    AstNode, AstVisitor,
    Field, Const, Unary, UnaryOp, Binary, BinaryOp, Call, Conditional,
    TmpRef,
)


class CpuGenerator(AstVisitor):
    """Generate Python expression string from AST.

    Result is safe for eval() with restricted namespace
    (abs, max, min only, no __builtins__).
    
    Field("low") -> "low"
    Const(0.5) -> "0.5"
    Binary(ADD, left, right) -> "(left + right)"
    """

    def generate(self, node: AstNode) -> str:
        """Generate Python expression string."""
        return self.visit(node)

    def visit_Field(self, node: Field) -> str:
        return node.name

    def visit_Const(self, node: Const) -> str:
        v = node.value
        if v == int(v):
            return str(int(v))
        # Use repr for precision, but avoid scientific notation
        return repr(v)

    def visit_Unary(self, node: Unary) -> str:
        expr = self.visit(node.expr)
        op = node.op.to_python().strip()
        if node.op == UnaryOp.NEG:
            if isinstance(node.expr, (Binary, Unary, Call)):
                return f"(-{expr})"
            return f"-{expr}"
        return f"{op}{expr}"

    def visit_Binary(self, node: Binary) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op.to_python()
        # Parenthesize children with lower precedence
        if isinstance(node.left, Binary) and self._prec(node.left.op) < self._prec(node.op):
            left = f"({left})"
        if isinstance(node.right, Binary) and self._prec(node.right.op) < self._prec(node.op):
            right = f"({right})"
        # Division: guard against zero
        if node.op == BinaryOp.DIV:
            return f"({left} / ({right} if {right} != 0 else 1e-300))"
        if node.op in (BinaryOp.SUB, BinaryOp.ADD):
            return f"({left} {op} {right})"
        return f"({left} {op} {right})"

    @staticmethod
    def _prec(op: BinaryOp) -> int:
        return {
            BinaryOp.OR: 1, BinaryOp.AND: 2,
            BinaryOp.LT: 3, BinaryOp.GT: 3, BinaryOp.LE: 3, BinaryOp.GE: 3,
            BinaryOp.EQ: 3, BinaryOp.NE: 3,
            BinaryOp.ADD: 4, BinaryOp.SUB: 4,
            BinaryOp.MUL: 5, BinaryOp.DIV: 5,
        }.get(op, 0)

    def visit_Call(self, node: Call) -> str:
        args = ", ".join(self.visit(a) for a in node.args)
        return f"{node.name}({args})"

    def visit_Conditional(self, node: Conditional) -> str:
        cond = self.visit(node.cond)
        then_expr = self.visit(node.then_expr)
        else_expr = self.visit(node.else_expr)
        return f"({then_expr} if {cond} else {else_expr})"

    def visit_TmpRef(self, node: TmpRef) -> str:
        return node.name


# --- Convenience ---

def cpu_expr(node: AstNode) -> str:
    """Generate Python expression and return safe eval namespace."""
    gen = CpuGenerator()
    return gen.generate(node)


def cpu_eval(node: AstNode, data: dict[str, object]) -> object:
    """Evaluate AST node directly on CPU with given data dict.
    
    Args:
        node: AST node to evaluate
        data: dict of field_name -> array/scalar values
    
    Returns:
        Evaluated result (numpy array or scalar)
    
    Example:
        node = parse_expr("low + dHigh")
        result = cpu_eval(node, {"low": np.array([1,2,3]), "dHigh": np.array([0.1, 0.2, 0.3])})
    """
    expr = cpu_expr(node)
    safe_globals = {
        "__builtins__": {},
        "abs": abs,
        "max": max,
        "min": min,
    }
    return eval(expr, safe_globals, data)
