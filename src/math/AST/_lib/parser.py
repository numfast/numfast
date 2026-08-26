"""AST Parser for NumFast.

Parses string expressions into Abstract Syntax Trees (AST).
Supported:
  - Arithmetic: +, -, *, /, %, **
  - Comparison: >, <, >=, <=, ==, !=
  - Logical: and, or, not
  - Function calls: SMA(), EMA(), where(), ...
  - Identifiers, literals, member access
"""

import ast as python_ast
from typing import List, Any, Optional
from enum import Enum
from .ast_nodes import (
    ASTNode,
    Identifier,
    Literal,
    BinaryOp,
    UnaryOp,
    Call,
    MemberAccess,
)


class BinaryOpType(Enum):
    """Supported binary operation types."""
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    POWER = "**"
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="
    AND = "and"
    OR = "or"

    @classmethod
    def from_string(cls, op_str: str) -> 'BinaryOpType':
        for op_type in cls:
            if op_type.value == op_str:
                return op_type
        raise ValueError(f"Unknown binary operator: {op_str}")


class UnaryOpType(Enum):
    """Supported unary operation types."""
    NEGATIVE = "-"
    NOT = "not"

    @classmethod
    def from_string(cls, op_str: str) -> 'UnaryOpType':
        for op_type in cls:
            if op_type.value == op_str:
                return op_type
        raise ValueError(f"Unknown unary operator: {op_str}")


class ASTParser:
    """Parser for NumFast AST expressions."""

    def __init__(self):
        pass

    def parse(self, expression: str) -> ASTNode:
        """Parse a string expression into an AST.

        Args:
            expression: The expression string to parse

        Returns:
            ASTNode representing the parsed expression

        Raises:
            SyntaxError: If the expression cannot be parsed
        """
        try:
            python_ast_expr = python_ast.parse(expression, mode='eval')
            return self._convert_ast(python_ast_expr.body)
        except Exception as e:
            raise SyntaxError(f"Error parsing expression '{expression}': {e}")

    def _convert_ast(self, node: Any) -> ASTNode:
        """Convert Python AST to NumFast AST."""
        if isinstance(node, python_ast.Expression):
            return self._convert_ast(node.body)

        elif isinstance(node, python_ast.Name):
            return Identifier(name=node.id)

        elif isinstance(node, python_ast.Constant):
            return Literal(value=node.value)

        elif isinstance(node, python_ast.UnaryOp):
            if isinstance(node.op, python_ast.USub):
                op_type = UnaryOpType.NEGATIVE
            elif isinstance(node.op, python_ast.Not):
                op_type = UnaryOpType.NOT
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

            operand = self._convert_ast(node.operand)
            return UnaryOp(op_type=op_type, operand=operand)

        elif isinstance(node, python_ast.BinOp):
            op_type = BinaryOpType.from_string(self._get_binop_symbol(node.op))
            left = self._convert_ast(node.left)
            right = self._convert_ast(node.right)
            return BinaryOp(op_type=op_type, left=left, right=right)

        elif isinstance(node, python_ast.Call):
            func_name = None
            if isinstance(node.func, python_ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, python_ast.Attribute):
                if isinstance(node.func.value, python_ast.Name):
                    func_name = f"{node.func.value.id}.{node.func.attr}"
                else:
                    func_name = node.func.attr

            if not func_name:
                raise SyntaxError(f"Unsupported function call: {python_ast.unparse(node.func)}")

            args = [self._convert_ast(arg) for arg in node.args]
            keywords = {kw.arg: self._convert_ast(kw.value) for kw in node.keywords}

            return Call(func_name=func_name, args=args, kwargs=keywords)

        elif isinstance(node, python_ast.Attribute):
            if isinstance(node.value, python_ast.Name):
                return MemberAccess(object=self._convert_ast(node.value),
                                  attr_name=node.attr)
            else:
                raise SyntaxError(f"Unsupported attribute access: {python_ast.unparse(node)}")

        elif isinstance(node, python_ast.Compare):
            return self._convert_compare(node)

        elif isinstance(node, python_ast.BoolOp):
            return self._convert_boolop(node)

        else:
            raise SyntaxError(f"Unsupported AST node: {type(node).__name__}")

    def _get_binop_symbol(self, op: Any) -> str:
        """Get string representation of binary operator."""
        if isinstance(op, python_ast.Add):
            return "+"
        elif isinstance(op, python_ast.Sub):
            return "-"
        elif isinstance(op, python_ast.Mult):
            return "*"
        elif isinstance(op, python_ast.Div):
            return "/"
        elif isinstance(op, python_ast.Mod):
            return "%"
        elif isinstance(op, python_ast.Pow):
            return "**"
        else:
            raise ValueError(f"Unsupported binary operator: {type(op).__name__}")

    def _convert_compare(self, node: python_ast.Compare) -> ASTNode:
        """Convert Python Compare node to NumFast AST.
        
        Handles: >, <, >=, <=, ==, !=
        Python AST represents chained comparisons as multiple ops,
        but we only handle single comparisons (a op b).
        """
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise SyntaxError("Chained comparisons not supported")
        
        left = self._convert_ast(node.left)
        right = self._convert_ast(node.comparators[0])
        py_op = node.ops[0]
        
        if isinstance(py_op, python_ast.Gt):
            op_type = BinaryOpType.GT
        elif isinstance(py_op, python_ast.GtE):
            op_type = BinaryOpType.GE
        elif isinstance(py_op, python_ast.Lt):
            op_type = BinaryOpType.LT
        elif isinstance(py_op, python_ast.LtE):
            op_type = BinaryOpType.LE
        elif isinstance(py_op, python_ast.Eq):
            op_type = BinaryOpType.EQ
        elif isinstance(py_op, python_ast.NotEq):
            op_type = BinaryOpType.NE
        else:
            raise ValueError(f"Unsupported comparison operator: {type(py_op).__name__}")
        
        return BinaryOp(op_type=op_type, left=left, right=right)

    def _convert_boolop(self, node: python_ast.BoolOp) -> ASTNode:
        """Convert Python BoolOp node to NumFast AST.
        
        Handles: and, or
        Python AST allows multiple values: a and b and c
        We chain them as binary: ((a and b) and c)
        """
        values = [self._convert_ast(v) for v in node.values]
        op_type = BinaryOpType.AND if isinstance(node.op, python_ast.And) else BinaryOpType.OR
        
        # Chain multiple values
        result = values[0]
        for v in values[1:]:
            result = BinaryOp(op_type=op_type, left=result, right=v)
        return result


# Create alias for backward compatibility
def parse(expression: str) -> ASTNode:
    """Parse a string expression into an AST.

    Args:
        expression: The expression string to parse

    Returns:
        ASTNode representing the parsed expression
    """
    parser = ASTParser()
    return parser.parse(expression)


if __name__ == "__main__":
    parser = ASTParser()
    test_exprs = [
        "SMA(close, 14)",
        "close + open",
        "price * 1.5",
        "-(close - high)",
        "EMA(high, period=20)",
    ]

    for expr in test_exprs:
        print(f"\nParsing: {expr}")
        try:
            ast = parser.parse(expr)
            print(f"  AST: {ast}")
        except SyntaxError as e:
            print(f"  Error: {e}")
