"""AST Compiler for NumFast.

Compiles AST nodes into executable ISA instructions.
"""

import numpy as np
from typing import Dict, List, Any, Optional
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
from .parser import BinaryOpType, UnaryOpType


class ISAInstruction:
    """ISA instruction representation."""

    def __init__(self, op: str, args: List[str], uniforms: Optional[Dict[str, Any]] = None):
        self.op = op
        self.args = args
        self.uniforms = uniforms or {}

    def __repr__(self):
        return f"ISAInstruction(op='{self.op}', args={self.args}, uniforms={self.uniforms})"


class ASTCompiler:
    """Compiles AST to ISA instructions."""

    def __init__(self, context=None):
        self.context = context

    def compile(self, node: ASTNode, context=None) -> List[ISAInstruction]:
        """Compile an AST node to ISA instructions.

        Args:
            node: AST node to compile
            context: Runtime context for compilation

        Returns:
            List of ISA instructions
        """
        self.context = context
        return self._compile_node(node)

    def _compile_node(self, node: ASTNode) -> List[ISAInstruction]:
        """Compile a single AST node."""
        if isinstance(node, Identifier):
            return self._compile_identifier(node)
        elif isinstance(node, Literal):
            return self._compile_literal(node)
        elif isinstance(node, BinaryOp):
            return self._compile_binary_op(node)
        elif isinstance(node, UnaryOp):
            return self._compile_unary_op(node)
        elif isinstance(node, Call):
            return self._compile_call(node)
        elif isinstance(node, MemberAccess):
            return self._compile_member_access(node)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def _compile_identifier(self, node: Identifier) -> List[ISAInstruction]:
        """Compile identifier reference."""
        return [ISAInstruction(op='LOAD', args=[node.name])]

    def _compile_literal(self, node: Literal) -> List[ISAInstruction]:
        """Compile literal value."""
        if isinstance(node.value, (int, float)):
            return [ISAInstruction(op='LOAD_CONST', args=[str(node.value)])]
        elif isinstance(node.value, str):
            return [ISAInstruction(op='LOAD_STR', args=[node.value])]
        else:
            return [ISAInstruction(op='LOAD_CONST', args=[repr(node.value)])]

    def _compile_binary_op(self, node: BinaryOp) -> List[ISAInstruction]:
        """Compile binary operation."""
        instructions = []
        instructions.extend(self._compile_node(node.left))
        instructions.extend(self._compile_node(node.right))

        op_map = {
            BinaryOpType.PLUS: 'ADD',
            BinaryOpType.MINUS: 'SUB',
            BinaryOpType.MULTIPLY: 'MUL',
            BinaryOpType.DIVIDE: 'DIV',
            BinaryOpType.MODULO: 'MOD',
            BinaryOpType.POWER: 'POW',
        }

        op = op_map.get(node.op_type, 'UNKNOWN')
        instructions.append(ISAInstruction(op=op, args=['result']))
        return instructions

    def _compile_unary_op(self, node: UnaryOp) -> List[ISAInstruction]:
        """Compile unary operation."""
        instructions = []
        instructions.extend(self._compile_node(node.operand))

        op_map = {
            UnaryOpType.NEGATIVE: 'NEG',
            UnaryOpType.NOT: 'NOT',
        }

        op = op_map.get(node.op_type, 'UNKNOWN')
        instructions.append(ISAInstruction(op=op, args=['result']))
        return instructions

    def _compile_call(self, node: Call) -> List[ISAInstruction]:
        """Compile function call."""
        instructions = []

        for arg in node.args:
            instructions.extend(self._compile_node(arg))

        if node.kwargs:
            for value in node.kwargs.values():
                instructions.extend(self._compile_node(value))

        func_op = self._resolve_function_name(node.func_name)
        instructions.append(ISAInstruction(op=func_op, args=node.args))
        return instructions

    def _compile_member_access(self, node: MemberAccess) -> List[ISAInstruction]:
        """Compile member access (e.g., context.close)."""
        instructions = []
        instructions.extend(self._compile_node(node.object))

        if self.context and node.attr_name in self.context.isa_ops_map:
            op = self.context.isa_ops_map[node.attr_name]
        else:
            op = f"GET_ATTR({node.attr_name})"

        instructions.append(ISAInstruction(op=op, args=[node.attr_name]))
        return instructions

    def _resolve_function_name(self, func_name: str) -> str:
        """Resolve function name to ISA operation."""
        if '.' in func_name:
            base, method = func_name.split('.', 1)
            return f"{base}_{method}"

        op_map = {
            'SMA': 'RUN_SMA',
            'EMA': 'RUN_EMA',
            'ATR': 'RUN_ATR',
            'TRANGE': 'RUN_TRANGE',
            'RSI': 'RUN_RSI',
            'STOCH': 'RUN_STOCH',
            'CCI': 'RUN_CCI',
            'ROC': 'RUN_ROC',
            'ROLLING_MIN': 'RUN_ROLLING_MIN',
            'ROLLING_MAX': 'RUN_ROLLING_MAX',
            'ROLLING_STDDEV': 'RUN_ROLLING_STDDEV',
        }

        return op_map.get(func_name.upper(), func_name.upper())

    def compile_to_isa(self, node: ASTNode, context=None) -> List[ISAInstruction]:
        """Public method to compile AST to ISA.

        Args:
            node: AST node to compile
            context: Runtime context for compilation

        Returns:
            List of ISA instructions
        """
        return self.compile(node, context)


# Standalone compilation function
def compile_to_isa(node: ASTNode, context=None) -> List[ISAInstruction]:
    """Compile an AST node to ISA instructions.

    Args:
        node: AST node to compile
        context: Runtime context for compilation

    Returns:
        List of ISA instructions
    """
    compiler = ASTCompiler(context)
    return compiler.compile_to_isa(node, context)


if __name__ == "__main__":
    from .parser import parse

    test_exprs = [
        "SMA(close, 14)",
        "close + open",
        "price * 1.5",
        "-(close - high)",
        "EMA(high, period=20)",
    ]

    for expr in test_exprs:
        print(f"\nCompiling: {expr}")
        try:
            ast = parse(expr)
            instructions = compile_to_isa(ast)
            print(f"  ISA Instructions:")
            for instr in instructions:
                print(f"    {instr}")
        except Exception as e:
            print(f"  Error: {e}")
