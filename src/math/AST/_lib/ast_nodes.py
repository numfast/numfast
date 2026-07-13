"""AST Node Definitions for NumFast.

Defines the Abstract Syntax Tree node types for expression parsing and compilation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum


class ASTNode(ABC):
    """Base class for all AST nodes."""

    def __init__(self):
        pass

    @abstractmethod
    def accept(self, visitor):
        """Accept a visitor pattern visitor."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Identifier(ASTNode):
    """AST node representing a variable/identifier reference."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def accept(self, visitor):
        return visitor.visit_identifier(self)

    def __repr__(self):
        return f"Identifier(name='{self.name}')"


class Literal(ASTNode):
    """AST node representing a literal value."""

    def __init__(self, value: Any):
        super().__init__()
        self.value = value

    def accept(self, visitor):
        return visitor.visit_literal(self)

    def __repr__(self):
        return f"Literal(value={repr(self.value)})"


class BinaryOp(ASTNode):
    """AST node representing a binary operation."""

    def __init__(self, op_type: Enum, left: ASTNode, right: ASTNode):
        super().__init__()
        self.op_type = op_type
        self.left = left
        self.right = right

    def accept(self, visitor):
        return visitor.visit_binary_op(self)

    def __repr__(self):
        return f"BinaryOp(op={self.op_type.value}, left={self.left}, right={self.right})"


class UnaryOp(ASTNode):
    """AST node representing a unary operation."""

    def __init__(self, op_type: Enum, operand: ASTNode):
        super().__init__()
        self.op_type = op_type
        self.operand = operand

    def accept(self, visitor):
        return visitor.visit_unary_op(self)

    def __repr__(self):
        return f"UnaryOp(op={self.op_type.value}, operand={self.operand})"


class Call(ASTNode):
    """AST node representing a function/method call."""

    def __init__(self, func_name: str, args: List[ASTNode], kwargs: Optional[Dict[str, ASTNode]] = None):
        super().__init__()
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs or {}

    def accept(self, visitor):
        return visitor.visit_call(self)

    def __repr__(self):
        return f"Call(func_name='{self.func_name}', args={self.args}, kwargs={self.kwargs})"


class MemberAccess(ASTNode):
    """AST node representing attribute access (e.g., context.close)."""

    def __init__(self, object: ASTNode, attr_name: str):
        super().__init__()
        self.object = object
        self.attr_name = attr_name

    def accept(self, visitor):
        return visitor.visit_member_access(self)

    def __repr__(self):
        return f"MemberAccess(object={self.object}, attr_name='{self.attr_name}')"


class FunctionDefinition(ASTNode):
    """AST node representing a function definition."""

    def __init__(self, name: str, params: List[str], body: ASTNode):
        super().__init__()
        self.name = name
        self.params = params
        self.body = body

    def accept(self, visitor):
        return visitor.visit_function_definition(self)

    def __repr__(self):
        return f"FunctionDefinition(name='{self.name}', params={self.params}, body={self.body})"


class IfNode(ASTNode):
    """AST node representing an if statement."""

    def __init__(self, test: ASTNode, body: ASTNode, orelse: Optional[ASTNode] = None):
        super().__init__()
        self.test = test
        self.body = body
        self.orelse = orelse

    def accept(self, visitor):
        return visitor.visit_if_node(self)

    def __repr__(self):
        return f"IfNode(test={self.test}, body={self.body}, orelse={self.orelse})"


class ReturnNode(ASTNode):
    """AST node representing a return statement."""

    def __init__(self, value: ASTNode):
        super().__init__()
        self.value = value

    def accept(self, visitor):
        return visitor.visit_return_node(self)

    def __repr__(self):
        return f"ReturnNode(value={self.value})"


# ============================================================================
# Visitors for AST traversal and compilation
# ============================================================================

class ASTVisitor(ABC):
    """Abstract base class for AST visitors."""

    @abstractmethod
    def visit_identifier(self, node: Identifier):
        pass

    @abstractmethod
    def visit_literal(self, node: Literal):
        pass

    @abstractmethod
    def visit_binary_op(self, node: BinaryOp):
        pass

    @abstractmethod
    def visit_unary_op(self, node: UnaryOp):
        pass

    @abstractmethod
    def visit_call(self, node: Call):
        pass

    @abstractmethod
    def visit_member_access(self, node: MemberAccess):
        pass

    def visit_function_definition(self, node: FunctionDefinition):
        return None

    def visit_if_node(self, node: IfNode):
        return None

    def visit_return_node(self, node: ReturnNode):
        return None


class ASTPrinter(ASTVisitor):
    """Visitor for printing AST nodes."""

    def visit_identifier(self, node: Identifier):
        return repr(node)

    def visit_literal(self, node: Literal):
        return repr(node)

    def visit_binary_op(self, node: BinaryOp):
        return repr(node)

    def visit_unary_op(self, node: UnaryOp):
        return repr(node)

    def visit_call(self, node: Call):
        return repr(node)

    def visit_member_access(self, node: MemberAccess):
        return repr(node)

    def visit_function_definition(self, node: FunctionDefinition):
        return repr(node)

    def visit_if_node(self, node: IfNode):
        return repr(node)

    def visit_return_node(self, node: ReturnNode):
        return repr(node)


# ============================================================================
# Serialization: AST -> dict (for use without class imports)
# ============================================================================

def ast_to_dict(node) -> dict:
    """Convert AST node to a plain dictionary.

    Recursively converts AST nodes to dicts so consumers can work
    without importing any class types.

    Returns dict with at minimum {"type": "NodeTypeName"} plus
    type-specific fields.
    """
    if isinstance(node, Identifier):
        return {"type": "Identifier", "name": node.name}
    elif isinstance(node, Literal):
        return {"type": "Literal", "value": node.value}
    elif isinstance(node, BinaryOp):
        return {
            "type": "BinaryOp",
            "op_type": node.op_type.value,
            "left": ast_to_dict(node.left),
            "right": ast_to_dict(node.right),
        }
    elif isinstance(node, UnaryOp):
        return {
            "type": "UnaryOp",
            "op_type": node.op_type.value,
            "operand": ast_to_dict(node.operand),
        }
    elif isinstance(node, Call):
        return {
            "type": "Call",
            "func_name": node.func_name,
            "args": [ast_to_dict(a) for a in node.args],
            "kwargs": {k: ast_to_dict(v) for k, v in node.kwargs.items()},
        }
    elif isinstance(node, MemberAccess):
        return {
            "type": "MemberAccess",
            "object": ast_to_dict(node.object),
            "attr_name": node.attr_name,
        }
    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")
