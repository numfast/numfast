"""AST node types for NumFast expression ISA.

Universal node types (not op-specific):
  Field(name)        — reference to a named field (e.g. "low", "dHigh")
  Const(value)       — literal float value
  Unary(op, expr)    — unary operation (-x, !x)
  Binary(op, l, r)   — binary operation (x+y, x>y, x&&y)
  Call(name, args)   — function call (abs(x), max(a,b))
  Conditional(cond, then, else) — ternary (cond ? then : else)

Visitor pattern: subclass AstVisitor and implement visit_* methods.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Optional


# Symbol maps used by BinaryOp and UnaryOp (module-level for Python 3.14+ enum compat)
_BINARYOP_SYMBOL_MAP: dict = {}
_UNARYOP_SYMBOL_MAP: dict = {}


class BinaryOp(Enum):
    """Binary operators."""
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()

    # Comparison
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    EQ = auto()
    NE = auto()

    # Logical
    AND = auto()
    OR = auto()

    @classmethod
    def from_symbol(cls, s: str) -> Optional[BinaryOp]:
        global _BINARYOP_SYMBOL_MAP
        if not _BINARYOP_SYMBOL_MAP:
            _BINARYOP_SYMBOL_MAP = {
                '+': cls.ADD, '-': cls.SUB,
                '*': cls.MUL, '/': cls.DIV,
                '<': cls.LT, '>': cls.GT,
                '<=': cls.LE, '>=': cls.GE,
                '==': cls.EQ, '!=': cls.NE,
                '&&': cls.AND, '||': cls.OR,
            }
        return _BINARYOP_SYMBOL_MAP.get(s)

    def to_wgsl(self) -> str:
        return {
            BinaryOp.ADD: '+', BinaryOp.SUB: '-',
            BinaryOp.MUL: '*', BinaryOp.DIV: '/',
            BinaryOp.LT: '<', BinaryOp.GT: '>',
            BinaryOp.LE: '<=', BinaryOp.GE: '>=',
            BinaryOp.EQ: '==', BinaryOp.NE: '!=',
            BinaryOp.AND: '&&', BinaryOp.OR: '||',
        }[self]

    def to_python(self) -> str:
        return {
            BinaryOp.ADD: '+', BinaryOp.SUB: '-',
            BinaryOp.MUL: '*', BinaryOp.DIV: '/',
            BinaryOp.LT: '<', BinaryOp.GT: '>',
            BinaryOp.LE: '<=', BinaryOp.GE: '>=',
            BinaryOp.EQ: '==', BinaryOp.NE: '!=',
            BinaryOp.AND: ' and ', BinaryOp.OR: ' or ',
        }[self]

    def is_comparison(self) -> bool:
        return self in (BinaryOp.LT, BinaryOp.GT, BinaryOp.LE,
                        BinaryOp.GE, BinaryOp.EQ, BinaryOp.NE)

    def is_logical(self) -> bool:
        return self in (BinaryOp.AND, BinaryOp.OR)


class UnaryOp(Enum):
    """Unary operators."""
    NEG = auto()
    NOT = auto()

    @classmethod
    def from_symbol(cls, s: str) -> Optional[UnaryOp]:
        global _UNARYOP_SYMBOL_MAP
        if not _UNARYOP_SYMBOL_MAP:
            _UNARYOP_SYMBOL_MAP = {'-': cls.NEG, '!': cls.NOT}
        return _UNARYOP_SYMBOL_MAP.get(s)

    def to_wgsl(self) -> str:
        return {UnaryOp.NEG: '-', UnaryOp.NOT: '!'}[self]

    def to_python(self) -> str:
        return {UnaryOp.NEG: '-', UnaryOp.NOT: ' not '}[self]


class AstNode:
    """Base class for all AST nodes."""
    __slots__ = ()
    _children: tuple

    def __repr__(self) -> str:
        return self._repr(0)

    def _repr(self, indent: int) -> str:
        raise NotImplementedError


class Field(AstNode):
    """Reference to a named field (e.g. low, dHigh, close)."""
    __slots__ = ('name',)
    name: str

    def __init__(self, name: str):
        self.name = name

    def _repr(self, indent: int) -> str:
        return '  ' * indent + f'Field({self.name})'


class Const(AstNode):
    """Literal float constant."""
    __slots__ = ('value',)
    value: float

    def __init__(self, value: float):
        self.value = value

    def _repr(self, indent: int) -> str:
        return '  ' * indent + f'Const({self.value})'


class Unary(AstNode):
    """Unary operation."""
    __slots__ = ('op', 'expr')
    op: UnaryOp
    expr: AstNode

    def __init__(self, op: UnaryOp, expr: AstNode):
        self.op = op
        self.expr = expr

    def _repr(self, indent: int) -> str:
        return ('  ' * indent + f'Unary({self.op.name})\n'
                + self.expr._repr(indent + 1))


class Binary(AstNode):
    """Binary operation."""
    __slots__ = ('op', 'left', 'right')
    op: BinaryOp
    left: AstNode
    right: AstNode

    def __init__(self, op: BinaryOp, left: AstNode, right: AstNode):
        self.op = op
        self.left = left
        self.right = right

    def _repr(self, indent: int) -> str:
        return ('  ' * indent + f'Binary({self.op.name})\n'
                + self.left._repr(indent + 1) + '\n'
                + self.right._repr(indent + 1))


class Call(AstNode):
    """Function call."""
    __slots__ = ('name', 'args')
    name: str
    args: tuple[AstNode, ...]

    def __init__(self, name: str, args: list[AstNode]):
        self.name = name
        self.args = tuple(args)

    def _repr(self, indent: int) -> str:
        lines = ['  ' * indent + f'Call({self.name})']
        for a in self.args:
            lines.append(a._repr(indent + 1))
        return '\n'.join(lines)


class Conditional(AstNode):
    """Ternary conditional: cond ? then_expr : else_expr."""
    __slots__ = ('cond', 'then_expr', 'else_expr')
    cond: AstNode
    then_expr: AstNode
    else_expr: AstNode

    def __init__(self, cond: AstNode, then_expr: AstNode, else_expr: AstNode):
        self.cond = cond
        self.then_expr = then_expr
        self.else_expr = else_expr

    def _repr(self, indent: int) -> str:
        return ('  ' * indent + f'Conditional\n'
                + self.cond._repr(indent + 1) + '\n'
                + self.then_expr._repr(indent + 1) + '\n'
                + self.else_expr._repr(indent + 1))


class TmpRef(AstNode):
    """Reference to a computed temporary variable (from CSE).

    Used internally after Common Subexpression Elimination.
    `name` is the generated temporary variable name.
    """
    __slots__ = ('name',)
    name: str

    def __init__(self, name: str):
        self.name = name

    def _repr(self, indent: int) -> str:
        return '  ' * indent + f'TmpRef({self.name})'


# --- Visitor ---

class AstVisitor:
    """Base visitor for AST nodes.

    Subclass and implement visit_Field, visit_Const, etc.
    Override generic_visit for fallback.
    """

    def visit(self, node: AstNode):
        method = f'visit_{type(node).__name__}'
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: AstNode):
        raise NotImplementedError(
            f"No visitor for {type(node).__name__}"
        )


# --- Helpers ---

def extract_physicals(node: AstNode) -> list[str]:
    """Extract physical field names from an AST node.

    Traverses AST and collects all Field names in order.
    """
    result: list[str] = []
    seen: set[str] = set()

    def _walk(n: AstNode):
        if isinstance(n, Field):
            if n.name not in seen:
                seen.add(n.name)
                result.append(n.name)
        elif isinstance(n, Unary):
            _walk(n.expr)
        elif isinstance(n, Binary):
            _walk(n.left)
            _walk(n.right)
        elif isinstance(n, Call):
            for a in n.args:
                _walk(a)
        elif isinstance(n, Conditional):
            _walk(n.cond)
            _walk(n.then_expr)
            _walk(n.else_expr)
        elif isinstance(n, TmpRef):
            pass  # No fields in temp references
        # Const: no fields

    _walk(node)
    return result
