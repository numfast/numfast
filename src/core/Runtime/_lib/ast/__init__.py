"""AST — Abstract Syntax Tree for NumFast expression ISA.

Strings exist only at TOML loading. Everything inside works with AstNode objects.
"""

from .ast_nodes import (
    AstNode, AstVisitor,
    Field, Const, Unary, UnaryOp, Binary, BinaryOp, Call, Conditional, TmpRef,
    extract_physicals,
)
from .parser import parse_expr, ParseError
from .wgsl_gen import generate as wgsl_generate
from .cpu_gen import CpuGenerator, cpu_expr, cpu_eval
from .optimizer import fold_constants, optimize, optimize_with_bindings, cse
