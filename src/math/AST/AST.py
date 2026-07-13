"""AST Extension -- language frontend.

Entry point. All implementation in _lib/.
"""

from _lib.parser import parse
from _lib.compiler import compile_to_isa
from _lib.evaluator import evaluate
from _lib.ast_nodes import ast_to_dict


def setup(kernel):
    """Register AST ops in kernel metadata."""
    kernel.metadata.setdefault("AST", {})
    kernel.metadata["AST"]["version"] = "0.1.0"
