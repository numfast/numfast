"""QuoteTable resolver — resolves logical fields to AST expressions.

QuoteTable defines how logical fields (high, low, close) map to
physical buffer expressions (e.g. "low + dHigh").

Strings exist only at load time (TOML parsing). All internal
operations use AST nodes.

Two resolution modes:
  Inline: embed AST expression directly into WGSL preamble
          (for workgroup_size(64) parallel kernels)
  Materialized: evaluate AST on CPU via AstEval kernel
          (for serial kernels)
"""

import tomllib
from typing import Optional


class QuoteTable:
    """Logical-to-physical field resolver for QuoteTable data."""

    def __init__(self):
        self._fields: dict[str, dict] = {}
        self._physical_set: set[str] = set()

    def load_string(self, toml_text: str):
        """Load QuoteTable definitions from TOML string.
        
        Parses expression strings to AST immediately.
        After loading, strings no longer exist — only AST.
        """
        # Late import at registration boundary
        from Runtime._lib.ast import parse_expr

        data = tomllib.loads(toml_text)

        layout = data.get("layout", {})
        self._physical_set = set(layout.get("physical", []))

        field_section = data.get("field", {})
        for field_name, entry in field_section.items():
            if isinstance(entry, dict) and "expression" in entry:
                expr_str = entry["expression"]
                try:
                    ast_node = parse_expr(expr_str)
                except SyntaxError as e:
                    raise ValueError(
                        f"Failed to parse expression for field '{field_name}': "
                        f"{expr_str!r}: {e}"
                    ) from e
                self._fields[field_name] = {
                    "ast": ast_node,
                    "description": entry.get("description", ""),
                }

    def load_file(self, path: str):
        with open(path, 'rb') as f:
            self.load_string(f.read().decode('utf-8'))

    def is_logical(self, field: str) -> bool:
        return field in self._fields

    def is_physical(self, field: str) -> bool:
        return field in self._physical_set

    def get_ast(self, field: str):
        from Runtime._lib.ast import AstNode
        entry = self._fields.get(field)
        return entry["ast"] if entry else None

    def resolve(self, field: str, _depth: int = 0):
        """Resolve a logical field to AST expression + physical inputs.

        Returns:
            (ast_node, [physical_field_names])
        """
        from Runtime._lib.ast import extract_physicals, Field as AstField
        # _substitute_fields is defined in this module

        if _depth > 10:
            return (None, [])

        entry = self._fields.get(field)
        if entry is None:
            return (None, [])

        node = entry["ast"]
        raw_physicals = extract_physicals(node)

        all_physicals: list[str] = []
        substitutions: dict = {}

        for p in raw_physicals:
            if p in self._fields:
                sub_node, sub_physicals = self.resolve(p, _depth + 1)
                if sub_node is not None and sub_physicals:
                    substitutions[p] = sub_node
                    all_physicals.extend(sub_physicals)
                else:
                    all_physicals.append(p)
            else:
                all_physicals.append(p)

        if substitutions:
            node = _substitute_fields(node, substitutions)

        seen = set()
        deduped = []
        for p in all_physicals:
            if p not in seen:
                seen.add(p)
                deduped.append(p)

        return (node, deduped) if deduped else (None, [])

    def transform(self, jobs: list[dict]) -> list[dict]:
        """Transform jobs: replace logical fields with AstEval jobs.

        Args:
            jobs: List of job dicts

        Returns:
            Modified job list with only physical field references
        """
        new_jobs = []
        expr_counter = 0

        for job in jobs:
            raw_inputs = job.get("inputs", [])
            new_inputs = []
            needs_transform = False

            for inp in raw_inputs:
                if self.is_logical(inp):
                    node, physicals = self.resolve(inp)
                    if node is not None and physicals:
                        needs_transform = True
                        # Late import
                        from Runtime._lib.ast.cpu_gen import cpu_expr
                        py_expr = cpu_expr(node)
                        tmp_name = f"__qt_{inp}_{expr_counter}"
                        expr_counter += 1

                        ast_job = {
                            "op": "AstEval",
                            "params": {
                                "expr": py_expr,
                                "num_vars": len(physicals),
                                "field_names": ",".join(physicals),
                            },
                            "inputs": physicals,
                            "out": tmp_name,
                        }
                        new_jobs.append(ast_job)
                        new_inputs.append(tmp_name)
                    else:
                        new_inputs.append(inp)
                else:
                    new_inputs.append(inp)

            new_job = dict(job)
            if needs_transform:
                new_job["inputs"] = new_inputs
            new_jobs.append(new_job)

        return new_jobs


# --- Internal helpers ---

def _substitute_fields(node, substitutions: dict):
    """Replace Field nodes with substituted AST nodes."""
    from Runtime._lib.ast import (
        Field as AstField, Const as AstConst, Unary as AstUnary,
        Binary as AstBinary, Call as AstCall, Conditional as AstConditional,
    )

    if isinstance(node, AstField):
        sub = substitutions.get(node.name)
        if sub is not None:
            return sub
        return node

    if isinstance(node, AstConst):
        return node

    if isinstance(node, AstUnary):
        expr = _substitute_fields(node.expr, substitutions)
        return AstUnary(node.op, expr) if expr is not node.expr else node

    if isinstance(node, AstBinary):
        left = _substitute_fields(node.left, substitutions)
        right = _substitute_fields(node.right, substitutions)
        if left is node.left and right is node.right:
            return node
        return AstBinary(node.op, left, right)

    if isinstance(node, AstCall):
        args = [_substitute_fields(a, substitutions) for a in node.args]
        if all(a is orig for a, orig in zip(args, node.args)):
            return node
        return AstCall(node.name, args)

    if isinstance(node, AstConditional):
        cond = _substitute_fields(node.cond, substitutions)
        then_expr = _substitute_fields(node.then_expr, substitutions)
        else_expr = _substitute_fields(node.else_expr, substitutions)
        if cond is node.cond and then_expr is node.then_expr and else_expr is node.else_expr:
            return node
        return AstConditional(cond, then_expr, else_expr)

    return node


# --- Default QuoteTable instance ---

_default_qt = None


def get_default(path=None):
    global _default_qt
    if _default_qt is None:
        _default_qt = QuoteTable()
        if path:
            _default_qt.load_file(path)
        else:
            _default_qt.load_string("""
[layout]
physical = ["low", "dHigh", "dOpen", "dClose", "volume"]

[field.high]
expression = "low + dHigh"
description = "High = low + delta high"

[field.open]
expression = "low + dOpen"
description = "Open = low + delta open"

[field.close]
expression = "low + dClose"
description = "Close = low + delta close"

[field.hl2]
expression = "(high + low) * 0.5"
description = "HL/2 midpoint"

[field.oc2]
expression = "(open + close) * 0.5"
description = "OC/2 midpoint"

[field.typical]
expression = "(high + low + close) / 3.0"
description = "Typical price (HLC/3)"
""")
    return _default_qt
