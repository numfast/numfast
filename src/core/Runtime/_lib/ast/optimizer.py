"""AST optimizer: constant folding, CSE, canonical forms.

Usage:
    node = parse_expr("(2 + 3) * high")
    opt = optimize(node)
    # Result: Const(5) * Field(high)
"""

from .ast_nodes import (
    AstNode, AstVisitor,
    Field, Const, Unary, UnaryOp, Binary, BinaryOp, Call, Conditional,
    TmpRef,
)
from typing import Optional


# ============================================================
# Constant Folding
# ============================================================

def fold_constants(node: AstNode) -> AstNode:
    """Fold constant subexpressions.

    Examples:
        Const(2) + Const(3)  -> Const(5)
        Const(0) + Field(x)  -> Field(x)
        Const(1) * Field(x)  -> Field(x)
        Const(0) * Field(x)  -> Const(0)
    """
    return _ConstantFolder().fold(node)


class _ConstantFolder(AstVisitor):
    """Walk AST bottom-up, folding constants."""

    def fold(self, node: AstNode) -> AstNode:
        return self.visit(node)

    def visit_Field(self, node: Field) -> AstNode:
        return node

    def visit_TmpRef(self, node: TmpRef) -> AstNode:
        return node

    def visit_Const(self, node: Const) -> AstNode:
        return node

    def visit_Unary(self, node: Unary) -> AstNode:
        expr = self.visit(node.expr)
        if isinstance(expr, Const):
            v = expr.value
            if node.op == UnaryOp.NEG:
                return Const(-v)
            if node.op == UnaryOp.NOT:
                return Const(1.0 if v == 0.0 else 0.0)
        if expr is node.expr:
            return node
        return Unary(node.op, expr)

    def visit_Binary(self, node: Binary) -> AstNode:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(left, Const) and isinstance(right, Const):
            return self._fold_binary_const(node.op, left.value, right.value)

        op = node.op

        if isinstance(left, Const) and left.value == 0.0:
            if op == BinaryOp.ADD:
                return right
            if op == BinaryOp.SUB:
                return Unary(UnaryOp.NEG, right) if not isinstance(right, Const) else Const(-right.value)
            if op == BinaryOp.MUL:
                return Const(0.0)
            if op == BinaryOp.DIV:
                return Const(0.0)

        if isinstance(right, Const) and right.value == 0.0:
            if op == BinaryOp.ADD:
                return left
            if op == BinaryOp.SUB:
                return left
            if op == BinaryOp.MUL:
                return Const(0.0)

        if isinstance(left, Const) and left.value == 1.0:
            if op == BinaryOp.MUL:
                return right
        if isinstance(right, Const) and right.value == 1.0:
            if op == BinaryOp.MUL:
                return left
            if op == BinaryOp.DIV:
                return left

        if op in (BinaryOp.ADD, BinaryOp.MUL) and isinstance(left, Const) and not isinstance(right, Const):
            left, right = right, left

        if left is node.left and right is node.right:
            return node
        return Binary(op, left, right)

    def _fold_binary_const(self, op: BinaryOp, a: float, b: float) -> Const:
        if op == BinaryOp.ADD: return Const(a + b)
        if op == BinaryOp.SUB: return Const(a - b)
        if op == BinaryOp.MUL: return Const(a * b)
        if op == BinaryOp.DIV: return Const(a / b) if b != 0.0 else Const(float('inf'))
        if op == BinaryOp.LT:  return Const(1.0 if a < b else 0.0)
        if op == BinaryOp.GT:  return Const(1.0 if a > b else 0.0)
        if op == BinaryOp.LE:  return Const(1.0 if a <= b else 0.0)
        if op == BinaryOp.GE:  return Const(1.0 if a >= b else 0.0)
        if op == BinaryOp.EQ:  return Const(1.0 if a == b else 0.0)
        if op == BinaryOp.NE:  return Const(1.0 if a != b else 0.0)
        if op == BinaryOp.AND: return Const(1.0 if a != 0.0 and b != 0.0 else 0.0)
        if op == BinaryOp.OR:  return Const(1.0 if a != 0.0 or b != 0.0 else 0.0)
        return Const(a + b)

    def visit_Call(self, node: Call) -> AstNode:
        args = [self.visit(a) for a in node.args]
        if all(isinstance(a, Const) for a in args):
            values = [a.value for a in args]
            return self._fold_call(node.name, values)
        if all(a is orig for a, orig in zip(args, node.args)):
            return node
        return Call(node.name, args)

    def _fold_call(self, name: str, values: list[float]) -> Const:
        import math
        fns = {
            'abs': lambda v: abs(v[0]),
            'max': lambda v: max(v),
            'min': lambda v: min(v),
            'sqrt': lambda v: math.sqrt(v[0]),
            'sign': lambda v: 1.0 if v[0] > 0 else (-1.0 if v[0] < 0 else 0.0),
            'floor': lambda v: math.floor(v[0]),
            'ceil': lambda v: math.ceil(v[0]),
            'round': lambda v: round(v[0]),
            'exp': lambda v: math.exp(v[0]),
            'log': lambda v: math.log(v[0]),
            'pow': lambda v: math.pow(v[0], v[1]) if len(v) > 1 else v[0],
        }
        fn = fns.get(name)
        if fn:
            try:
                return Const(fn(values))
            except (ValueError, ZeroDivisionError):
                pass
        return Call(name, [Const(v) for v in values])

    def visit_Conditional(self, node: Conditional) -> AstNode:
        cond = self.visit(node.cond)
        then_expr = self.visit(node.then_expr)
        else_expr = self.visit(node.else_expr)
        if isinstance(cond, Const):
            return then_expr if cond.value != 0.0 else else_expr
        if cond is node.cond and then_expr is node.then_expr and else_expr is node.else_expr:
            return node
        return Conditional(cond, then_expr, else_expr)


# ============================================================
# Common Subexpression Elimination (CSE)
# ============================================================

def _node_hash(node: AstNode) -> int:
    """Compute a content hash for CSE detection.
    
    Two nodes with the same hash are candidates for CSE.
    Field/Const names/values are included in the hash.
    """
    if isinstance(node, Field):
        return hash(('Field', node.name))
    if isinstance(node, Const):
        return hash(('Const', node.value))
    if isinstance(node, TmpRef):
        return hash(('TmpRef', node.name))
    if isinstance(node, Unary):
        return hash(('Unary', node.op.value, _node_hash(node.expr)))
    if isinstance(node, Binary):
        return hash(('Binary', node.op.value, _node_hash(node.left), _node_hash(node.right)))
    if isinstance(node, Call):
        return hash(('Call', node.name, tuple(_node_hash(a) for a in node.args)))
    if isinstance(node, Conditional):
        return hash(('Conditional', _node_hash(node.cond),
                     _node_hash(node.then_expr), _node_hash(node.else_expr)))
    return hash(id(node))


def _node_equal(a: AstNode, b: AstNode) -> bool:
    """Structural equality for CSE detection."""
    if type(a) is not type(b):
        return False
    if isinstance(a, Field):
        return a.name == b.name
    if isinstance(a, Const):
        return a.value == b.value
    if isinstance(a, TmpRef):
        return a.name == b.name
    if isinstance(a, Unary):
        return a.op == b.op and _node_equal(a.expr, b.expr)
    if isinstance(a, Binary):
        return a.op == b.op and _node_equal(a.left, b.left) and _node_equal(a.right, b.right)
    if isinstance(a, Call):
        return a.name == b.name and len(a.args) == len(b.args) \
               and all(_node_equal(x, y) for x, y in zip(a.args, b.args))
    if isinstance(a, Conditional):
        return _node_equal(a.cond, b.cond) and _node_equal(a.then_expr, b.then_expr) \
               and _node_equal(a.else_expr, b.else_expr)
    return a is b


def _is_cse_candidate(node: AstNode) -> bool:
    """Check if a node is worth CSE elimination.
    
    Skip trivial nodes: Field, Const, TmpRef (single tokens).
    Only eliminate compound expressions (Binary, Unary, Call).
    """
    # Skip single tokens — they're already cheap
    if isinstance(node, (Field, Const, TmpRef)):
        return False
    # Skip unary negation — too cheap
    if isinstance(node, Unary) and node.op == UnaryOp.NEG:
        return False
    return True


def cse(node: AstNode) -> tuple[AstNode, list[tuple[str, AstNode]]]:
    """Common Subexpression Elimination.

    Detects duplicate subtrees and replaces them with TmpRef nodes.

    Args:
        node: Root AST node

    Returns:
        (optimized_node, bindings)
        bindings: list of (temp_name, subexpression) pairs to hoist

    Example:
        (low + dHigh + low) * 0.5
        ->
        bindings=[("_cse_0", Binary(ADD, Field("low"), Field("dHigh")))],
        result = Binary(MUL,
                    Binary(ADD, TmpRef("_cse_0"), Field("low")),
                    Const(0.5))
    """
    # Step 1: Walk bottom-up, collect all candidate nodes with their hashes
    candidates: list[AstNode] = []
    _collect_candidates(node, candidates)

    # Step 2: Find duplicates
    # Group by hash, then check structural equality
    hash_groups: dict[int, list[AstNode]] = {}
    for c in candidates:
        h = _node_hash(c)
        if h not in hash_groups:
            hash_groups[h] = []
        hash_groups[h].append(c)

    # Build substitution map: duplicate_node -> original_node
    substitutions: dict[int, AstNode] = {}  # id(node) -> original
    temp_counter = 0
    bindings: list[tuple[str, AstNode]] = []

    for h, group in hash_groups.items():
        if len(group) < 2:
            continue
        # Find unique subtrees within this hash group
        unique: list[AstNode] = []
        for g in group:
            is_dup = False
            for u in unique:
                if _node_equal(g, u):
                    is_dup = True
                    substitutions[id(g)] = u
                    break
            if not is_dup:
                unique.append(g)

        if len(group) < 2:
            continue

        # Keep first occurrence, replace rest with TmpRef
        first = unique[0]
        tmp_name = f"_cse_{temp_counter}"
        temp_counter += 1
        bindings.append((tmp_name, first))
        substitutions[id(first)] = first  # self-reference (will be replaced with TmpRef)

        for u in unique[1:]:
            substitutions[id(u)] = first
        for g in group:
            if g is not first and id(g) not in substitutions:
                substitutions[id(g)] = first

    # Step 3: Apply substitutions
    # For the first occurrence, replace with TmpRef
    # For duplicates, replace with TmpRef
    # For non-duplicates, keep as-is

    # Actually, let's redo: the first unique should become TmpRef,
    # and all positions that reference it should also use TmpRef

    if not bindings:
        return (node, [])

    # Build the set of nodes to replace with TmpRef
    replace_set: dict[int, str] = {}  # id -> tmp_name
    for tmp_name, first_expr in bindings:
        # All nodes that equal this expression become TmpRef(tmp_name)
        replace_set[id(first_expr)] = tmp_name
        for c in candidates:
            if _node_equal(c, first_expr) and id(c) != id(first_expr):
                replace_set[id(c)] = tmp_name

    # Apply replacements
    result = _apply_cse(node, replace_set)
    return (result, bindings)


def _collect_candidates(node: AstNode, out: list[AstNode]):
    """Collect all CSE candidate nodes (compound expressions)."""
    if _is_cse_candidate(node):
        out.append(node)
    if isinstance(node, Unary):
        _collect_candidates(node.expr, out)
    elif isinstance(node, Binary):
        _collect_candidates(node.left, out)
        _collect_candidates(node.right, out)
    elif isinstance(node, Call):
        for a in node.args:
            _collect_candidates(a, out)
    elif isinstance(node, Conditional):
        _collect_candidates(node.cond, out)
        _collect_candidates(node.then_expr, out)
        _collect_candidates(node.else_expr, out)


def _apply_cse(node: AstNode, replace_set: dict[int, str]) -> AstNode:
    """Replace CSE-duplicated nodes with TmpRef."""
    nid = id(node)
    if nid in replace_set:
        return TmpRef(replace_set[nid])

    if isinstance(node, Field):
        return node
    if isinstance(node, Const):
        return node
    if isinstance(node, TmpRef):
        return node
    if isinstance(node, Unary):
        expr = _apply_cse(node.expr, replace_set)
        return Unary(node.op, expr) if expr is not node.expr else node
    if isinstance(node, Binary):
        left = _apply_cse(node.left, replace_set)
        right = _apply_cse(node.right, replace_set)
        if left is node.left and right is node.right:
            return node
        return Binary(node.op, left, right)
    if isinstance(node, Call):
        args = [_apply_cse(a, replace_set) for a in node.args]
        if all(a is orig for a, orig in zip(args, node.args)):
            return node
        return Call(node.name, args)
    if isinstance(node, Conditional):
        cond = _apply_cse(node.cond, replace_set)
        then_expr = _apply_cse(node.then_expr, replace_set)
        else_expr = _apply_cse(node.else_expr, replace_set)
        if cond is node.cond and then_expr is node.then_expr and else_expr is node.else_expr:
            return node
        return Conditional(cond, then_expr, else_expr)
    return node


# ============================================================
# Public API
# ============================================================

def optimize(node: AstNode) -> AstNode:
    """Run all optimization passes on AST node.

    Pipeline:
        1. Constant folding
        2. CSE (Common Subexpression Elimination)
        3. Future: DCE, canonicalization

    Returns optimized AST (with TmpRef nodes if CSE applied).
    Caller is responsible for emitting bindings before the expression.
    """
    node = fold_constants(node)
    node, _ = cse(node)  # CSE returns (node, bindings); node already has TmpRefs
    return node


def optimize_with_bindings(node: AstNode) -> tuple[AstNode, list[tuple[str, AstNode]]]:
    """Run all optimization passes and return CSE bindings.

    For cases where the caller needs to emit the let bindings
    before the expression (e.g., WGSL preamble generation).

    Returns:
        (optimized_ast, [(tmp_name, subexpression), ...])
    """
    node = fold_constants(node)
    node, bindings = cse(node)
    return (node, bindings)


__all__ = ["fold_constants", "optimize", "optimize_with_bindings", "cse"]
