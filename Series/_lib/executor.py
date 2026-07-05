# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Executor — plan and run a LazyExpr DAG.

Flattens the expression tree into a linear sequence of ops,
then executes them on the available backend (numpy CPU, later WGSL).
"""

from _core.backend import get_xp, get_active_name

xp = get_xp()


def _plan(expr):
    """Flatten a LazyExpr tree into a topologically-sorted list of _Op.

    Returns:
        list of (op_name, [input_ids], depends_on_scalar)
    """
    import operator

    ops = []
    cache = {}

    def _visit(node):
        if id(node) in cache:
            return cache[id(node)]
        if isinstance(node, (int, float)):
            cache[id(node)] = ("scalar", node)
            return ("scalar", node)
        if hasattr(node, 'data') and callable(node.data) and not hasattr(node, 'op'):
            # NumericSeries leaf — extract on execution
            cache[id(node)] = ("series", node)
            return ("series", node)
        if hasattr(node, 'op') and hasattr(node, 'operands'):
            # LazyExpr node
            dep_ids = [_visit(op) for op in node.operands]
            op_id = len(ops)
            ops.append((node.op, dep_ids))
            cache[id(node)] = ("op", op_id)
            return ("op", op_id)
        cache[id(node)] = ("unknown", node)
        return ("unknown", node)

    _visit(expr)
    return ops


def _eval_leaf(leaf):
    """Evaluate a leaf node to a numpy array."""
    if isinstance(leaf, (int, float)):
        return float(leaf)
    if hasattr(leaf, 'data') and callable(leaf.data):
        arr = xp.asarray(leaf.data(), dtype=xp.float64)
        return arr
    return leaf


def _to_f32(arr):
    """Convert array to float32 for GPU compatibility."""
    return xp.asarray(arr, dtype=xp.float32)


def _execute_plan(plan, leaves):
    """Execute a flattened plan and return the final result as list.

    Args:
        plan: list of (op_name, dep_ids)
        leaves: dict mapping leaf id to value

    Returns:
        list of float values
    """
    results = {}

    for leaf_id, leaf_val in leaves.items():
        results[leaf_id] = _eval_leaf(leaf_val)

    for op_idx, (op_name, dep_ids) in enumerate(plan):
        args = []
        for dep_id in dep_ids:
            if isinstance(dep_id, tuple):
                tag, val = dep_id
                if tag == "op":
                    args.append(results[("result", val)])
                elif tag == "scalar":
                    args.append(val)
                elif tag == "series":
                    args.append(_eval_leaf(val))
                elif tag == "unknown":
                    args.append(val)
                else:
                    args.append(val)
            elif dep_id in results:
                args.append(results[dep_id])
            else:
                args.append(dep_id)

        result = _execute_op_numpy(op_name, *args)
        results[("result", op_idx)] = result

    # Last result is the final output
    final = results[("result", len(plan) - 1)]
    return list(final)


_UNARY_FUNCS = {
    "sin": xp.sin,
    "cos": xp.cos,
    "tan": xp.tan,
    "exp": xp.exp,
    "log": xp.log,
    "sqrt": xp.sqrt,
    "neg": xp.negative,
    "abs": xp.abs,
}

_BINARY_FUNCS = {
    "add": xp.add,
    "sub": xp.subtract,
    "mul": xp.multiply,
    "truediv": xp.divide,
    "pow": xp.power,
}


def _execute_op_numpy(op_name, *args):
    """Execute a single operation on numpy arrays."""
    if op_name in _UNARY_FUNCS:
        a = args[0]
        if isinstance(a, (int, float)):
            a = xp.asarray([a], dtype=xp.float64)
        return _to_f32(_UNARY_FUNCS[op_name](a))

    if op_name in _BINARY_FUNCS:
        a, b = args
        if isinstance(a, (int, float)):
            a = xp.asarray([a], dtype=xp.float64)
        if isinstance(b, (int, float)):
            b = xp.asarray([b], dtype=xp.float64)
        fn = _BINARY_FUNCS[op_name]
        return _to_f32(fn(a, b))

    raise ValueError(f"Unknown operation: {op_name}")


def execute(expr):
    """Plan and execute a LazyExpr expression, return result as list.

    Args:
        expr: LazyExpr tree (or NumericSeries leaf)

    Returns:
        list of float values
    """
    plan = _plan(expr)

    # Collect leaf values
    leaves = {}
    seen = set()

    def _collect_leaves(node):
        nid = id(node)
        if nid in seen:
            return
        seen.add(nid)
        if isinstance(node, (int, float, list)):
            return
        if hasattr(node, 'data') and callable(node.data):
            leaves[nid] = node
            return
        if hasattr(node, 'operands'):
            for op in node.operands:
                _collect_leaves(op)

    _collect_leaves(expr)

    return _execute_plan(plan, leaves)
