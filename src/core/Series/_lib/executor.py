# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Executor — plan and run a LazyExpr DAG.

Flattens the expression tree into a linear sequence of ops,
then executes them on the available backend (numpy CPU, or GPU via Runtime+Compute).
"""

import numpy as np
from _core.backend import get_xp, get_active_name

xp = get_xp()


def _plan(expr):
    """Flatten a LazyExpr tree into a topologically-sorted list of _Op.

    Returns:
        list of (op_name, [input_ids])
    """
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
            op_name = node.op
            if op_name == 'cmp':
                # Encode comparison variant into plan-level op name
                # ("cmp_gt" ...) — consumed by CPU/GPU executors below.
                op_name = f"cmp_{getattr(node, 'cmp_op', None) or 'gt'}"
            ops.append((op_name, dep_ids))
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
        arr = xp.asarray(leaf.data(), dtype=xp.float32)
        return arr
    return leaf


def _to_f32(arr):
    """Convert array to float32 for GPU compatibility."""
    return xp.asarray(arr, dtype=xp.float32)


def _fmod_trunc(a, b):
    """C-style fmod: a - trunc(a/b)*b (WGSL % parity), NOT floor-mod.

    np.fmod is trunc semantics; np.mod is floor semantics.
    Guard mirrors div (S50): b == 0 -> +0.0 (GPU WGSL % by zero is NaN).
    """
    out = np.fmod(a, b)
    return np.where(np.asarray(b) == 0, np.float32(0.0), out)


def _execute_plan(plan, leaves):
    """Execute a flattened plan via numpy (CPU).

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
    "square": xp.square,
}

_BINARY_FUNCS = {
    "add": xp.add,
    "sub": xp.subtract,
    "mul": xp.multiply,
    "truediv": xp.divide,
    "pow": xp.power,
    "mod": _fmod_trunc,
}

# Compare op codes — mirror math/Compute/_lib/compare/descriptor.py
_COMPARE_CODES = {"gt": 0.0, "ge": 1.0, "lt": 2.0,
                  "le": 3.0, "eq": 4.0, "ne": 5.0}

_CMP_FUNCS = {
    "gt": xp.greater,
    "ge": xp.greater_equal,
    "lt": xp.less,
    "le": xp.less_equal,
    "eq": xp.equal,
    "ne": xp.not_equal,
}


def _execute_op_numpy(op_name, *args):
    """Execute a single operation on numpy arrays."""
    if op_name.startswith("cmp_"):
        cmp_fn = _CMP_FUNCS.get(op_name[4:])
        if cmp_fn is None:
            raise ValueError(f"Unknown operation: {op_name}")
        a, b = args
        if isinstance(a, (int, float)):
            a = xp.asarray([a], dtype=xp.float32)
        if isinstance(b, (int, float)):
            b = xp.asarray([b], dtype=xp.float32)
        return _to_f32(cmp_fn(a, b))

    if op_name in _UNARY_FUNCS:
        a = args[0]
        if isinstance(a, (int, float)):
            a = xp.asarray([a], dtype=xp.float32)
        return _to_f32(_UNARY_FUNCS[op_name](a))

    if op_name in _BINARY_FUNCS:
        a, b = args
        if isinstance(a, (int, float)):
            a = xp.asarray([a], dtype=xp.float32)
        if isinstance(b, (int, float)):
            b = xp.asarray([b], dtype=xp.float32)
        fn = _BINARY_FUNCS[op_name]
        return _to_f32(fn(a, b))

    raise ValueError(f"Unknown operation: {op_name}")


# ── GPU execution via Runtime + Compute ─────────────────────────────

# Mapping from LazyExpr op names to Compute kernel + params
_UNARY_MAP = {
    "sin": 0,
    "cos": 1,
    "exp": 2,
    "sqrt": 3,
    "log": 4,
    "abs": 5,
    "neg": 6,
    "square": 7,
}

_BINARY_MAP = {
    "add": 0,
    "sub": 1,
    "mul": 2,
    "truediv": 3,
    "mod": 6,
}


def _all_gpu_compatible(plan):
    """Check if all operations in the plan have GPU kernels."""
    for op_name, _ in plan:
        if op_name in _UNARY_MAP or op_name in _BINARY_MAP:
            continue
        if op_name.startswith("cmp_") and op_name[4:] in _COMPARE_CODES:
            continue
        return False
    return True


def _collect_source_data(plan, leaves):
    """Collect all leaf arrays from plan and leaves dict into source_data dict.

    Returns (source_data, op_output_names) where source_data maps names to
    numpy arrays, and op_output_names maps (tag, val) to input names.
    """
    source_data = {}
    name_map = {}  # id(val) -> name

    # Add known leaves (NumericSeries)
    for leaf_id, leaf_val in leaves.items():
        name = f"_leaf_{leaf_id}"
        arr = _eval_leaf(leaf_val)
        if isinstance(arr, np.ndarray):
            source_data[name] = arr.astype(np.float32)
        else:
            source_data[name] = np.asarray(arr, dtype=np.float32)
        name_map[("series", leaf_id)] = name
        name_map[id(leaf_val)] = name

    # Scan plan for "unknown" leaves (lists, plain data)
    for op_name, dep_ids in plan:
        for dep_id in dep_ids:
            if isinstance(dep_id, tuple):
                tag, val = dep_id
                if tag == "unknown":
                    vid = id(val)
                    if vid not in name_map:
                        name = f"_leaf_{vid}"
                        arr = np.asarray(val, dtype=np.float32)
                        source_data[name] = arr
                        name_map[vid] = name
                        name_map[("unknown", vid)] = name
                elif tag == "series":
                    sid = id(val)
                    if sid not in name_map:
                        name = f"_leaf_{sid}"
                        arr = _eval_leaf(val)
                        source_data[name] = np.asarray(arr, dtype=np.float32)
                        name_map[sid] = name
                        name_map[("series", sid)] = name

    return source_data, name_map


def _resolve_dep_id(dep_id, name_map):
    """Resolve a dep_id to ('input', name) or ('scalar', value) or ('op', name)."""
    if isinstance(dep_id, tuple):
        tag, val = dep_id
        if tag == "op":
            return ("op", f"_op_{val}")
        elif tag == "scalar":
            return ("scalar", float(val))
        elif tag == "series":
            name = name_map.get(id(val))
            if name is None:
                name = name_map.get(("series", id(val)))
            if name is None:
                # Evaluate on the fly
                arr = _eval_leaf(val)
                name = f"_leaf_{id(val)}"
                # This would need to add to source_data... but we already collected
                # Fall back: return as unknown
                return ("input", name)
            return ("input", name)
        elif tag == "unknown":
            name = name_map.get(id(val))
            if name is None:
                name = name_map.get(("unknown", id(val)))
            if name:
                return ("input", name)
            # Fall back: make array on the fly
            return ("input", str(val))
    return ("input", str(dep_id))


def _execute_plan_gpu(plan, leaves):
    """Execute plan on GPU via Runtime + Compute pipeline.

    Returns list of float values.
    Falls back to CPU if any op is not GPU-compatible.
    """
    if not _all_gpu_compatible(plan):
        return _execute_plan(plan, leaves)

    if not plan:
        for leaf_id, leaf_val in leaves.items():
            return list(np.asarray(_eval_leaf(leaf_val), dtype=np.float32))
        return []

    # Collect source data
    source_data, name_map = _collect_source_data(plan, leaves)

    # Build jobs
    jobs = []
    for op_idx, (op_name, dep_ids) in enumerate(plan):
        out_name = f"_op_{op_idx}"

        if op_name in _UNARY_MAP:
            inputs = []
            for dep_id in dep_ids:
                kind, val = _resolve_dep_id(dep_id, name_map)
                if kind == "input":
                    inputs.append(val)
                elif kind == "scalar":
                    # Map doesn't support scalars — fall back to CPU
                    return _execute_plan(plan, leaves)
                elif kind == "op":
                    inputs.append(val)

            func_code = _UNARY_MAP[op_name]
            jobs.append({
                "op": "Map",
                "inputs": inputs,
                "params": {"func": func_code},
                "out": out_name,
            })

        elif op_name in _BINARY_MAP or op_name.startswith("cmp_"):
            if op_name in _BINARY_MAP:
                kernel = "MapBinary"
                op_code = _BINARY_MAP[op_name]
            else:
                # Compare primitive (L1): numeric op code, u32 0/1 output.
                kernel = "Compare"
                op_code = _COMPARE_CODES[op_name[4:]]
            inputs = []
            params = {"op": op_code}
            scalars = []
            scalar_pos = []

            for dep_idx, dep_id in enumerate(dep_ids):
                kind, val = _resolve_dep_id(dep_id, name_map)
                if kind == "input":
                    inputs.append(val)
                elif kind == "scalar":
                    scalars.append(val)
                    scalar_pos.append(dep_idx)
                elif kind == "op":
                    inputs.append(val)

            # Scalar mode via uniforms. Operand POSITION decides the slot:
            # scalar on the left operand -> scalar_a (e.g. 2 / s, 10 - s),
            # scalar on the right operand -> scalar_b (e.g. s * 2).
            if len(scalars) == 1:
                if scalar_pos[0] == 0:
                    params["scalar_a"] = scalars[0]
                    params["use_scalar_a"] = 1
                else:
                    params["scalar_b"] = scalars[0]
                    params["use_scalar_b"] = 1
            elif len(scalars) == 2 and kernel == "MapBinary":
                params["scalar_a"] = scalars[0]
                params["use_scalar_a"] = 1
                params["scalar_b"] = scalars[1]
                params["use_scalar_b"] = 1

            jobs.append({
                "op": kernel,
                "inputs": inputs,
                "params": params,
                "out": out_name,
            })

    # Execute via Runtime + WebGpuDriver
    from Runtime._lib.Drivers.WebGPU import WebGpuDriver
    from Runtime import Runtime as RuntimeClass
    from Compute import register_all

    driver = WebGpuDriver()
    rt = RuntimeClass(driver=driver)
    register_all(rt)

    tasks = rt.compile(jobs)
    rt.execute(tasks, source_data)

    last_out = f"_op_{len(plan) - 1}"
    result = rt.driver.resolve_output(last_out)

    driver.release()

    return list(result)


# ── Public entry point ──────────────────────────────────────────────


def execute(expr):
    """Plan and execute a LazyExpr expression, return result as list.

    Uses GPU (via Runtime + Compute) when wgpu backend is active,
    otherwise uses CPU (numpy).

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

    # Decide path based on active backend
    active = get_active_name()
    if active == "wgpu":
        return _execute_plan_gpu(plan, leaves)
    return _execute_plan(plan, leaves)
