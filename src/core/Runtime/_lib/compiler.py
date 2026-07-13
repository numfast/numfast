"""Compiler — jobs -> tasks.

Compilation phases:
1. QuoteTable inline: embed logical field AST directly into WGSL preamble
   (for workgroup_size(64) kernels, skip intermediate buffer)
2. QuoteTable materialize: evaluate AST on CPU for serial kernels
3. Build tasks from jobs

Strings exist only at TOML loading. AST accessed via kernel_table["__ast__"].
"""

from typing import Optional
from .task import Task, ResourceRef
from .mod_iface import InputSlot, OutputSlot, ExecutionPlan, ExecutionContext
import re


def _cartesian_product(params: dict) -> list[dict]:
    import itertools
    array_keys = []
    array_values = []
    scalar_params = {}

    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            array_keys.append(k)
            array_values.append(v)
        else:
            scalar_params[k] = v

    if not array_keys:
        return [dict(params)]

    combos = []
    for combo_values in itertools.product(*array_values):
        combo = dict(scalar_params)
        for i, k in enumerate(array_keys):
            combo[k] = combo_values[i]
        combos.append(combo)

    return combos


def _apply_template(template: str, params: dict) -> str:
    return template.format(**params)


# --- QuoteTable Inline Resolver ---

def _extract_wgsl(entry: dict) -> Optional[str]:
    """Get WGSL string from kernel entry (resolve callable if needed)."""
    wgsl = entry["drivers"].get("wgsl")
    if wgsl is None:
        return None
    if callable(wgsl):
        return wgsl({})
    return wgsl


def _is_parallel_wgsl(wgsl: str) -> bool:
    return '@workgroup_size(64)' in wgsl


def _count_input_bindings(wgsl: str) -> int:
    return len(re.findall(r'var<storage,\s*read>', wgsl))


def _inline_field_in_kernel(
    kernel_entry: dict,
    logical_field: str,
    node,
    physicals: list[str],
    kernel_table: dict,
    new_alias: str,
):
    """Create wrapped kernel with inlined field AST expression.

    AST functions accessed via kernel_table["__ast__"].
    Optimizer is always run (mandatory pass).
    """
    ast = kernel_table.get("__ast__")
    if ast is None:
        return None

    # Step 0: Always optimize AST before inline
    opt_node, cse_bindings = ast["optimize_with_bindings"](node)

    wgsl = _extract_wgsl(kernel_entry)
    if wgsl is None or not _is_parallel_wgsl(wgsl):
        return None

    num_phys = len(physicals)
    num_orig_inputs = _count_input_bindings(wgsl)
    if num_orig_inputs < 1:
        return None

    shift = num_phys - num_orig_inputs
    if shift <= 0:
        return None

    # Step 1: Shift all @binding(N) by shift
    def _shift_binding(m):
        n = int(m.group(1))
        return f"@binding({n + shift})"

    shifted = re.sub(r'@binding\((\d+)\)', _shift_binding, wgsl)

    # Step 2: Find first input binding (now at @binding(shift))
    input_pat = rf'@group\(0\)\s*@binding\({shift}\)\s*var<storage,\s*read>\s*(\w+)\s*:\s*array<f32>;'
    m = re.search(input_pat, shifted)
    if not m:
        return None
    orig_var = m.group(1)

    # Step 3: Replace with physical input bindings at 0..(num_phys-1)
    new_bindings = '\n'.join(
        f"@group(0) @binding({i}) var<storage, read> {p}: array<f32>;"
        for i, p in enumerate(physicals)
    )

    result = shifted[:m.start()] + new_bindings + shifted[m.end():]

        # Step 4: Build WGSL preamble via AST generate function (accessed through kernel_table)
    gen_fn = ast["wgsl_gen"]

    # Emit CSE temporaries first (if any)
    preamble_lines = []
    for tmp_name, subexpr in cse_bindings:
        wgsl_sub = gen_fn(subexpr)
        preamble_lines.append(f"    let {tmp_name} = {wgsl_sub};")

    # Emit main logical field expression
    wgsl_expr = gen_fn(opt_node)
    preamble_lines.append(f"    let {logical_field} = {wgsl_expr};")
    preamble = "\n".join(preamble_lines)

    # Step 5: Insert preamble after fn main(...) {
    result = re.sub(
        r'(fn\s+main\s*\([^)]*\)\s*\{)',
        lambda m: m.group(1) + '\n' + preamble,
        result
    )

    # Step 6: Replace orig_var references in body
    result = re.sub(
        r'\b' + re.escape(orig_var) + r'\[i\]', logical_field, result
    )
    result = re.sub(
        r'arrayLength\s*\(\s*&' + re.escape(orig_var) + r'\s*\)',
        f'arrayLength(&{physicals[0]})', result
    )
    result = re.sub(
        r'\b' + re.escape(orig_var) + r'\b', logical_field, result
    )

    # Step 7: Register modified kernel
    orig_describe = kernel_entry["describe"]

    def _inlined_describe(params):
        plan = orig_describe(params)
        if plan.inputs:
            first = plan.inputs[0]
            plan.inputs = [
                InputSlot(name=p, dtype=first.dtype) for p in physicals
            ] + plan.inputs[1:]
        return plan

    kernel_table[new_alias] = {
        "describe": _inlined_describe,
        "drivers": {
            "cpu": kernel_entry["drivers"]["cpu"],
            "wgsl": result,
        },
        "abi_version": kernel_entry.get("abi_version", 1),
        "capabilities": kernel_entry.get("capabilities", {}),
    }

    return new_alias


# --- AstEval kernel (registered in kernel_table, no direct imports) ---

def _describe_ast_eval(params: dict) -> ExecutionPlan:
    expr = params.get("expr", "")
    num_vars = int(params.get("num_vars", 1))
    inputs = [InputSlot(name=f"in_{i}", dtype="float") for i in range(num_vars)]
    return ExecutionPlan(
        inputs=inputs,
        outputs=[OutputSlot(dtype="float", template="ast_expr")],
        workspace=[],
        uniforms={
            "expr": expr,
            "num_vars": float(num_vars),
            "field_names": params.get("field_names", ""),
        },
    )


def _cpu_ast_eval(ctx: ExecutionContext):
    expr = ctx.uniforms.get("expr", "0")
    field_names_str = ctx.uniforms.get("field_names", "")
    field_names = field_names_str.split(",") if field_names_str else []

    dst = ctx.outputs[0].view
    n = dst.length()

    views = {}
    for i, name in enumerate(field_names):
        if i < len(ctx.inputs):
            views[name] = ctx.inputs[i].view

    safe_globals = {
        "__builtins__": {},
        "abs": abs,
        "max": max,
        "min": min,
    }

    for i in range(n):
        local_vars = {}
        for name, view in views.items():
            local_vars[name] = view.read(i)
        try:
            result = eval(expr, safe_globals, local_vars)
        except Exception:
            result = 0.0
        dst.write(i, result)


def _register_ast_eval(kernel_table: dict):
    """Register AstEval kernel (if not already registered)."""
    if "AstEval" not in kernel_table:
        kernel_table["AstEval"] = {
            "describe": _describe_ast_eval,
            "drivers": {
                "cpu": _cpu_ast_eval,
                "wgsl": None,
            },
            "abi_version": 1,
            "capabilities": {"streaming": False, "workspace": False,
                             "multi_input": True, "multi_output": False},
        }


# --- Register AST module in kernel_table ---

def _register_ast(kernel_table: dict):
    """Register AST module functions in kernel_table["__ast__"].
    
    Builder pattern: no direct Python imports, everything via kernel_table.
    """
    if "__ast__" in kernel_table:
        return

    # Late imports — only here, at the registration boundary
    from .ast import wgsl_generate, optimize, optimize_with_bindings, cpu_expr, parse_expr, extract_physicals

    kernel_table["__ast__"] = {
        "wgsl_gen": wgsl_generate,
        "optimize": optimize,
        "optimize_with_bindings": optimize_with_bindings,
        "cpu_expr": cpu_expr,
        "parse_expr": parse_expr,
        "extract_physicals": extract_physicals,
    }


# --- QuoteTable Resolver ---

def _resolve_quote_table(jobs: list[dict], kernel_table: dict) -> list[dict]:
    """Resolve QuoteTable logical fields in jobs."""
    from .quote_table.qt_resolver import get_default
    qt = get_default()

    # Ensure AST and AstEval are registered
    _register_ast(kernel_table)
    _register_ast_eval(kernel_table)

    ast = kernel_table["__ast__"]
    cpu_expr_fn = ast["cpu_expr"]

    new_jobs = []
    inline_counter = 0
    qt_cache: dict[str, str] = {}  # field_name -> tmp_name (QuoteTable memoization)

    for job in jobs:
        raw_inputs = job.get("inputs", [])
        new_inputs = []
        inserted_jobs = []
        needs_transform = False

        for inp in raw_inputs:
            if qt.is_logical(inp):
                node, physicals = qt.resolve(inp)
                if node is not None and physicals:
                    op = job.get("op", "")
                    kernel_entry = kernel_table.get(op)

                    # Phase 1: Try inline (parallel kernels only)
                    if kernel_entry:
                        wgsl = _extract_wgsl(kernel_entry)
                        if wgsl and _is_parallel_wgsl(wgsl) and len(physicals) > 1:
                            new_alias = f"__{op}_{inp}_{inline_counter}"
                            inline_counter += 1
                            result = _inline_field_in_kernel(
                                kernel_entry, inp, node, physicals,
                                kernel_table, new_alias
                            )
                            if result:
                                job["op"] = new_alias
                                new_inputs.extend(physicals)
                                needs_transform = True
                                continue

                    # Phase 2: Materialized — AstEval with QuoteTable cache
                    if inp in qt_cache:
                        # Reuse previously materialized field
                        new_inputs.append(qt_cache[inp])
                        needs_transform = True
                        continue

                    py_expr = cpu_expr_fn(node)
                    tmp_name = f"__qt_{inp}_{inline_counter}"
                    inline_counter += 1
                    qt_cache[inp] = tmp_name  # cache for reuse

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
                    inserted_jobs.append(ast_job)
                    new_inputs.append(tmp_name)
                    needs_transform = True
                else:
                    new_inputs.append(inp)
            else:
                new_inputs.append(inp)

        new_jobs.extend(inserted_jobs)
        new_job = dict(job)
        if needs_transform:
            new_job["inputs"] = new_inputs
        new_jobs.append(new_job)

    return new_jobs


# --- Main compile ---

def compile(jobs: list[dict], kernel_table: dict) -> list[Task]:
    """Convert Job list (dict) to Task list."""
    # Step 1: QuoteTable resolution (inline + materialized)
    jobs = _resolve_quote_table(jobs, kernel_table)

    tasks: list[Task] = []
    symbol_table: dict[str, tuple[int, int]] = {}
    next_id = 0

    for job_idx, job in enumerate(jobs):
        op = job.get("op", "")
        raw_params = job.get("params", {})
        raw_inputs = job.get("inputs", [])
        resource = job.get("resource", "@tmp")
        out_override = job.get("out", None)

        if not op:
            raise ValueError(f"Job {job_idx}: missing 'op'")
        if op not in kernel_table:
            raise KeyError(f"Kernel '{op}' not registered. Available: {list(kernel_table.keys())}")

        kernel = kernel_table[op]
        param_combos = _cartesian_product(raw_params)

        for combo in param_combos:
            plan: ExecutionPlan = kernel["describe"](combo)

            names = []
            if out_override is not None:
                if isinstance(out_override, str):
                    names = [_apply_template(out_override, combo)]
                elif isinstance(out_override, (list, tuple)):
                    names = list(out_override)
                else:
                    names = [str(out_override)]
            else:
                for slot in plan.outputs:
                    names.append(_apply_template(slot.template, combo))

            input_refs: list[ResourceRef] = []
            for inp_str in raw_inputs:
                if inp_str in symbol_table:
                    ref_task_id, ref_out_idx = symbol_table[inp_str]
                    input_refs.append(ResourceRef(
                        type="task",
                        task_id=ref_task_id,
                        output_idx=ref_out_idx,
                    ))
                else:
                    input_refs.append(ResourceRef(
                        type="input",
                        column=inp_str,
                    ))

            task = Task(
                id=next_id,
                op=op,
                params=combo,
                inputs=input_refs,
                resource=resource,
                num_outputs=len(plan.outputs),
                out_names=names,
                workspace=list(plan.workspace),
                uniforms=dict(plan.uniforms),
                output_size_fn=plan.output_size_fn,
                dispatch=plan.dispatch,
            )
            tasks.append(task)

            for i, name in enumerate(names):
                symbol_table[name] = (next_id, i)

            next_id += 1

    return tasks
