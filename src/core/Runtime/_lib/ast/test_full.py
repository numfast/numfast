"""Full smoke test: AST -> all generators -> QuoteTable -> compile (no Expression)."""
import sys
import os

# Must run from numfast root
os.chdir(r'C:\App\numfast')

# Ensure paths
for p in [r'C:\App\numfast', r'C:\App\numfast\develop\backtest', r'C:\App\numfast\develop\backtest\Loaders']:
    if p not in sys.path:
        sys.path.insert(0, p)


def run_tests():
    passed = 0
    total = 0
    errors = []

    def check(name, ok):
        nonlocal passed, total
        total += 1
        if ok:
            passed += 1
        else:
            errors.append(name)
            print(f"  FAIL: {name}")

    # ====== 1. Parser ======
    from Runtime._lib.ast.parser import parse_expr, ParseError
    from Runtime._lib.ast.ast_nodes import (Field, Const, Binary, BinaryOp, 
                                            Unary, UnaryOp, Call, Conditional, TmpRef)
    
    n = parse_expr("low + dHigh")
    check("parse1: ADD", isinstance(n, Binary) and n.op == BinaryOp.ADD)
    
    n = parse_expr("(high + low) * 0.5")
    check("parse2: MUL", isinstance(n, Binary) and n.op == BinaryOp.MUL
          and isinstance(n.left, Binary) and n.left.op == BinaryOp.ADD)
    
    n = parse_expr("3.0")
    check("parse3: float", isinstance(n, Const) and abs(n.value - 3.0) < 1e-9)
    
    n = parse_expr("42")
    check("parse4: int", isinstance(n, Const) and abs(n.value - 42.0) < 1e-9)
    
    n = parse_expr("max(high, low)")
    check("parse5: call", isinstance(n, Call) and n.name == "max" and len(n.args) == 2)
    
    try:
        parse_expr("+")
        check("parse6: error", False)
    except ParseError:
        check("parse6: error", True)
    
    # ====== 2. wgsl_gen ======
    from Runtime._lib.ast.wgsl_gen import generate as wgsl_gen
    
    n = parse_expr("low + dHigh")
    w = wgsl_gen(n, id_var="id.x")
    check("wgsl1: add", w == "low[id.x] + dHigh[id.x]")
    
    n = parse_expr("(high + low) * 0.5")
    w = wgsl_gen(n, id_var="id.x")
    check("wgsl2: mul", w == "(high[id.x] + low[id.x]) * 0.5")
    
    n = parse_expr("abs(high - low)")
    w = wgsl_gen(n, id_var="id.x")
    check("wgsl3: call", "abs" in w and "high[id.x]" in w)
    
    n = parse_expr("-high")
    w = wgsl_gen(n, id_var="id.x")
    check("wgsl4: neg", "-high[id.x]" in w)
    
    # ====== 3. CpuGenerator ======
    from Runtime._lib.ast.cpu_gen import CpuGenerator, cpu_expr, cpu_eval
    
    gen_cpu = CpuGenerator()
    
    n = parse_expr("low + dHigh")
    p = gen_cpu.generate(n)
    check("cpu1: add", p == "(low + dHigh)")
    
    n = parse_expr("(high + low) * 0.5")
    p = gen_cpu.generate(n)
    check("cpu2: mul", "high" in p and "low" in p and "0.5" in p)
    
    # eval test
    n = parse_expr("a + b")
    result = cpu_eval(n, {"a": 3.0, "b": 4.0})
    check("cpu3: eval", result == 7.0)
    
    n = parse_expr("max(a, b)")
    result = cpu_eval(n, {"a": 1.0, "b": 5.0})
    check("cpu4: max", result == 5.0)
    
    # ====== 4. Optimizer ======
    from Runtime._lib.ast.optimizer import fold_constants, optimize, cse, optimize_with_bindings
    
    n = parse_expr("2 + 3")
    opt = fold_constants(n)
    check("opt1: const fold", isinstance(opt, Const) and abs(opt.value - 5.0) < 1e-9)
    
    n = parse_expr("1 * high + 0")
    opt = optimize(n)
    check("opt2: identity", isinstance(opt, Field) and opt.name == "high")
    
    n = parse_expr("(2 + 3) * high")
    opt = optimize(n)
    check("opt3: partial fold", isinstance(opt, Binary) and opt.op == BinaryOp.MUL
          and isinstance(opt.right, Const) and abs(opt.right.value - 5.0) < 1e-9)
    
    n = parse_expr("(a + b) + (a + b)")
    opt, bindings = optimize_with_bindings(n)
    check("cse1: has bindings", len(bindings) > 0)
    check("cse2: TmpRef in tree", 
          isinstance(opt, Binary) and opt.op == BinaryOp.ADD
          and isinstance(opt.left, TmpRef) and isinstance(opt.right, TmpRef))
    print(f"  CSE test: {n._repr(0)} -> bindings={[(n, b._repr(0)) for n,b in bindings]}")
    
    n = parse_expr("a + b + c")
    opt, bindings = optimize_with_bindings(n)
    # a+b+c has no duplicate, so no CSE
    check("cse3: no false CSE", len(bindings) == 0 or not isinstance(opt, TmpRef))
    print(f"  CSE unique test: {n._repr(0)} -> bindings={len(bindings)}")
    
    # ====== 5. QuoteTable AST ======
    from Runtime._lib.quote_table.qt_resolver import get_default, QuoteTable
    
    qt = get_default()
    
    node, physicals = qt.resolve("hl2")
    check("qt1: hl2 resolves", node is not None and physicals == ["low", "dHigh"])
    print(f"  QT hl2: physicals={physicals}")
    
    # Verify recursively resolved AST
    w = wgsl_gen(node, id_var="id.x")
    check("qt2: hl2 wgsl", "low[id.x]" in w and "dHigh[id.x]" in w and "0.5" in w)
    print(f"  QT hl2 wgsl: {w}")
    
    node, physicals = qt.resolve("close")
    check("qt3: close resolves", node is not None and physicals == ["low", "dClose"])
    
    w = wgsl_gen(node, id_var="id.x")
    check("qt4: close wgsl", "low[id.x]" in w and "dClose[id.x]" in w)
    print(f"  QT close wgsl: {w}")
    
    # CSE on resolved AST
    opt_node, bindings = optimize_with_bindings(node)
    w = wgsl_gen(opt_node, id_var="id.x")
    # close = low + dClose after fold_constants has no duplicates
    check("qt5: close CSE no-op", "low[id.x]" in w and "dClose[id.x]" in w)
    
    # ====== 6. AstEval registration ======
    from Runtime._lib.compiler import _register_ast_eval, _describe_ast_eval
    
    kernel_table = {}
    _register_ast_eval(kernel_table)
    
    check("asteval1: registered", "AstEval" in kernel_table)
    
    plan = _describe_ast_eval({"expr": "low + dHigh", "num_vars": 2, "field_names": "low,dHigh"})
    check("asteval2: plan has 2 inputs", len(plan.inputs) == 2)
    check("asteval3: has output", len(plan.outputs) == 1)
    check("asteval4: has uniforms", plan.uniforms.get("expr") == "low + dHigh")
    check("asteval5: field_names preserved", plan.uniforms.get("field_names") == "low,dHigh")
    
    # ====== 7. Full compile with AstEval ======
    from Runtime._lib.compiler import compile as _compile
    
    # Simulate a kernel table with a simple op
    def _describe_identity(params):
        from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan
        return ExecutionPlan(
            inputs=[InputSlot(name="x", dtype="float")],
            outputs=[OutputSlot(dtype="float", template="out")],
            workspace=[],
            uniforms={},
        )
    
    def _cpu_identity(ctx):
        dst = ctx.outputs[0].view
        src = ctx.inputs[0].view
        n = dst.length()
        for i in range(n):
            dst.write(i, src.read(i))
    
    test_table = {
        "Identity": {
            "describe": _describe_identity,
            "drivers": {"cpu": _cpu_identity, "wgsl": None},
            "abi_version": 1,
            "capabilities": {},
        },
    }
    
    # Job with logical field "close"
    test_jobs = [
        {"op": "Identity", "params": {}, "inputs": ["close"]},
    ]
    
    try:
        tasks = _compile(test_jobs, test_table)
        check("compile1: tasks created", len(tasks) > 0)
        # Should have 2 tasks: AstEval for close, then Identity
        ops = [t.op for t in tasks]
        print(f"  Compile tasks: {ops}")
        check("compile2: has AstEval", "AstEval" in ops)
        check("compile3: has Identity", "Identity" in ops)
    except Exception as e:
        check(f"compile failed: {e}", False)
        import traceback
        traceback.print_exc()
    
    # ====== Results ======
    print(f"\nResults: {passed}/{total} passed")
    for e in errors:
        print(f"  FAIL: {e}")
    return {"passed": passed, "total": total, "failed": total - passed, "errors": errors}


if __name__ == "__main__":
    result = run_tests()
    print(f"DONE: {result['passed']}/{result['total']} passed")
