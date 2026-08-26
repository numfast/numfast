"""Smoke tests for AST module + QuoteTable AST integration."""

import sys
import os

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
    
    # --- 1. Parser tests ---
    from Runtime._lib.ast.parser import parse_expr
    from Runtime._lib.ast.ast_nodes import Field, Const, Binary, BinaryOp, Call
    
    node = parse_expr("low + dHigh")
    check("parse: low + dHigh",
          isinstance(node, Binary) and node.op == BinaryOp.ADD
          and isinstance(node.left, Field) and node.left.name == "low"
          and isinstance(node.right, Field) and node.right.name == "dHigh")
    print(f"  OK: parse 'low + dHigh' -> {node._repr(0)}")
    
    node = parse_expr("(high + low) * 0.5")
    check("parse: (high+low)*0.5",
          isinstance(node, Binary) and node.op == BinaryOp.MUL
          and isinstance(node.left, Binary) and node.left.op == BinaryOp.ADD
          and isinstance(node.right, Const) and abs(node.right.value - 0.5) < 1e-9)
    print(f"  OK: parse '(high+low)*0.5' -> {node._repr(0)}")
    
    node = parse_expr("max(high, low)")
    check("parse: max()",
          isinstance(node, Call) and node.name == "max"
          and len(node.args) == 2
          and isinstance(node.args[0], Field) and node.args[0].name == "high")
    print(f"  OK: parse 'max(high, low)' -> {node._repr(0)}")
    
    node = parse_expr("3.0")
    check("parse: const float",
          isinstance(node, Const) and abs(node.value - 3.0) < 1e-9)
    
    node = parse_expr("42")
    check("parse: const int",
          isinstance(node, Const) and abs(node.value - 42.0) < 1e-9)
    
    # --- 2. WGSL Generator tests ---
    from Runtime._lib.ast.wgsl_gen import generate as wgsl_gen
    
    node = parse_expr("low + dHigh")
    wgsl = wgsl_gen(node, id_var="id.x")
    check("wgsl: low + dHigh",
          wgsl == "low[id.x] + dHigh[id.x]")
    print(f"  OK: wgsl 'low + dHigh' -> {wgsl}")
    
    node = parse_expr("(high + low) * 0.5")
    wgsl = wgsl_gen(node, id_var="id.x")
    check("wgsl: (high+low)*0.5",
          wgsl == "(high[id.x] + low[id.x]) * 0.5")
    print(f"  OK: wgsl '(high+low)*0.5' -> {wgsl}")
    
    node = parse_expr("abs(high - low)")
    wgsl = wgsl_gen(node, id_var="id.x")
    check("wgsl: abs(high-low)",
          "abs" in wgsl and "high[id.x]" in wgsl and "low[id.x]" in wgsl)
    print(f"  OK: wgsl 'abs(high-low)' -> {wgsl}")
    
    # --- 3. Optimizer tests ---
    from Runtime._lib.ast.optimizer import fold_constants, optimize
    
    node = parse_expr("2 + 3")
    opt = fold_constants(node)
    check("optimize: 2+3 -> 5",
          isinstance(opt, Const) and abs(opt.value - 5.0) < 1e-9)
    print(f"  OK: optimize '2+3' -> {opt._repr(0)}")
    
    node = parse_expr("1 * high + 0")
    opt = optimize(node)
    check("optimize: 1*high+0 -> high",
          isinstance(opt, Field) and opt.name == "high")
    print(f"  OK: optimize '1*high+0' -> {opt._repr(0)}")
    
    node = parse_expr("(2 + 3) * high")
    opt = optimize(node)
    has_const5 = (isinstance(opt.left, Const) and abs(opt.left.value - 5.0) < 1e-9) or (isinstance(opt.right, Const) and abs(opt.right.value - 5.0) < 1e-9)
    has_field_high = (isinstance(opt.left, Field) and opt.left.name == "high") or (isinstance(opt.right, Field) and opt.right.name == "high")
    check("optimize: (2+3)*high -> 5*high",
          isinstance(opt, Binary) and opt.op == BinaryOp.MUL and has_const5 and has_field_high)
    print(f"  OK: optimize '(2+3)*high' -> {opt._repr(0)}")
    
    # --- 4. QuoteTable integration test ---
    from Runtime._lib.quote_table.qt_resolver import get_default, QuoteTable
    from Runtime._lib.ast import extract_physicals
    
    qt = get_default()
    node, physicals = qt.resolve("hl2")
    check("qt: hl2 resolves",
          node is not None and len(physicals) > 0)
    print(f"  OK: qt resolve hl2 -> physicals={physicals}")
    print(f"  OK: qt resolve hl2 -> AST={node._repr(0)}")
    
    # Physicals should contain low and dHigh (not high)
    check("qt: hl2 physicals contain low",
          "low" in physicals)
    check("qt: hl2 physicals contain dHigh",
          "dHigh" in physicals)
    check("qt: hl2 physicals do NOT contain high (virtual)",
          "high" not in physicals)
    
    # Generate WGSL from resolved AST
    wgsl = wgsl_gen(node, id_var="id.x")
    print(f"  OK: wgsl hl2 -> {wgsl}")
    check("qt: hl2 wgsl contains low[id.x]",
          "low[id.x]" in wgsl)
    check("qt: hl2 wgsl contains dHigh[id.x]",
          "dHigh[id.x]" in wgsl)
    check("qt: hl2 wgsl contains 0.5",
          "0.5" in wgsl)
    
    # Test close resolution
    node, physicals = qt.resolve("close")
    check("qt: close resolves",
          node is not None and len(physicals) > 0)
    print(f"  OK: qt resolve close -> physicals={physicals}")
    wgsl = wgsl_gen(node, id_var="id.x")
    check("qt: close wgsl contains low[id.x]",
          "low[id.x]" in wgsl)
    check("qt: close wgsl contains dClose[id.x]",
          "dClose[id.x]" in wgsl)
    print(f"  OK: wgsl close -> {wgsl}")
    
    # --- 5. Test optimized resolves ---
    opt_node = optimize(node)
    check("qt: optimize close AST preserves structure",
          isinstance(opt_node, Binary) and opt_node.op == BinaryOp.ADD)
    
    # --- 6. Test get_expression (Materialized resolver path) ---
    expr_str = qt.get_expression("hl2")
    check("qt: get_expression hl2",
          expr_str == "(high + low) * 0.5")
    print(f"  OK: qt get_expression hl2 -> {expr_str}")
    
    print(f"\nResults: {passed}/{total} passed")
    return {"passed": passed, "total": total, "failed": total - passed, "errors": errors}


if __name__ == "__main__":
    run_tests()
