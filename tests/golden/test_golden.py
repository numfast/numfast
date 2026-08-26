"""Golden Tests — compare current output vs reference.

Usage:
    python -m numfast.tests.golden.test_golden

Returns non-zero exit code on failure.
"""

import sys, os
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from Runtime import Runtime
from Trading import register_all as register_trading_kernels

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "data")


def make_runtime():
    runtime = Runtime()
    register_trading_kernels(runtime)
    return runtime


def load_manifest():
    manifest_path = os.path.join(GOLDEN_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"  No manifest found at {manifest_path}")
        print(f"  Run generate.py first!")
        return None
    with open(manifest_path, "r") as f:
        return json.load(f)


def test_golden():
    """Run all golden test cases and compare with reference."""
    manifest = load_manifest()
    if manifest is None:
        return False
    
    runtime = make_runtime()
    
    tolerance_map = {
        "EMA": 1e-12,
        "SMA": 1e-12,
        "RSI": 1e-12,
        "MACD": 1e-12,
    }
    
    all_passed = True
    
    print(f"\n{'='*60}")
    print(f"Golden Tests: {len(manifest)} cases")
    print(f"{'='*60}")
    
    for case in manifest:
        alias = case["alias"]
        params = case["params"]
        key = case["key"]
        
        input_data = {}
        for k, v_list in case["inputs"].items():
            input_data[k] = np.array(v_list, dtype=np.float64)
        
        jobs = [{"op": alias, "inputs": list(input_data.keys()), "params": params}]
        tasks = runtime.compile(jobs)
        runtime.execute(tasks, input_data)
        
        case_passed = True
        for t in tasks:
            for name in t.out_names:
                current = runtime.driver.memory.resolve_output(name)
                golden_path = os.path.join(GOLDEN_DIR, case["outputs"].get(name, ""))
                
                if not os.path.exists(golden_path):
                    print(f"  ! {key}/{name}: golden file not found: {golden_path}")
                    case_passed = False
                    continue
                
                golden = np.load(golden_path)
                
                if current.shape != golden.shape:
                    print(f"  FAIL {key}/{name}: shape mismatch {current.shape} vs {golden.shape}")
                    case_passed = False
                    continue
                
                abs_error = np.abs(current - golden).max()
                rel_error = np.abs(current - golden) / (np.abs(golden) + 1e-15)
                max_rel = rel_error.max()
                
                tol = tolerance_map.get(alias, 1e-12)
                
                status = "PASS" if abs_error < tol else "FAIL"
                if status == "FAIL":
                    case_passed = False
                
                print(f"  [{status:4s}] {key}/{name}: "
                      f"max_abs_err={abs_error:.2e} "
                      f"max_rel_err={max_rel:.2e} "
                      f"tol={tol:.0e}")
        
        if case_passed:
            print(f"    -> {key}: OK")
        else:
            print(f"    -> {key}: FAILED")
            all_passed = False
    
    print(f"{'='*60}")
    if all_passed:
        print(f"  All golden tests PASSED")
    else:
        print(f"  Some golden tests FAILED")
    print(f"{'='*60}\n")
    
    return all_passed


if __name__ == "__main__":
    success = test_golden()
    sys.exit(0 if success else 1)
