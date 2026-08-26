"""Audit: what does ExecutionGraph give us that list[Task] cannot?

This is NOT an optimizer. This is diagnostics only.
Proves the graph abstraction is useful before building Optimizer/Scheduler.

Three proofs:
1. Visualization -- DAG shows structure invisible in flat task list
2. Duplicate detection -- graph reveals CSE opportunities invisible in tasks
3. Parallelism detection -- independent branches visible in graph
"""

import os
import sys

# Repo-relative import roots (no hardcoded absolute paths)
_HERE = os.path.dirname(os.path.abspath(__file__))       # tests/runtime_internal
_CORE = os.path.abspath(os.path.join(_HERE, '..', '..', 'src', 'core'))  # src/core
_SRC = os.path.dirname(_CORE)                            # src
for p in (_CORE, os.path.join(_SRC, 'math')):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from Runtime._lib.runtime import Runtime
from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
from Runtime._lib.compiler import compile as _compile
from Runtime._lib.planner import planner
from Runtime._lib.exec_graph_serde import exec_graph_to_dot


def _register_compute_kernels(runtime):
    """Register all Compute ISA kernels into runtime.kernel_table."""
    from Compute._lib.rolling_sum import describe as rs_desc, cpu as rs_cpu
    from Compute._lib.rolling_min import describe as rmin_desc, cpu as rmin_cpu
    from Compute._lib.rolling_max import describe as rmax_desc, cpu as rmax_cpu
    from Compute._lib.map_binary import describe as mb_desc, cpu as mb_cpu
    from Compute._lib.state_kernel import describe as sk_desc, cpu as sk_cpu
    from Compute._lib.true_range import describe as tr_desc, cpu as tr_cpu

    for alias, desc, cpu_fn in [
        ("RollingSum", rs_desc, rs_cpu),
        ("RollingMin", rmin_desc, rmin_cpu),
        ("RollingMax", rmax_desc, rmax_cpu),
        ("MapBinary", mb_desc, mb_cpu),
        ("StateKernel", sk_desc, sk_cpu),
        ("TrueRange", tr_desc, tr_cpu),
    ]:
        runtime.register_kernel(alias, describe=desc, cpu=cpu_fn, abi_version=1)


def audit():
    driver = CpuDriver()
    runtime = Runtime(driver=driver)
    _register_compute_kernels(runtime)

    n = 100
    np.random.seed(42)
    high = np.random.randint(51000, 53000, n).astype(np.float64)
    low  = np.random.randint(49000, 51000, n).astype(np.float64)
    close = np.random.randint(50000, 52000, n).astype(np.float64)

    # ---- Build a pipeline with KNOWN duplicates ----
    jobs = [
        # SMA(20): RollingSum(close, 20) / 20
        # MapBinary op codes: 0=add, 1=sub, 2=mul, 3=div, 4=max, 5=min
        {"op": "RollingSum", "params": {"period": 20}, "inputs": ["close"], "out": "sma_rs"},
        {"op": "MapBinary", "params": {"op": 3, "use_scalar_b": True, "scalar_b": 20},
         "inputs": ["sma_rs"], "out": "sma_20"},

        # EMA(20): StateKernel(close, ema, 20)
        {"op": "StateKernel", "params": {"mode": 0, "a": 2.0/21.0, "b": 19.0/21.0},
         "inputs": ["close"], "out": "ema_20"},

        # ATR(14): TrueRange + RollingSum + div
        {"op": "TrueRange", "params": {}, "inputs": ["high", "low", "close"], "out": "tr"},
        {"op": "RollingSum", "params": {"period": 14}, "inputs": ["tr"], "out": "atr_sum"},
        {"op": "MapBinary", "params": {"op": 3, "use_scalar_b": True, "scalar_b": 14},
         "inputs": ["atr_sum"], "out": "atr_14"},

        # DUPLICATE: RollingSum(close, 14) computed TWICE
        {"op": "RollingSum", "params": {"period": 14}, "inputs": ["close"], "out": "rs_a"},
        {"op": "RollingSum", "params": {"period": 14}, "inputs": ["close"], "out": "rs_b"},
        {"op": "MapBinary", "params": {"op": 0}, "inputs": ["rs_a", "rs_b"], "out": "dup_result"},

        # RollingMin/Max -- independent branches off same input
        {"op": "RollingMin", "params": {"period": 20}, "inputs": ["low"], "out": "min_20"},
        {"op": "RollingMax", "params": {"period": 20}, "inputs": ["high"], "out": "max_20"},
    ]

    tasks = _compile(jobs, runtime.kernel_table)
    graph = planner(tasks, runtime.kernel_table)

    print("=" * 80)
    print("AUDIT: ExecutionGraph vs list[Task]")
    print("=" * 80)
    print()

    # ---- Proof 1: Visualization ----
    print("--- Proof 1: DAG Visualization ---")
    print("Task list is flat -- no parent/child visible without scanning all tasks.")
    print("ExecutionGraph has explicit edges via PortRef.")
    print()
    print("DOT graph (graphviz compatible):")
    dot = exec_graph_to_dot(graph)
    print(dot)
    print()

    # ---- Proof 2: Duplicate Detection (CSE) ----
    print("--- Proof 2: Duplicate Detection (CSE) ---")
    print("Find nodes with identical kernel_id + params from same input source")
    from collections import defaultdict

    def node_signature(node):
        """Create a signature for CSE detection."""
        param_items = tuple(sorted((k, v) for k, v in node.params.items()))
        input_keys = []
        for pr in node.inputs:
            if pr.source == "@input":
                input_keys.append(("@input", str(pr.port)))
            else:
                input_keys.append(("node", pr.source, pr.port))
        return (node.kernel_id, param_items, tuple(input_keys))

    sig_groups = defaultdict(list)
    for node in graph.nodes:
        sig = node_signature(node)
        sig_groups[sig].append(node)

    duplicates_found = 0
    for sig, nodes in sig_groups.items():
        if len(nodes) > 1:
            duplicates_found += 1
            ids = [n.id for n in nodes]
            print(f"  DUPLICATE: {sig[0]} with params {dict(sig[1])}")
            print(f"    Nodes: {ids}")
            print(f"    Same inputs: {sig[2]}")
            print(f"    -> Can be computed ONCE, result reused")
            print()

    if duplicates_found == 0:
        print("  No duplicates in this graph")

    print()

    # ---- Proof 3: Parallelism Detection ----
    print("--- Proof 3: Parallelism Detection ---")
    print("Find independent branches -- nodes that can execute in parallel")
    print()

    depends_on = defaultdict(set)
    for node in graph.nodes:
        for pr in node.inputs:
            if isinstance(pr.source, int):
                depends_on[node.id].add(pr.source)

    depth = {}
    def get_depth(nid):
        if nid in depth:
            return depth[nid]
        deps = depends_on.get(nid, set())
        if not deps:
            depth[nid] = 0
        else:
            depth[nid] = 1 + max(get_depth(d) for d in deps)
        return depth[nid]

    for node in graph.nodes:
        get_depth(node.id)

    waves = defaultdict(list)
    for node in graph.nodes:
        waves[depth[node.id]].append(node)

    for wave_idx in sorted(waves.keys()):
        wave_nodes = waves[wave_idx]
        print(f"  Wave {wave_idx}: {len(wave_nodes)} independent nodes")
        for node in wave_nodes:
            deps = depends_on.get(node.id, set())
            dep_str = f" (depends on {deps})" if deps else " (source data only)"
            print(f"    Node {node.id}: {node.kernel_id}{dep_str}")
    print()

    parallel_count = sum(1 for w in waves.values() if len(w) > 1)
    if parallel_count > 0:
        print(f"  -> {parallel_count} waves have parallel nodes")
    print()

    # ---- Summary ----
    print("--- SUMMARY ---")
    print()
    print("What ExecutionGraph gives vs list[Task]:")
    print()
    print("  1. VISIBILITY: graph structure is explicit (edges, not flat list)")
    print(f"     DAG: {len(graph.nodes)} nodes, {len(graph.inputs)} inputs, {len(graph.outputs)} outputs")
    print()
    print("  2. CSE DETECTION: duplicates found by signature matching")
    print(f"     Duplicate groups: {duplicates_found}")
    if duplicates_found > 0:
        print("     -> Ready for CSE optimization (no Planner change needed)")
    else:
        print("     -> (none in this test -- need explicit duplicate to trigger)")
    print()
    print("  3. PARALLELISM: waves detected by dependency depth")
    print(f"     Waves: {len(waves)}")
    for w in sorted(waves.keys()):
        print(f"       Wave {w}: {len(waves[w])} nodes")
    if parallel_count > 0:
        print("     -> Ready for parallel scheduling")
    print()
    print("  4. PORTABILITY: graph is JSON-serializable (no callables)")
    print("     Task has output_size_fn (Callable) -- not serializable")
    print("     ExecutionGraph has only int/float/bool/str -- JSON ready")
    print()

    # Save DOT for visualization
    import tempfile
    dot_path = os.path.join(tempfile.gettempdir(), '_graph_audit.dot')
    with open(dot_path, 'w') as f:
        f.write(dot)
    print(f"DOT saved to {dot_path}")
    print("Visualize: dot -Tsvg _graph_audit.dot > graph.svg")


if __name__ == '__main__':
    audit()
