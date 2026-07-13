"""Test DCE pass — dead node elimination."""

import sys
sys.path.insert(0, r'C:\App\numfast')
sys.path.insert(0, r'C:\App\numfast\numfast\src\core')

from Runtime._lib.exec_graph import ExecutionGraph, Node, PortRef, OutputSpec, DataPort
from Runtime._lib.optimizer import pass_dce


def test_no_dead_nodes():
    """All nodes referenced — no change."""
    graph = ExecutionGraph(
        inputs=[DataPort("close")],
        outputs=[DataPort("sma_20", node_id=0, port=0)],
        nodes=[
            Node(id=0, kernel_id="RollingSum",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("sma_20")],
                 params={"period": 20}),
        ],
    )
    result = pass_dce(graph)
    assert result is graph  # identity
    print("PASS: no_dead_nodes")


def test_dead_node():
    """Node with no consumers AND not an output — removed."""
    graph = ExecutionGraph(
        inputs=[DataPort("close")],
        outputs=[DataPort("ema_20", node_id=0, port=0)],
        nodes=[
            # Alive: is an external output
            Node(id=0, kernel_id="StateKernel",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("ema_20")],
                 params={"mode": 0, "a": 0.095, "b": 0.905}),
            # Dead: no one reads it, not an output
            Node(id=1, kernel_id="RollingSum",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("useless_rs")],
                 params={"period": 14}),
        ],
    )
    result = pass_dce(graph)
    assert len(result.nodes) == 1  # dead node removed
    assert result.nodes[0].kernel_id == "StateKernel"
    assert result.metadata.get("dce_removed") == 1
    print("PASS: dead_node")


def test_node_referenced_by_another():
    """Node consumed by another node — alive."""
    graph = ExecutionGraph(
        inputs=[DataPort("close")],
        outputs=[DataPort("result", node_id=1, port=0)],
        nodes=[
            # Node 0 is consumed by node 1 — alive
            Node(id=0, kernel_id="RollingSum",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("rs")],
                 params={"period": 14}),
            Node(id=1, kernel_id="MapBinary",
                 inputs=[PortRef(0, 0)],  # consumes node 0
                 outputs=[OutputSpec("result")],
                 params={"op": "add", "use_scalar_b": True, "scalar_b": 100}),
        ],
    )
    result = pass_dce(graph)
    assert len(result.nodes) == 2  # both alive
    print("PASS: node_referenced_by_another")


def test_cse_creates_dead_nodes():
    """After CSE, some nodes may become dead."""
    # Two identical RollingSums, one consumed, one not
    graph = ExecutionGraph(
        inputs=[DataPort("close")],
        outputs=[DataPort("rs_used", node_id=0, port=0)],
        nodes=[
            # Consumed by MapBinary
            Node(id=0, kernel_id="RollingSum",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("rs_used")],
                 params={"period": 14}),
            # Also consumed — need to check both
            Node(id=1, kernel_id="RollingSum",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("rs_unused")],
                 params={"period": 14}),
            # MapBinary consumes node 1
            Node(id=2, kernel_id="MapBinary",
                 inputs=[PortRef(1, 0)],
                 outputs=[OutputSpec("dummy")],
                 params={"op": "add", "scalar_b": 0}),
        ],
    )
    # Apply CSE first: node 1 becomes alias of node 0
    from Runtime._lib.optimizer import pass_cse
    g = pass_cse(graph)
    assert len(g.nodes) == 2  # RollingSum merged, MapBinary remains

    # MapBinary output "dummy" is NOT in graph.outputs → DCE removes it
    g2 = pass_dce(g)
    assert len(g2.nodes) == 1  # MapBinary dead (dummy output unreferenced)
    assert g2.nodes[0].kernel_id == "RollingSum"
    assert g2.metadata.get("dce_removed") == 1
    print("PASS: cse_creates_dead_nodes")


if __name__ == '__main__':
    test_no_dead_nodes()
    test_dead_node()
    test_node_referenced_by_another()
    test_cse_creates_dead_nodes()
    print()
    print("ALL DCE TESTS PASSED")
