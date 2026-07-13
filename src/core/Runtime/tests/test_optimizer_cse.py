"""Test CSE pass — node deduplication."""

import sys
sys.path.insert(0, r'C:\App\numfast')
sys.path.insert(0, r'C:\App\numfast\numfast\src\core')

from Runtime._lib.exec_graph import ExecutionGraph, Node, PortRef, OutputSpec, DataPort
from Runtime._lib.optimizer import pass_cse as optimizer_cse


def test_no_duplicates():
    """Graph with no duplicates — optimizer returns original."""
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
    result = optimizer_cse(graph)
    assert result is graph  # identity
    assert len(result.nodes) == 1
    print("PASS: no_duplicates")


def test_duplicate_rolling_sum():
    """Two identical RollingSum nodes — second is removed."""
    graph = ExecutionGraph(
        inputs=[DataPort("close")],
        outputs=[
            DataPort("rs_a", node_id=0, port=0),
            DataPort("rs_b", node_id=1, port=0),
        ],
        nodes=[
            Node(id=0, kernel_id="RollingSum",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("rs_a")],
                 params={"period": 14}),
            Node(id=1, kernel_id="RollingSum",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("rs_b")],
                 params={"period": 14}),  # identical!
        ],
    )
    result = optimizer_cse(graph)
    assert len(result.nodes) == 1  # one removed
    assert result.nodes[0].id == 0  # first kept
    assert result.metadata.get("cse_removed") == 1
    # Both outputs point to node 0 now
    for out in result.outputs:
        assert out.node_id == 0
    print("PASS: duplicate_rolling_sum")


def test_duplicate_chain():
    """A depends on duplicate B — after CSE, A depends on canonical."""
    graph = ExecutionGraph(
        inputs=[DataPort("close")],
        outputs=[DataPort("result", node_id=2, port=0)],
        nodes=[
            # Canonical RollingSum
            Node(id=0, kernel_id="RollingSum",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("rs_1")],
                 params={"period": 14}),
            # Duplicate RollingSum
            Node(id=1, kernel_id="RollingSum",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("rs_2")],
                 params={"period": 14}),
            # MapBinary consumes duplicate output
            Node(id=2, kernel_id="MapBinary",
                 inputs=[PortRef(1, 0)],  # was: duplicate's output
                 outputs=[OutputSpec("result")],
                 params={"op": "add", "use_scalar_b": True, "scalar_b": 100}),
        ],
    )
    result = optimizer_cse(graph)
    assert len(result.nodes) == 2  # duplicate removed
    # MapBinary now points to node 0
    map_node = result.nodes[1]  # the MapBinary (id=2)
    assert map_node.inputs[0].source == 0  # now points to canonical
    print("PASS: duplicate_chain")


def test_different_params():
    """Different params = not duplicate."""
    graph = ExecutionGraph(
        inputs=[DataPort("close")],
        outputs=[
            DataPort("rs_14", node_id=0, port=0),
            DataPort("rs_20", node_id=1, port=0),
        ],
        nodes=[
            Node(id=0, kernel_id="RollingSum",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("rs_14")],
                 params={"period": 14}),
            Node(id=1, kernel_id="RollingSum",
                 inputs=[PortRef("@input", "close")],
                 outputs=[OutputSpec("rs_20")],
                 params={"period": 20}),  # different period!
        ],
    )
    result = optimizer_cse(graph)
    assert len(result.nodes) == 2  # no duplicates
    print("PASS: different_params")


if __name__ == '__main__':
    test_no_duplicates()
    test_duplicate_rolling_sum()
    test_duplicate_chain()
    test_different_params()
    print()
    print("ALL CSE TESTS PASSED")
