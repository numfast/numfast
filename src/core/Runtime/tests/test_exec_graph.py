"""Test Execution Graph IR — pure data, no methods."""

import sys, json
sys.path.insert(0, r'C:\App\numfast\numfast\src\core')

from Runtime._lib.exec_graph import (
    ExecutionGraph, Node, PortRef, OutputSpec, DataPort
)
from Runtime._lib.exec_graph_serde import (
    exec_graph_to_dict, exec_graph_from_dict, exec_graph_to_dot
)


def test_pure_data_no_methods():
    """Verify dataclasses have NO methods (only fields)."""
    methods_toxins = ['to_jobs', 'to_dict', 'from_dict', 'to_dot', 
                      'execute', 'compile', 'optimize', 'build']
    for cls in [ExecutionGraph, Node, PortRef, OutputSpec, DataPort]:
        for toxin in methods_toxins:
            assert not hasattr(cls, toxin), f"{cls.__name__} has forbidden method {toxin}"
    print("PASS: no methods on dataclasses")


def test_build_keltner():
    """Build Keltner Channels graph (from spec section 3.1)."""
    graph = ExecutionGraph(
        inputs=[
            DataPort(name="high"),
            DataPort(name="low"),
            DataPort(name="close"),
        ],
        outputs=[
            DataPort(name="kc_middle_20", node_id=3, port=0),
            DataPort(name="kc_upper_20_14", node_id=5, port=0),
            DataPort(name="kc_lower_20_14", node_id=6, port=0),
        ],
        nodes=[
            Node(id=0, kernel_id="TrueRange",
                inputs=[PortRef("@input","high"), PortRef("@input","low"), PortRef("@input","close")],
                outputs=[OutputSpec("TR")], params={}),
            Node(id=1, kernel_id="RollingSum",
                inputs=[PortRef(0, 0)], outputs=[OutputSpec("sumTR")],
                params={"period": 14}),
            Node(id=2, kernel_id="MapBinary",
                inputs=[PortRef(1, 0)], outputs=[OutputSpec("ATR")],
                params={"op": "div", "use_scalar_b": True, "scalar_b": 14}),
            Node(id=3, kernel_id="StateKernel",
                inputs=[PortRef("@input", "close")], outputs=[OutputSpec("EMA")],
                params={"mode": "ema", "period": 20}),
            Node(id=4, kernel_id="MapBinary",
                inputs=[PortRef(2, 0)], outputs=[OutputSpec("ATRx2")],
                params={"op": "mul", "use_scalar_b": True, "scalar_b": 2.0}),
            Node(id=5, kernel_id="MapBinary",
                inputs=[PortRef(3, 0), PortRef(4, 0)], outputs=[OutputSpec("Upper")],
                params={"op": "add"}),
            Node(id=6, kernel_id="MapBinary",
                inputs=[PortRef(3, 0), PortRef(4, 0)], outputs=[OutputSpec("Lower")],
                params={"op": "sub"}),
        ],
        metadata={"version": "1.0", "source": "test"},
    )

    assert len(graph.nodes) == 7
    assert len(graph.inputs) == 3
    assert len(graph.outputs) == 3
    assert graph.nodes[0].kernel_id == "TrueRange"
    assert graph.nodes[1].params["period"] == 14
    print("PASS: Keltner graph built")


def test_serialize_roundtrip():
    """exec_graph_to_dict -> json -> exec_graph_from_dict."""
    graph = ExecutionGraph(
        inputs=[DataPort(name="close")],
        outputs=[DataPort(name="sma_20", node_id=0, port=0)],
        nodes=[
            Node(id=0, kernel_id="RollingSum",
                inputs=[PortRef("@input", "close")],
                outputs=[OutputSpec("sma_20")],
                params={"period": 20}),
        ],
        metadata={"test": True},
    )

    d = exec_graph_to_dict(graph)
    json_str = json.dumps(d, ensure_ascii=False, indent=2)
    d2 = json.loads(json_str)
    graph2 = exec_graph_from_dict(d2)

    assert len(graph2.nodes) == 1
    assert graph2.nodes[0].kernel_id == "RollingSum"
    assert graph2.nodes[0].params["period"] == 20
    assert graph2.inputs[0].name == "close"
    assert graph2.outputs[0].name == "sma_20"
    print("PASS: serialize roundtrip")


def test_to_dot():
    """exec_graph_to_dot produces valid DOT."""
    graph = ExecutionGraph(
        inputs=[DataPort(name="close")],
        outputs=[DataPort(name="sma_20", node_id=0, port=0)],
        nodes=[
            Node(id=0, kernel_id="RollingSum",
                inputs=[PortRef("@input", "close")],
                outputs=[OutputSpec("sma_20")],
                params={"period": 20}),
        ],
    )

    dot = exec_graph_to_dot(graph)
    assert "digraph ExecutionGraph" in dot
    assert "RollingSum" in dot
    assert "sma_20" in dot
    assert "output:sma_20" in dot
    print("PASS: DOT generation")


def test_multi_output():
    """ArgMinMax with 2 outputs."""
    graph = ExecutionGraph(
        inputs=[DataPort(name="close")],
        outputs=[
            DataPort(name="argmin_val", node_id=0, port=0),
            DataPort(name="argmin_idx", node_id=0, port=1),
        ],
        nodes=[
            Node(id=0, kernel_id="ArgMinMax",
                inputs=[PortRef("@input", "close")],
                outputs=[
                    OutputSpec(name="argmin_val"),
                    OutputSpec(name="argmin_idx"),
                ],
                params={"period": 14}),
        ],
    )
    assert len(graph.nodes[0].outputs) == 2
    assert graph.nodes[0].outputs[0].name == "argmin_val"
    assert graph.nodes[0].outputs[1].name == "argmin_idx"
    print("PASS: multi-output")


if __name__ == '__main__':
    test_pure_data_no_methods()
    test_build_keltner()
    test_serialize_roundtrip()
    test_to_dot()
    test_multi_output()
    print()
    print("ALL TESTS PASSED")
