"""Serialization for Execution Graph IR.

Чистые функции: exec_graph_to_dict / exec_graph_from_dict.
Никаких методов на dataclass'ах.
"""

from .exec_graph import ExecutionGraph, Node, PortRef, OutputSpec, DataPort


def exec_graph_to_dict(graph: ExecutionGraph) -> dict:
    """ExecutionGraph -> JSON-совместимый dict."""
    return {
        "version": "1.0",
        "inputs": [
            {"name": p.name, "dtype": p.dtype}
            for p in graph.inputs
        ],
        "outputs": [
            {
                "name": p.name, "dtype": p.dtype,
                "node_id": p.node_id, "port": p.port,
            }
            for p in graph.outputs
        ],
        "nodes": [
            {
                "id": n.id,
                "kernel_id": n.kernel_id,
                "inputs": [
                    {"source": pr.source, "port": pr.port, "dtype": pr.dtype}
                    for pr in n.inputs
                ],
                "outputs": [
                    {"name": o.name, "dtype": o.dtype, "size_fn": o.size_fn}
                    for o in n.outputs
                ],
                "params": dict(n.params),
                "dispatch": list(n.dispatch) if n.dispatch else None,
            }
            for n in graph.nodes
        ],
        "metadata": dict(graph.metadata),
    }


def exec_graph_from_dict(data: dict) -> ExecutionGraph:
    """dict -> ExecutionGraph."""
    return ExecutionGraph(
        inputs=[DataPort(**p) for p in data["inputs"]],
        outputs=[DataPort(**p) for p in data["outputs"]],
        nodes=[
            Node(
                id=n["id"],
                kernel_id=n["kernel_id"],
                inputs=[PortRef(**pr) for pr in n["inputs"]],
                outputs=[OutputSpec(**o) for o in n["outputs"]],
                params=n.get("params", {}),
                dispatch=tuple(n["dispatch"]) if n.get("dispatch") else None,
            )
            for n in data["nodes"]
        ],
        metadata=data.get("metadata", {}),
    )


def exec_graph_to_dot(graph: ExecutionGraph) -> str:
    """ExecutionGraph -> DOT format (Graphviz).

    Usage:
        dot = exec_graph_to_dot(graph)
        # dot -Tsvg > graph.svg
    """
    lines = ['digraph ExecutionGraph {']
    lines.append('    rankdir=TB;')
    lines.append('    node [shape=box style=filled fillcolor=lightyellow];')

    # Input nodes
    for p in graph.inputs:
        name = f"@input:{p.name}"
        lines.append(f'    "{name}" [shape=box style=filled fillcolor=lightgrey label="input:{p.name}"];')

    # Operation nodes
    for node in graph.nodes:
        label = node.kernel_id
        if node.params:
            param_str = ", ".join(f"{k}={v}" for k, v in node.params.items())
            label += f"\\n({param_str})"
        lines.append(f'    "Node{node.id}" [label="{label}" fillcolor=lightyellow];')

    # Edges: from inputs to nodes
    for node in graph.nodes:
        for pr in node.inputs:
            if pr.source == "@input":
                src = f"@input:{pr.port}"
            else:
                src = f"Node{pr.source}"
            dst = f"Node{node.id}"
            label = str(pr.port) if isinstance(pr.port, int) else pr.port
            lines.append(f'    "{src}" -> "{dst}" [label="{label}"];')

    # Output edges
    for out in graph.outputs:
        if out.node_id is not None:
            src = f"Node{out.node_id}"
            lines.append(f'    "{src}" -> "output:{out.name}" [style=bold label="{out.name}"];')
            lines.append(f'    "output:{out.name}" [shape=box style=filled fillcolor=lightgreen label="out:{out.name}"];')

    lines.append('}')
    return "\n".join(lines)
