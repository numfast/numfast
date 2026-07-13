"""Planner -- pure function: Task list -> ExecutionGraph.

AST dies here. After planner(), no AST references exist.
ExecutionGraph is the only IR from this point on.
"""

from .exec_graph import ExecutionGraph, Node, PortRef, OutputSpec, DataPort
from .task import Task


def planner(tasks: list[Task], kernel_table: dict) -> ExecutionGraph:
    """Convert compiled Task list into ExecutionGraph.

    This is a dumb, mechanical conversion. No optimization.
    Each Task becomes one Node. Dependencies become PortRef edges.

    Args:
        tasks: list of Task from Compiler.compile()
        kernel_table: runtime kernel registry

    Returns:
        ExecutionGraph -- pure DAG IR
    """
    # Phase 1: collect external inputs (all unique "@input" columns)
    external_inputs: dict[str, str] = {}
    for task in tasks:
        for ref in task.inputs:
            if ref.type == "input":
                external_inputs[ref.column] = "f32"

    inputs_list = [
        DataPort(name=name, dtype=dtype)
        for name, dtype in sorted(external_inputs.items())
    ]

    # Phase 2: build nodes
    nodes: list[Node] = []
    for task in tasks:
        port_refs: list[PortRef] = []
        for ref in task.inputs:
            if ref.type == "input":
                port_refs.append(PortRef(source="@input", port=ref.column, dtype="f32"))
            elif ref.type == "task":
                port_refs.append(PortRef(source=ref.task_id, port=ref.output_idx, dtype="f32"))

        output_specs: list[OutputSpec] = []
        kernel_entry = kernel_table.get(task.op)
        plan = None
        if kernel_entry and kernel_entry.get("describe"):
            plan = kernel_entry["describe"](task.params)

        if plan and len(plan.outputs) > 1:
            for i, slot in enumerate(plan.outputs):
                name = task.out_names[i] if i < len(task.out_names) else slot.template
                output_specs.append(OutputSpec(name=name, dtype=slot.dtype, size_fn=None))
        else:
            for i in range(task.num_outputs):
                name = task.out_names[i] if i < len(task.out_names) else f"out_{i}"
                output_specs.append(OutputSpec(name=name, dtype="f32", size_fn=None))

        nodes.append(Node(
            id=task.id,
            kernel_id=task.op,
            inputs=port_refs,
            outputs=output_specs,
            params=dict(task.params),
            dispatch=task.dispatch,
        ))

    # Phase 3: determine external outputs
    # A task output is "external" if no other task consumes it
    consumed_outputs: set[tuple[int, int]] = set()
    for node in nodes:
        for pr in node.inputs:
            if isinstance(pr.source, int):
                consumed_outputs.add((pr.source, pr.port if isinstance(pr.port, int) else 0))

    outputs_list: list[DataPort] = []
    for node in nodes:
        for i, out_spec in enumerate(node.outputs):
            if (node.id, i) not in consumed_outputs:
                outputs_list.append(DataPort(
                    name=out_spec.name,
                    dtype=out_spec.dtype,
                    node_id=node.id,
                    port=i,
                ))

    # Phase 4: sort nodes topologically
    node_order = _topological_sort(nodes)
    sorted_nodes = sorted(nodes, key=lambda n: node_order.get(n.id, n.id))

    return ExecutionGraph(
        nodes=sorted_nodes,
        inputs=inputs_list,
        outputs=outputs_list,
        metadata={
            "version": "1.0",
            "num_tasks": len(tasks),
            "num_nodes": len(nodes),
            "num_inputs": len(inputs_list),
            "num_outputs": len(outputs_list),
        },
    )


def _topological_sort(nodes: list[Node]) -> dict[int, int]:
    """Assign topological order index to each node.

    Returns dict[node_id] = order_index (lower = earlier in execution).
    """
    children: dict[int, set[int]] = {}
    parent_count: dict[int, int] = {}

    for node in nodes:
        children.setdefault(node.id, set())
        parent_count.setdefault(node.id, 0)

    for node in nodes:
        for pr in node.inputs:
            if isinstance(pr.source, int):
                parent_id = pr.source
                children.setdefault(parent_id, set()).add(node.id)
                parent_count[node.id] = parent_count.get(node.id, 0) + 1

    queue = [n.id for n in nodes if parent_count.get(n.id, 0) == 0]
    order = {}
    idx = 0

    while queue:
        nid = queue.pop(0)
        order[nid] = idx
        idx += 1
        for child_id in children.get(nid, set()):
            parent_count[child_id] -= 1
            if parent_count[child_id] == 0:
                queue.append(child_id)

    for node in nodes:
        if node.id not in order:
            order[node.id] = idx
            idx += 1

    return order
