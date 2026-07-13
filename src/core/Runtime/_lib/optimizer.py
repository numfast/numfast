"""Optimizer — набор независимых pass'ов над ExecutionGraph.

Каждый pass — чистая функция: ExecutionGraph -> ExecutionGraph.
Pass'ы не имеют состояния. Композиция через optimize_graph().

Уровни:
  0 — без оптимизаций (identity)
  1 — CSE (Node deduplication)
  2 — CSE + DCE (Dead Code Elimination)
  3 — CSE + DCE + Buffer reuse (coming soon)
  4 — CSE + DCE + Buffer reuse + Fusion (coming soon)
"""

from .exec_graph import ExecutionGraph, Node, PortRef, DataPort, OutputSpec


def _node_signature(node: Node) -> tuple:
    params_key = tuple(sorted(
        (k, float(v) if isinstance(v, (int, float)) else v)
        for k, v in node.params.items()
    ))
    inputs_key = tuple(
        (pr.source, pr.port)
        for pr in node.inputs
    )
    return (node.kernel_id, params_key, inputs_key)


def pass_cse(graph: ExecutionGraph) -> ExecutionGraph:
    """Pass 1: Node CSE — удаление дублирующихся вычислений.

    Чистая функция: не модифицирует исходный граф.
    Если дубликатов нет — возвращает исходный граф (identity).
    """
    seen: dict[tuple, int] = {}
    replacement: dict[int, int] = {}

    for node in graph.nodes:
        sig = _node_signature(node)
        if sig in seen:
            replacement[node.id] = seen[sig]
        else:
            seen[sig] = node.id

    if not replacement:
        return graph

    new_nodes: list[Node] = []
    for node in graph.nodes:
        if node.id in replacement:
            continue
        new_inputs = []
        for pr in node.inputs:
            if isinstance(pr.source, int) and pr.source in replacement:
                new_inputs.append(PortRef(replacement[pr.source], pr.port, pr.dtype))
            else:
                new_inputs.append(PortRef(pr.source, pr.port, pr.dtype))
        new_nodes.append(Node(
            id=node.id,
            kernel_id=node.kernel_id,
            inputs=new_inputs,
            outputs=[OutputSpec(o.name, o.dtype, o.size_fn) for o in node.outputs],
            params=dict(node.params),
            dispatch=node.dispatch,
        ))

    new_outputs = []
    for out in graph.outputs:
        if out.node_id is not None and out.node_id in replacement:
            new_outputs.append(DataPort(out.name, out.dtype, replacement[out.node_id], out.port))
        else:
            new_outputs.append(DataPort(out.name, out.dtype, out.node_id, out.port))

    return ExecutionGraph(
        nodes=new_nodes,
        inputs=list(graph.inputs),
        outputs=new_outputs,
        metadata={**graph.metadata, "cse_removed": len(replacement)},
    )


def pass_dce(graph: ExecutionGraph) -> ExecutionGraph:
    """Pass 2: Dead Code Elimination — удаление неиспользуемых узлов.

    После CSE некоторые узлы могут стать недостижимыми:
    - Их выходы никем не читаются
    - Они не являются внешними выходами графа

    Чистая функция. Если мёртвых узлов нет — identity.

    Алгоритм:
    1. Собрать все node_id, на которые есть ссылки:
       - из inputs других узлов
       - из outputs графа
    2. Узлы, не попавшие в referenced set — мертвы.
    3. Удалить их. Оставшиеся узлы сохранить в топологическом порядке.
    """
    referenced: set[int] = set()

    for out in graph.outputs:
        if out.node_id is not None:
            referenced.add(out.node_id)

    for node in graph.nodes:
        for pr in node.inputs:
            if isinstance(pr.source, int):
                referenced.add(pr.source)

    new_nodes = [n for n in graph.nodes if n.id in referenced]

    removed_count = len(graph.nodes) - len(new_nodes)
    if removed_count == 0:
        return graph

    return ExecutionGraph(
        nodes=new_nodes,
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        metadata={**graph.metadata, "dce_removed": removed_count},
    )


def optimize_graph(graph: ExecutionGraph, level: int = 1) -> ExecutionGraph:
    """Запустить оптимизации графа по уровню.

    Args:
        graph: исходный ExecutionGraph
        level: уровень оптимизации
            0 — без оптимизаций (identity)
            1 — CSE (Node deduplication)
            2 — CSE + DCE (Dead Code Elimination)

    Returns:
        оптимизированный ExecutionGraph
    """
    if level < 1:
        return graph

    g = graph

    # Level 1: CSE
    if level >= 1:
        g = pass_cse(g)

    # Level 2: DCE
    if level >= 2:
        g = pass_dce(g)

    # Level 3+: reserved
    # if level >= 3:
    #     g = pass_buffer_reuse(g)
    # if level >= 4:
    #     g = pass_fusion(g)

    return g
