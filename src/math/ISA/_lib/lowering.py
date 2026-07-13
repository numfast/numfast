"""ISA Lowering: AST dict -> Compute primitives graph.

Two stages in one file:
  1. lower(ast_dict) -> (graph, output_name)
     Walks the AST dict, builds a graph of Compute primitives.
     For calls (SMA, EMA, ATR), delegates to registered resolvers.

  2. format_jobs(graph, output_name) -> list[dict]
     Wraps the graph as Runtime job list.

Adding a new operation:
  @register('RSI')
  def _resolve_rsi(builder, args, kwargs):
      ...

Architecture:
  Expression -> AST -> ISA -> lower() -> Primitive Graph -> Runtime
"""

from typing import List, Tuple, Dict, Any, Callable, Optional


# ============================================================================
# Resolver registry (lowering table: abstract op -> primitive subgraph)
# ============================================================================

_RESOLVERS: Dict[str, Callable] = {}


def register(name: str, resolver_fn: Callable) -> None:
    """Register a resolver for an abstract operation."""
    _RESOLVERS[name] = resolver_fn


def resolve(name: str, builder, args: List[str],
            kwargs: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Resolve an operation to a primitive subgraph."""
    if kwargs is None:
        kwargs = {}
    fn = _RESOLVERS.get(name)
    if fn is None:
        return None
    return fn(builder, args, kwargs)


def registered_ops() -> List[str]:
    """Return sorted list of registered operation names."""
    return sorted(_RESOLVERS.keys())


# ============================================================================
# Graph builder (traverses AST dict, builds Compute primitives graph)
# ============================================================================

class _GraphBuilder:
    """Building a primitive computation graph from AST dict."""

    def __init__(self):
        self._counter = 0
        self._graph: List[dict] = []

    def _fresh(self) -> str:
        name = f"__n{self._counter}"
        self._counter += 1
        return name

    def add_primitive(self, op: str, inputs: List[str], params: Dict[str, Any],
                      out: Optional[str] = None) -> str:
        if out is None:
            out = self._fresh()
        self._graph.append({
            "op": op,
            "inputs": inputs,
            "params": dict(params),
            "out": out,
        })
        return out

    def build(self, ast_dict: dict) -> Tuple[str, List[dict]]:
        self._counter = 0
        self._graph = []
        out = self._compile(ast_dict)
        return out, self._graph

    def _compile(self, node: dict) -> str:
        node_type = node["type"]
        if node_type == "Identifier":
            return node["name"]
        elif node_type == "Literal":
            val = node["value"]
            if isinstance(val, (int, float)):
                return str(float(val))
            return "0.0"
        elif node_type == "BinaryOp":
            return self._compile_binary_op(node)
        elif node_type == "UnaryOp":
            return self._compile_unary_op(node)
        elif node_type == "Call":
            return self._compile_call(node)
        elif node_type == "MemberAccess":
            return node["attr_name"]
        else:
            raise ValueError(f"Unsupported AST node type: {node_type}")

    def _compile_binary_op(self, node: dict) -> str:
        op_type = node["op_type"]

        # Comparison ops -> Compare primitive
        compare_map = {
            ">": "gt", ">=": "ge", "<": "lt",
            "<=": "le", "==": "eq", "!=": "ne",
        }
        if op_type in compare_map:
            return self._compile_compare(node, compare_map[op_type])

        # Logical and:
        #   a_bool = Compare(a, 0.0, ne)  -> uint32 1 if a != 0
        #   b_bool = Compare(b, 0.0, ne)  -> uint32 1 if b != 0
        #   result = LogicalAnd(a_bool, b_bool) -> bitwise &
        if op_type == "and":
            left = self._compile(node["left"])
            right = self._compile(node["right"])
            left_bool = self.add_primitive("Compare", [left], {
                "op": "ne", "use_scalar_b": 1.0, "scalar_b": 0.0,
            })
            right_bool = self.add_primitive("Compare", [right], {
                "op": "ne", "use_scalar_b": 1.0, "scalar_b": 0.0,
            })
            return self.add_primitive("LogicalAnd", [left_bool, right_bool], {})

        # Logical or:
        #   a_bool = Compare(a, 0.0, ne)     -> uint32 1 if a != 0
        #   b_bool = Compare(b, 0.0, ne)     -> uint32 1 if b != 0
        #   result = LogicalOr(a_bool, b_bool)   -> bitwise |
        if op_type == "or":
            left = self._compile(node["left"])
            right = self._compile(node["right"])
            left_bool = self.add_primitive("Compare", [left], {
                "op": "ne", "use_scalar_b": 1.0, "scalar_b": 0.0,
            })
            right_bool = self.add_primitive("Compare", [right], {
                "op": "ne", "use_scalar_b": 1.0, "scalar_b": 0.0,
            })
            return self.add_primitive("LogicalOr", [left_bool, right_bool], {})

        # Arithmetic ops -> MapBinary
        op_map = {"+": 0, "-": 1, "*": 2, "/": 3}
        op_code = op_map.get(op_type)
        if op_code is None:
            raise ValueError(f"Binary op '{op_type}' not supported")

        left = self._compile(node["left"])
        right = self._compile(node["right"])

        params = {"op": op_code}
        inputs = []

        if self._is_scalar(left):
            params["use_scalar_a"] = 1.0
            params["scalar_a"] = float(left)
        else:
            params["use_scalar_a"] = 0.0
            params["scalar_a"] = 0.0
            inputs.append(left)

        if self._is_scalar(right):
            params["use_scalar_b"] = 1.0
            params["scalar_b"] = float(right)
        else:
            params["use_scalar_b"] = 0.0
            params["scalar_b"] = 0.0
            inputs.append(right)

        return self.add_primitive("MapBinary", inputs, params)

    def _compile_unary_op(self, node: dict) -> str:
        operand = self._compile(node["operand"])
        if node["op_type"] == "-":
            return self.add_primitive("Map", [operand], {"func": 6.0})
        if node["op_type"] == "not":
            # not(x) = (x == 0.0)
            return self.add_primitive("Compare", [operand], {
                "op": "eq", "use_scalar_b": 1.0, "scalar_b": 0.0,
            })
        raise ValueError(f"Unsupported unary op: '{node['op_type']}'")

    def _compile_compare(self, node: dict, compare_op: str) -> str:
        """Compile a comparison operation to Compare primitive."""
        left = self._compile(node["left"])
        right = self._compile(node["right"])

        params = {"op": compare_op}
        inputs = []

        if self._is_scalar(left):
            params["use_scalar_a"] = 1.0
            params["scalar_a"] = float(left)
        else:
            params["use_scalar_a"] = 0.0
            params["scalar_a"] = 0.0
            inputs.append(left)

        if self._is_scalar(right):
            params["use_scalar_b"] = 1.0
            params["scalar_b"] = float(right)
        else:
            params["use_scalar_b"] = 0.0
            params["scalar_b"] = 0.0
            inputs.append(right)

        return self.add_primitive("Compare", inputs, params)

    def _compile_call(self, node: dict) -> str:
        name = node["func_name"].upper()
        args = [self._compile(a) for a in node["args"]]
        kwargs = {k: self._compile(v) for k, v in node.get("kwargs", {}).items()}

        result = resolve(name, self, args, kwargs)
        if result is not None:
            return result

        element_map = {
            'ABS': 5, 'NEG': 6, 'SQRT': 3, 'EXP': 2,
            'LOG': 4, 'SIN': 0, 'COS': 1,
        }
        if name in element_map:
            return self.add_primitive("Map", [args[0]], {"func": float(element_map[name])})

        if name == 'MAX':
            return self._map_binary_from_args(args, 4)
        if name == 'MIN':
            return self._map_binary_from_args(args, 5)

        raise ValueError(f"ISA operation '{node['func_name']}' not mapped to any primitive.")

    def _map_binary_from_args(self, args: List[str], op_code: int) -> str:
        a, b = args[0], args[1]
        params = {"op": op_code}
        inputs = []

        if self._is_scalar(a):
            params["use_scalar_a"] = 1.0
            params["scalar_a"] = float(a)
        else:
            params["use_scalar_a"] = 0.0
            inputs.append(a)

        if self._is_scalar(b):
            params["use_scalar_b"] = 1.0
            params["scalar_b"] = float(b)
        else:
            params["use_scalar_b"] = 0.0
            inputs.append(b)

        return self.add_primitive("MapBinary", inputs, params)

    @staticmethod
    def _is_scalar(ref: str) -> bool:
        try:
            float(ref)
            return True
        except ValueError:
            return False


# ============================================================================
# Resolver implementations (abstract operations -> Compute subgraphs)
# ============================================================================

def _resolve_sma(builder, args: List[str], kwargs: Dict[str, str]) -> str:
    """SMA(data, period) -> RollingSum + MapBinary(DIV period)."""
    data = args[0]
    period_str = args[1] if len(args) > 1 else kwargs.get('period', '14.0')
    period = int(float(period_str))

    rs = builder.add_primitive("RollingSum", [data], {"period": period})
    return builder.add_primitive("MapBinary", [rs], {
        "op": 3, "use_scalar_b": 1.0, "scalar_b": float(period),
    })


def _resolve_ema(builder, args: List[str], kwargs: Dict[str, str]) -> str:
    """EMA(data, period) -> StateKernel(mode=0, a, b)."""
    data = args[0]
    period_str = args[1] if len(args) > 1 else kwargs.get('period', '20.0')
    period = int(float(period_str))
    alpha = 2.0 / (period + 1.0)

    return builder.add_primitive("StateKernel_Single", [data], {
        "mode": 0.0, "a": alpha, "b": 1.0 - alpha,
    })


def _resolve_atr(builder, args: List[str], kwargs: Dict[str, str]) -> str:
    """ATR(high, low, close, period) -> TrueRange + RollingSum + MapBinary DIV."""
    high = args[0]
    low = args[1]
    close = args[2]
    period_str = args[3] if len(args) > 3 else kwargs.get('period', '14.0')
    period = int(float(period_str))

    tr = builder.add_primitive("TrueRange", [high, low, close], {})
    rs = builder.add_primitive("RollingSum", [tr], {"period": period})
    return builder.add_primitive("MapBinary", [rs], {
        "op": 3, "use_scalar_b": 1.0, "scalar_b": float(period),
    })


def _resolve_where(builder, args, kwargs):
    """where(condition, a, b) -> Mask(cond, a, b).

    Supports scalar a/b values (literals like 1.0, 0.0).
    cond is always an array.
    """
    cond = args[0]
    a = args[1]
    b = args[2]

    inputs = [cond]
    params = {}

    if builder._is_scalar(a):
        params["use_scalar_a"] = 1.0
        params["scalar_a"] = float(a)
    else:
        params["use_scalar_a"] = 0.0
        params["scalar_a"] = 0.0
        inputs.append(a)

    if builder._is_scalar(b):
        params["use_scalar_b"] = 1.0
        params["scalar_b"] = float(b)
    else:
        params["use_scalar_b"] = 0.0
        params["scalar_b"] = 0.0
        inputs.append(b)

    return builder.add_primitive("Mask", inputs, params)


# Register all operations
register('SMA', _resolve_sma)
register('EMA', _resolve_ema)
register('ATR', _resolve_atr)
register('WHERE', _resolve_where)


# ============================================================================
# Public API
# ============================================================================

def lower(ast_dict: dict) -> Tuple[List[dict], str]:
    """Lower AST dict to Compute primitives graph.

    Args:
        ast_dict: AST from ast_to_dict()

    Returns:
        (graph, output_name)
        graph: list of primitive node dicts
        output_name: output name of the final node
    """
    builder = _GraphBuilder()
    out, graph = builder.build(ast_dict)
    return graph, out


def format_jobs(graph: List[dict], output_name: str,
                resource: str = "@tmp") -> List[dict]:
    """Format primitive graph as Runtime job list.

    Args:
        graph: Primitive graph from lower()
        output_name: Final output name
        resource: Storage resource hint

    Returns:
        List of job dicts ready for Runtime
    """
    jobs = []
    for nd in graph:
        job = {
            "op": nd["op"],
            "inputs": nd["inputs"],
            "params": nd["params"],
            "resource": resource,
            "out": nd["out"],
        }
        jobs.append(job)

    if jobs and jobs[-1]["out"] != output_name:
        jobs[-1]["out"] = output_name

    return jobs
