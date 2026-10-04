# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""plan.py -- the ONLY file in this Extension that knows `kernel.alias`.

Everything else in `_lib/` works through the public vocabulary of
`names.V0` plus the column/IR names handed back by this module. The tables
below are the whole of the facade's knowledge of the engine:

  NODE_ALIAS    public op name  -> IR node constructor in `kernel.alias`
  PREPASS_ALIAS pre-pass helper  -> non-IR helper in `kernel.alias`
  TOOL_ALIAS    planner/executor entry points

Control measure (DESIGN §8.3): a name that is not in `kernel.alias` is a
second IR, so `node()` and `prepass()` REFUSE LOUD instead of degrading.
No new node type, no new op string, no monkey-patched engine behaviour.
"""

import numpy as np

# --- the registries --------------------------------------------------------

NODE_ALIAS = {
    "series": "ir_series",
    "map": "ir_map",
    "compare": "ir_compare",
    "mask": "ir_mask",
    "where": "ir_where",
    "filter": "ir_filter",
    "gather": "ir_gather",
    "sort": "ir_sort",
    "slice": "ir_slice",
    "reduce": "ir_reduce",
    "cumsum": "ir_cumsum",
    "shift": "ir_shift",
    "groupby_multi": "ir_groupby_multi",
    "count_distinct": "ir_count_distinct",
    "text_length": "ir_text_length",
    "text_contains": "ir_text_contains",
    "text_startswith": "ir_text_startswith",
    "text_endswith": "ir_text_endswith",
    "text_equals": "ir_text_equals",
}

PREPASS_ALIAS = {
    "dictionary_encode": "dictionary_encode",
    "dictionary_decode": "dictionary_decode",
    "dict_contains_lut": "dict_contains_lut",
    "dict_equal_lut": "dict_equal_lut",
    "codes_lut_mask": "codes_lut_mask",
    "nfs_stream_open": "nfs_stream_open",
    "nfs_stream_plan": "nfs_stream_plan",
    "nfs_stream_read_block": "nfs_stream_read_block",
}

TOOL_ALIAS = {
    "compile": "compile",
    "optimize": "optimize",
    "explain": "explain",
    "cpu_execute": "cpu_execute",
    "cpu_capability": "cpu_capability",
    # The GPU driver's own capability facts. Present in kernel.alias next to
    # cpu_capability; absent only in a kernel built without Drivers/GPU, which
    # `App.capabilities()` reports VISIBLY as gpu_ops=None (no GPU driver)
    # rather than as an empty GPU op list (the GPU does nothing) -- the two
    # are different facts and only one of them is true.
    "gpu_capability": "gpu_capability",
    "format_error": "format_error",
}

# ir_map knows exactly these seven (nodes.py::_MAP_FNS); `abs`/`neg` are
# GAP-13 and are NOT invented here.
MAP_FN = {
    "add": "add", "sub": "sub", "mul": "mul",
    "truediv": "div", "mod": "mod", "pow": "pow", "floordiv": "floor_div",
}

# ir_reduce / ir_groupby_multi know exactly these (nodes.py).
REDUCE_OPS = ("sum", "count", "mean", "min", "max", "var", "std")
GROUPBY_OPS = ("sum", "count", "mean", "min", "max")

TEXT_OP_NODE = {
    "str_len": "text_length",
    "str_contains": "text_contains",
    "str_startswith": "text_startswith",
    "str_endswith": "text_endswith",
    "str_eq": "text_equals",
}

_NP_LOGICAL = (
    (np.dtype(np.int32), "int32"),
    (np.dtype(np.int64), "int64"),
    (np.dtype(np.float32), "float32"),
    (np.dtype(np.float64), "float64"),
    (np.dtype(np.bool_), "bool"),
)

_box = {}


# --- kernel access ---------------------------------------------------------

def set_kernel(kernel):
    """Bind the kernel handed to setup(kernel) (Join/NfsStream/Planner shape)."""
    _box["kernel"] = kernel


def kernel():
    k = _box.get("kernel")
    if k is None:
        import numfast as nf
        k = _box["kernel"] = nf.get_kernel()
    return k


def alias():
    return kernel().alias


# --- errors ----------------------------------------------------------------

def fail(op, what, fix, doc=""):
    """ValueError in the engine's single error shape (what + how-to-fix)."""
    return alias()["format_error"](f"{op}: {what}", fix=fix, doc=doc)


# --- registries ------------------------------------------------------------

def _resolve(kind, name, registry):
    a = alias()
    target = registry.get(name)
    fn = a.get(target) if target else None
    if fn is None:
        raise fail(
            "tableexpr",
            f"'{name}' resolves to kernel.alias['{target}'], which this kernel "
            f"does not provide. A facade may not invent engine capability.",
            "fix: check the extension is registered in full.toml",
        )
    return fn


def node(op, *args, **kw):
    """Build one IR node dict through the registered constructor."""
    return _resolve("node", op, NODE_ALIAS)(*args, **kw)


# --- node identity (the facade's share of the Planner's CSE key) ------------

# Scalars compare by value; a short scalar SEQUENCE compares by value (that is
# the window in which the Planner's own cheap fingerprint compares one by
# value); anything else -- arrays, columns, long sequences -- compares by
# object identity. `_SEQ_VALUE_MAX` is the Planner's, mirrored deliberately and
# documented as such in `node_identity`; if the Planner ever changes the rule,
# the direction that fails is the SAFE one (see `node_identity`).
_SEQ_VALUE_MAX = 64


def _identity(v):
    """Structural identity of ONE param value (see `node_identity`)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return ("s", v)
    if isinstance(v, dict):
        return ("d", tuple((k, _identity(v[k])) for k in sorted(v)))
    if isinstance(v, (list, tuple)):
        if len(v) <= _SEQ_VALUE_MAX and all(
                x is None or isinstance(x, (bool, int, float, str))
                for x in v):
            return ("seq", tuple(v))
        return ("ref", id(v))
    return ("ref", id(v))


def node_identity(job):
    """`job` -> a hashable identity for "these two nodes compute the same thing".

    WHY THE FACADE NEEDS THIS. The Planner's CSE merges nodes with equal
    (kernel_id, inputs, params-fingerprint) and REWRITES the duplicate's `out`
    name. A facade that emits two such nodes therefore gets a graph whose second
    node has no buffer of its own, and `Chain._buffer` raises. The facade's
    expression memo cannot prevent that: it keys on EXPRESSION structure, which
    is strictly finer than node identity, and it is designed to return ONE node
    for one expression -- so two columns end up SHARING a node, and every op
    that emits one node per column (filter / sort / limit) emits the shared node
    twice. `ir_text_*` is the other door into the same room: it carries its
    decoded `values` as a param and takes no input node, so two different text
    columns with equal values build byte-identical nodes.

    WHY IT IS SAFE. This key is COARSER-or-equal to the Planner's fingerprint on
    every input, which is the only direction that can be wrong safely:

      * scalars: both compare by value -- equal;
      * short scalar sequences: both compare by value -- equal;
      * arrays / columns / long sequences: the Planner compares `id(v)`, this
        compares `id(v)` -- equal;
      * anything else: the Planner falls back to `id(v)` for objects it cannot
        serialise, this compares `id(v)` -- this is COARSER at worst, and it can
        only be coarser for the SAME object, so the nodes really are identical.

    A node the Planner would have merged and this key calls distinct is a
    failure, and it degrades to today's behaviour: both nodes are emitted, the
    Planner merges, `Chain._buffer` raises and names the rewritten node. It
    never produces a wrong VALUE, because a reused node has, by the Planner's
    own definition, identical inputs and params.
    """
    return (job["op"], tuple(job["inputs"]),
            tuple((k, _identity(job["params"][k])) for k in sorted(job["params"])))


def prepass(name):
    """Resolve one out-of-DAG pre-pass helper (dictionary / LUT / stream).

    Curried like `tool()`: the returned callable is the registered helper, so
    call sites read `plan.prepass('codes_lut_mask')(codes, luts, [validity])`.
    """
    return _resolve("prepass", name, PREPASS_ALIAS)


def tool(name):
    return _resolve("tool", name, TOOL_ALIAS)


# --- graph lifecycle -------------------------------------------------------

def graph_of(jobs):
    """jobs[] -> optimized ExecutionGraph (one planner call)."""
    compile_fn = tool("compile")
    return tool("optimize")(compile_fn(jobs))


def buffers_of(jobs):
    """jobs[] -> {buffer name: array} on the CPU oracle."""
    return tool("cpu_execute")(graph_of(jobs)["nodes"])


def buffers_of_graph(graph):
    return tool("cpu_execute")(graph["nodes"])


def report_of(jobs):
    """Human-readable EXPLAIN of jobs[] (Planner artifact, verbatim)."""
    return tool("explain")(graph_of(jobs))


# --- dtype helpers ---------------------------------------------------------

def logical_of(array):
    """numpy buffer -> logical dtype name, or None when unmapped."""
    dt = np.ascontiguousarray(np.asarray(array)).dtype
    for np_dt, logical in _NP_LOGICAL:
        if dt == np_dt:
            return logical
    return None


def as_array(buf):
    return np.ascontiguousarray(np.asarray(buf))


def as_validity(buf):
    if buf is None:
        return None
    return np.ascontiguousarray(np.asarray(buf, dtype=bool))


def null_count(validity):
    """Number of NULL rows in a validity sidecar; 0 when there is none."""
    if validity is None:
        return 0
    return int(np.count_nonzero(~np.ascontiguousarray(
        np.asarray(validity, dtype=bool))))


def decode_text(codes, sidecar, validity=None):
    """Dictionary codes -> [str|None] (display-only boundary restore)."""
    return prepass("dictionary_decode")(as_array(codes), list(sidecar),
                                       None if validity is None
                                       else as_validity(validity))