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