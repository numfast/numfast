# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical IR nodes: jobs[] {op, inputs, params, out} (spec 02)."""

_MAP_FNS = ("add", "sub", "mul", "div", "pow", "floor_div", "mod")
_REDUCE_OPS = ("sum", "count", "mean", "min", "max", "var", "std")
_CMP_OPS = ("==", "!=", "<", "<=", ">", ">=")
_MASK_OPS = ("and", "or", "not")
_GROUPBY_OPS = ("sum", "count", "mean", "min", "max")


def _err(what, fix, doc=""):
    """Single error-format helper (contract: what + how-to-fix + doc-link)."""
    return ValueError(f"{what} Fix: {fix}."
           + (f" See {doc}" if doc else ""))


def _keep_column(values):
    """Zero-copy column keep: array-likes (list/tuple/ndarray/arrow) stored
    by reference, never list()-copied; dict ENC shapes (resident
    dictionary carriers) stored by reference -- list(dict) would silently
    corrupt them to key names; other iterables materialized once."""
    if isinstance(values, (list, tuple, dict)):
        return values
    if getattr(values, "shape", None) is not None:  # numpy / array-api
        return values
    if type(values).__module__.split(".")[0] == "pyarrow":  # pa.Array/Chunked
        return values
    return list(values)


def ir_series(out, values, dtype="int32", validity=None, scale=1, offset=0):
    """Source node: materializes a rank-1 series.

    validity: optional list of 0/1 (DELTA-3 null contract); None = all valid.
    Stored as sidecar, never a sentinel in data.
    values: list (small) or numpy/arrow column (bulk, zero-copy by ref).
    scale/offset (SPEC-DELTA-7): controlled fixed-point contract for float64
    logical values. Default (1, 0) = identity, no contract. Non-default only
    on dtype float64: GPU converts logical -> int32 physical via
    rint((v-offset)/scale) with explicit range guards, aggregates exact
    integer WGSL, untransforms mean/sum on host in f64. scale comes from
    Schema.column_schema (caller queries Schema, never hardcodes). Other
    dtypes with non-default scale/offset are rejected (no silent rescale).
    """
    try:
        sc = float(scale)
        off = float(offset)
    except (TypeError, ValueError):
        raise _err(f"series scale/offset must be finite numbers, got {scale!r}/{offset!r}.",
                   "pass scale=1e-6/offset=0 from column_schema")
    import math as _math
    if not (_math.isfinite(sc) and _math.isfinite(off)) or sc == 0.0:
        raise _err(f"series scale must be finite nonzero, offset finite, got {scale!r}/{offset!r}.",
                   "pass scale=1e-6/offset=0 from column_schema")
    scaled = not (sc == 1.0 and off == 0.0)
    if scaled and dtype != "float64":
        raise _err(f"series scale/offset contract is float64-logical only, got dtype '{dtype}'.",
                   "pass dtype='float64' with scale, or scale=1/offset=0")
    params = {"values": _keep_column(values), "dtype": dtype}
    if scaled:
        params["scale"] = sc
        params["offset"] = off
    if validity is not None:
        params["validity"] = _keep_column(validity)
    return {"op": "series", "inputs": [], "params": params, "out": out}


def ir_map(out, inp, fn="mul", value=1):
    """Elementwise map node (chunkable=true).

    value scalar -> inputs [inp]; value str -> inputs [inp, value] (array-array).
    fn pow: scalar-exp only (spec 01), array exponent rejected at execute.
    fn floor_div/mod: Python floor semantics (np.floor_divide/remainder);
    WGSL `/`/`%` truncate, so negative inputs need a sign-correction prelude
    at GPU port time (non-negative domains map 1:1 today).
    """
    if fn not in _MAP_FNS:
        raise _err(f"unknown map fn '{fn}': use one of {list(_MAP_FNS)}.", "pass fn from add/sub/mul/div/pow/floor_div/mod")
    if isinstance(value, str):
        return {"op": "map", "inputs": [inp, value], "params": {"fn": fn}, "out": out}
    return {"op": "map", "inputs": [inp], "params": {"fn": fn, "value": value}, "out": out}


def ir_compare(out, left, right, op="=="):
    """Compare node -> BoolMask (chunkable=true, spec 08: NaN != NaN via NumPy).

    right scalar -> inputs [left]; right str -> inputs [left, right].
    """
    if op not in _CMP_OPS:
        raise _err(f"unknown compare op '{op}': use one of {list(_CMP_OPS)}.", "pass op from ==/!=/</<=/>/>=")
    if isinstance(right, str):
        return {"op": "compare", "inputs": [left, right], "params": {"op": op}, "out": out}
    return {"op": "compare", "inputs": [left], "params": {"op": op, "value": right}, "out": out}


def ir_filter(out, values, mask):
    """Boolean selection (WHERE): values[mask], order preserved (chunkable=true).

    Universal primitive: SQL WHERE -> column -> compare -> BoolMask -> filter.
    No SQL-specific nodes: predicates are compare+mask composition, selection
    is this node. Empty / all-true / all-false masks are valid (never errors).
    NULL contract (DELTA-3): invalid mask rows (mask validity sidecar) are
    excluded (SQL 3VL: only TRUE keeps the row); output validity = values
    validity rows kept by the effective mask.
    Gather is NOT involved: filter never materializes indices in IR (order
    preserved by construction). Driver-internal compare->scan->gather fusion
    stays inside the driver, never an IR decomposition.
    """
    return {"op": "filter", "inputs": [values, mask], "params": {}, "out": out}


def ir_mask(out, a, b=None, op="and"):
    """Boolean combine for AND/OR/NOT (chunkable=true): BoolMask -> BoolMask.

    op='not': inputs [a]. op='and'/'or': inputs [a, b]. No SQL code in
    Runtime: SQL AND/OR/NOT lower to this node only. Output validity = AND
    of input validities (DELTA-3 sidecar rule, same as map/compare).
    """
    if op not in _MASK_OPS:
        raise _err(f"unknown mask op '{op}': use one of {list(_MASK_OPS)}.", "pass op from and/or/not")
    if op == "not":
        if b is not None:
            raise _err("mask 'not' takes a single input.", "pass ir_mask(out, a, op='not')")
        return {"op": "mask", "inputs": [a], "params": {"op": op}, "out": out}
    if not isinstance(b, str):
        raise _err(f"mask '{op}' needs two masks, got {b!r}.", "pass ir_mask(out, a, b, op='and'/'or')")
    return {"op": "mask", "inputs": [a, b], "params": {"op": op}, "out": out}


def ir_where(out, mask, true, false):
    """Fused row-wise select (CASE): out[i] = true[i] if mask else false[i].

    Fused compare->mask->select (single kernel, chunkable=true, CPU-only
    observable). mask: BoolMask series name (compare/mask output); true /
    false: value series names (equal length, any numeric dtype; promotion
    follows np.where). NULL contract (DELTA-3): invalid mask rows select
    the false branch (SQL 3VL: only TRUE keeps); output validity = chosen
    branch validity per row (no sidecar iff all chosen rows valid).
    Empty / all-true / all-false masks are valid (never errors).
    """
    for name, v in (("mask", mask), ("true", true), ("false", false)):
        if not isinstance(v, str) or not v:
            raise _err(f"where {name} must be a non-empty series name, got {v!r}.",
                       "pass ir_where(out, 'm', 't', 'f')")
    return {"op": "where", "inputs": [mask, true, false],
            "params": {}, "out": out}


def ir_gather(out, values, indices):
    """Positional take: values[indices], output order = indices order.

    Filter (boolean selection, order-preserving, no index materialization)
    is the canonical WHERE primitive and stays a direct node. Gather exists
    for order-changing takes only (ORDER BY/LIMIT/OFFSET/top-k lower to
    sort/rank + this node later). Indices are int32/int64 positions,
    0 <= i < n; out-of-range is an explicit error at execute, never wrap.
    """
    return {"op": "gather", "inputs": [values, indices], "params": {}, "out": out}


def ir_sort(out, *keys, descending=False):
    """Stable order primitive: returns the permutation that sorts rows.

    Universal (no SQL): ORDER BY/LIMIT/OFFSET/top-k lower to sort -> slice ->
    gather. Single key: ir_sort('p', 'v'); composite ORDER BY: ir_sort('p',
    'k1', 'k2') is lexicographic (k1 major) via successive stable passes.
    descending: bool or per-key [bools]. Output: int32 positions (int64 only
    if n exceeds int32 -- never on chunked backends), ascending = argsort
    stable. Rank = this permutation; per-row rank = its inverse (a second
    sort of the permutation -- composition, no second primitive).
    Dictionary TEXT sorts by its precomputed numeric rank codes (caller
    passes codes, never strings). Invalid key rows sort LAST (stable input
    order), always -- documented v0.2 simplification of NULLS FIRST/LAST.
    chunkable=false (global order, spec 04).
    """
    if not keys:
        raise _err("sort needs >=1 key series.", "pass ir_sort(out, 'v') or ir_sort(out, 'k1', 'k2')")
    if isinstance(descending, bool):
        desc = [descending] * len(keys)
    else:
        desc = list(descending)
        if len(desc) != len(keys) or any(not isinstance(d, bool) for d in desc):
            raise _err(f"sort descending must be bool or per-key bools for {len(keys)} keys, "
                       f"got {descending!r}.", "pass descending=True/False or [True, False]")
    return {"op": "sort", "inputs": list(keys), "params": {"descending": desc}, "out": out}


def ir_slice(out, values, limit=None, offset=0):
    """Row-range take: values[offset:offset+limit] (LIMIT/OFFSET without SQL).

    Universal: top-k = sort -> slice -> gather. limit=None reads to the end.
    offset beyond n yields an empty column (SQL semantics), never an error;
    negative limit/offset or non-int types are explicit errors. Keeps dtype
    (BitPack stays packed); validity sidecar sliced alongside.
    chunkable=false in v0.2 (global offsets need full input, spec 04).
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)):
        raise _err(f"slice limit must be a non-negative int or None, got {limit!r}.",
                   "pass limit=10 or limit=None")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise _err(f"slice offset must be a non-negative int, got {offset!r}.",
                   "pass offset=0")
    if (limit is not None and limit < 0) or offset < 0:
        raise _err(f"slice needs limit>=0 and offset>=0, got limit={limit} offset={offset}.",
                   "pass non-negative limit/offset")
    return {"op": "slice", "inputs": [values],
            "params": {"limit": limit, "offset": offset}, "out": out}


def ir_reduce(out, inp, op="sum", skipna=False, ddof=0):
    """Associative reduce node -> scalar (chunkable=true, incremental D2H later).

    skipna=False + NaN -> NaN + warning (spec 08); var/std NumPy-compatible ddof.
    """
    if op not in _REDUCE_OPS:
        raise _err(f"unknown reduce op '{op}': use one of {list(_REDUCE_OPS)}.", "pass op from sum/count/mean/min/max/var/std")
    return {"op": "reduce", "inputs": [inp], "params": {"op": op, "skipna": skipna, "ddof": ddof}, "out": out}


def ir_rolling_sum(out, inp, window, min_periods=None):
    """Rolling sum node (window overlap constraint when chunked, spec 04).

    rolling_mean is NOT a node: compose RollingSum + ir_map div (spec 02).
    min_periods default = window (full windows only, composition exact).
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise _err(f"rolling window must be int >= 1, got {window!r}.", "pass window>=1")
    mp = window if min_periods is None else min_periods
    if not isinstance(mp, int) or isinstance(mp, bool) or not 1 <= mp <= window:
        raise _err(f"rolling min_periods must be int in [1, window], got {min_periods!r}.", "pass 1<=min_periods<=window")
    return {"op": "rolling_sum", "inputs": [inp], "params": {"window": window, "min_periods": mp}, "out": out}


def ir_shift(out, inp, periods):
    """Positional shift N->N (v1, CPU): periods>0 shifts right.

    New head rows (0:periods) are invalid (validity False, data fill 0);
    values + validity travel together (NaN is a value, rides as-is, never
    a validity signal). periods=0 is identity; negative periods are an
    explicit error in v1 (no left-shift); |periods|>=N yields all invalid;
    empty input yields empty output (same dtype). chunkable=false in v1
    (needs global halo, like slice/sort).
    """
    if isinstance(periods, bool) or not isinstance(periods, int):
        raise _err(f"shift periods must be a non-negative int, got {periods!r}.",
                   "pass periods>=0")
    if periods < 0:
        raise _err(f"shift negative periods={periods} unsupported in v1.",
                   "pass periods>=0 (left-shift is explicit error)")
    return {"op": "shift", "inputs": [inp], "params": {"periods": periods}, "out": out}


def ir_cumsum(out, inp):
    """Inclusive prefix sum N->N (v1, CPU-only, cumsum only).

    Narrow scope: only cumsum (no cumprod/cummin/cummax, no generic
    Scan framework, no inclusive/exclusive flag -- always inclusive).
    Spec surface: Compute/Scan `scan(x, op=cumsum, inclusive=True)`
    with `chunkable=false` in v0.2 (specs 04/06, ADR-001 in
    v0.2.1-tail-resolutions.md -- whole-dispatch, prefix dependency).

    Contract (explicit, v1):
    - N->N inclusive: out[i] = sum(x[0..i]); out dtype == in dtype
      (int32/float32/float64 only; bool/enum/text/int64 rejected
      at execute / Series boundary, never silently cast).
    - Validity (DELTA-3 sidecar): per-row carry, like shift. Invalid
      input rows contribute 0 to the running sum; out validity[i] ==
      in validity[i]. Downstream valid rows resume (not poisoned).
      All-valid input (no sidecar) -> no output sidecar. Empty input
      -> empty output (same dtype; empty sidecar iff input had one).
    - NaN is a VALUE (spec 08): rides in data with IEEE propagation
      (a valid NaN poisons its own and all downstream prefix sums);
      NaN never creates validity False. Invalid rows hold the prefix
      value computed with 0-fill (validity False decides NA, and
      Series.to_numpy restores NaN at invalid float rows).
    - Integer overflow: int32 wraps mod 2**32 (two's complement,
      never saturate/trap/raise -- same wrap policy as the GPU scan
      lanes); computed via int64 accumulator then wrapped.
    - Float: computed in the input dtype (f32 lanes stay f32, f64
      stays f64); conformance tolerance from
      specs/conformance-profile.toml vs an f64 oracle.
    """
    return {"op": "cumsum", "inputs": [inp], "params": {}, "out": out}


def ir_groupby(out, values, keys, op="sum", result="dict"):
    """GroupBy node -> {key: scalar} dict, keys sorted (chunkable=false v0.2, spec 04).

    mean is sum+count composition (no monolith). CPU-only observable (spec 02).
    result='dict' (default, frozen shape) or 'carry' (ColumnCarry: ukeys /
    counts / sums columns, mean derived; dict via explicit materialization).
    """
    if op not in _GROUPBY_OPS:
        raise _err(f"unknown groupby op '{op}': use one of {list(_GROUPBY_OPS)}.", "pass op from sum/count/mean")
    if result not in ("dict", "carry"):
        raise _err(f"unknown groupby result '{result}': use one of dict/carry.", "pass result='carry' for columnar")
    params = {"op": op}
    if result != "dict":
        params["result"] = result
    return {"op": "groupby", "inputs": [values, keys], "params": params, "out": out}


def ir_groupby_multi(out, values, keys, ops=("sum", "count"), result="dict"):
    """Fused multi-aggregate (DELTA-1): one grouping traversal, one generic
    state per column (sum,count fused; mean derived, never a second pass).

    Single-column (legacy): values=str, ops=tuple subset of {sum,count,mean};
    result {key: {op: scalar}}, keys sorted.
    Multi-column: values=[col...], ops={col: (ops...)}; result
    {key: {col: {op: scalar}}}, one traversal for ALL columns (Q3/Q4/Q5).
    Composite keys (Q36 4-col): keys is a pack_keys(mode='hash', 4 cols)
    output; CPU executes the generic tuple path (validity AND, lex-sort
    structured unique, sorted tuple ukeys, int64 sums/count), never an
    int64 scalar pack as the sole path.
    result='dict' (default, frozen shape) or 'carry' (ColumnCarry primary,
    dict via explicit materialization). chunkable=false (like groupby).
    """
    if result not in ("dict", "carry"):
        raise _err(f"unknown groupby_multi result '{result}': use one of dict/carry.",
                   "pass result='carry' for columnar")
    if isinstance(values, str):
        ops = tuple(ops)
        if not ops or any(o not in _GROUPBY_OPS for o in ops):
            raise _err(f"groupby_multi ops must be a non-empty subset of {list(_GROUPBY_OPS)}, got {list(ops)}.",
                       "pass ops from sum/count/mean, e.g. ('sum','count','mean')")
        params = {"ops": list(ops)}
        if result != "dict":
            params["result"] = result
        return {"op": "groupby_multi", "inputs": [values, keys],
                "params": params, "out": out}
    cols = list(values)
    if not cols or any(not isinstance(c, str) for c in cols):
        raise _err(f"groupby_multi values must be a series name or non-empty [names], got {values!r}.",
                   "pass 'v' or ['v1','v2']")
    if not isinstance(ops, dict) or set(ops) != set(cols):
        raise _err(f"groupby_multi multi-col ops must be {{col: (ops)}} for {cols}, got {ops!r}.",
                   "pass e.g. {'v1': ('sum',), 'v3': ('mean',)}")
    norm = {}
    for c in cols:
        oc = tuple(ops[c])
        if not oc or any(o not in _GROUPBY_OPS for o in oc):
            raise _err(f"groupby_multi ops for '{c}' must be a non-empty subset of {list(_GROUPBY_OPS)}, got {list(oc)}.",
                       "pass ops from sum/count/mean")
        norm[c] = list(oc)
    params = {"ops": norm, "cols": cols}
    if result != "dict":
        params["result"] = result
    return {"op": "groupby_multi", "inputs": cols + [keys],
            "params": params, "out": out}


def ir_pack_keys(out, *key_series, mode="pack", radix=None):
    """Composite keys (DELTA-2): int32 codes -> single int64/dense key.

    mode='pack' (default): 1-2 inputs; 1 -> widen int64, 2 (hi,lo) ->
    (hi<<32)|(uint32)lo. Zero-copy. 3+ inputs rejected (no scalar-pack
    as sole path for wide composites like Q36 4-col).
    mode='radix': 1+ inputs, mixed-radix composite
    ((c0*M1 + c1)*M2 + c2 ...), radix[i] = cardinality of column i;
    radix=None -> auto (max(code)+1 per column, one vectorized scan each).
    Product overflowing int64 -> explicit error on Python ints, never
    silent wrap; wide overflow (e.g. Q36 4-col) routes to the generic
    tuple path via pack_keys(mode='hash', 4 cols) -> groupby_multi.
    mode='hash': tuple-hash reference + exact tuple sidecar; 4-col
    composite grouping reuses pack_keys(mode='hash', 4 cols) ->
    groupby_multi (validity AND, lex-sort unique, int64 sums/count).
    chunkable=true. Non-int codes rejected at execute (encode first).
    """
    if mode not in ("pack", "radix", "hash"):
        raise _err(f"pack_keys mode must be pack/radix/hash, got {mode!r}.", "pass mode='pack' (or 'radix')")
    if mode == "pack" and not 1 <= len(key_series) <= 2:
        raise _err(f"pack_keys needs 1-2 key series, got {len(key_series)}.", "pass 1-2 int32 code series")
    if mode in ("radix", "hash") and len(key_series) < 1:
        raise _err(f"pack_keys {mode} needs >=1 key series, got 0.", "pass >=1 int32 code series")
    params = {"mode": mode}
    if mode == "radix" and radix is not None:
        r = list(radix)
        if len(r) != len(key_series) or any(not isinstance(m, int) or m < 1 for m in r):
            raise _err(f"pack_keys radix must be positive ints per input, got {radix!r}.",
                       "pass e.g. radix=[100, 100] or radix=None for auto")
        params["radix"] = r
    return {"op": "pack_keys", "inputs": list(key_series), "params": params, "out": out}


def ir_encode_pattern(out, values, prefix):
    """Pattern strings (DELTA-4): generic prefix+int -> int32 codes + validity.

    Non-matching/None -> invalid (None-equivalent), never an error.
    chunkable=true. Downstream uses int codes only.
    """
    if not isinstance(prefix, str) or not prefix:
        raise _err(f"encode_pattern prefix must be a non-empty str, got {prefix!r}.", "pass prefix like 'city#'")
    return {"op": "encode_pattern", "inputs": [],
            "params": {"values": _keep_column(values), "prefix": prefix}, "out": out}


def ir_text_length(out, values):
    """TEXT length -> int32 code-point counts (chunkable=true).

    Semantics (frozen): length = Unicode code points, NOT bytes
    (matches np.char.str_len + Arrow utf8_length; emoji/CJK = 1 per
    char). Non-str/None rows -> invalid (None-equivalent, DELTA-3),
    never an error; empty string -> 0 (valid). Output validity =
    input str-ness sidecar. Non-TEXT numerics never reach here
    (caller encodes first); exotic scalars are an explicit error
    at execute, never a silent coercion.
    """
    return {"op": "text_length", "inputs": [],
            "params": {"values": _keep_column(values)}, "out": out}


def ir_text_contains(out, values, substr):
    """TEXT contains -> BoolMask (chunkable=true).

    Semantics (frozen): UTF-8 substring search, case-sensitive
    (Python `in` / np.char.find parity). Empty substr matches every
    valid row. Invalid (None/non-str) rows never match (3VL: mask
    &= validity), never an error. substr must be str (else explicit
    error at execute).
    """
    if not isinstance(substr, str):
        raise _err(f"text_contains substr must be str, got {type(substr).__name__}.",
                   "pass a str literal")
    return {"op": "text_contains", "inputs": [],
            "params": {"values": _keep_column(values), "substr": substr}, "out": out}


def ir_text_startswith(out, values, prefix):
    """TEXT startswith -> BoolMask (chunkable=true).

    Semantics (frozen): anchored prefix match, case-sensitive
    (`str.startswith` / `np.char.startswith` parity). Empty prefix
    matches every valid row. Invalid rows never match (3VL).
    """
    if not isinstance(prefix, str):
        raise _err(f"text_startswith prefix must be str, got {type(prefix).__name__}.",
                   "pass a str literal")
    return {"op": "text_startswith", "inputs": [],
            "params": {"values": _keep_column(values), "prefix": prefix}, "out": out}


def ir_text_endswith(out, values, suffix):
    """TEXT endswith -> BoolMask (chunkable=true).

    Semantics (frozen): anchored suffix match, case-sensitive
    (`str.endswith` / `np.char.endswith` parity). Empty suffix
    matches every valid row. Invalid rows never match (3VL).
    """
    if not isinstance(suffix, str):
        raise _err(f"text_endswith suffix must be str, got {type(suffix).__name__}.",
                   "pass a str literal")
    return {"op": "text_endswith", "inputs": [],
            "params": {"values": _keep_column(values), "suffix": suffix}, "out": out}


def ir_text_equals(out, values, key):
    """TEXT equals -> BoolMask (chunkable=true).

    Semantics (frozen): full-row equality, case-sensitive
    (`str ==` parity). Empty key matches only empty valid rows.
    Invalid rows never match (3VL). `!=` is composition:
    ir_mask(not, ir_text_equals(...)).
    """
    if not isinstance(key, str):
        raise _err(f"text_equals key must be str, got {type(key).__name__}.",
                   "pass a str literal")
    return {"op": "text_equals", "inputs": [],
            "params": {"values": _keep_column(values), "key": key}, "out": out}


def ir_text_regex_replace(out, values, pattern, repl):
    """TEXT regexp_replace -> TEXT (chunkable=true, CPU-only v1).

    Semantics (frozen): DuckDB regexp_replace parity (first match
    only, `re.sub(count=1)`): replacement of the leftmost
    non-overlapping match, backrefs `\\1` honored, empty pattern
    matches at position 0, case-sensitive. Invalid (None/non-str)
    rows -> invalid (None-equivalent, DELTA-3), never a match,
    never an error. pattern/repl must be str literals (else
    explicit error at IR build); bad pattern is an explicit error
    at execute, never silent NULL. No hardcoded patterns: caller
    passes any pattern (Q29 passes its own ClickBench pattern).
    """
    if not isinstance(pattern, str):
        raise _err(f"text_regex_replace pattern must be str, got {type(pattern).__name__}.",
                   "pass a str literal")
    if not isinstance(repl, str):
        raise _err(f"text_regex_replace repl must be str, got {type(repl).__name__}.",
                   "pass a str literal")
    return {"op": "text_regex_replace", "inputs": [],
            "params": {"values": _keep_column(values),
                       "pattern": pattern, "repl": repl}, "out": out}


_RNG_MODES = ("bits",)


def _rng_domain(seed, stream, offset, what):
    """Shared (seed, stream, offset) counter-domain validation (RNG CORE).

    seed: uint64 int (bool rejected); stream/offset: non-negative ints.
    """
    for name, v, bits in (("seed", seed, 64), ("stream", stream, 64),
                          ("offset", offset, 64)):
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2 ** bits:
            raise _err(f"{what} {name} must be an int in [0, 2**{bits}), got {v!r}.",
                       f"pass {name}=0 (or a uint{bits} int)")
    return {"seed": seed, "stream": stream, "offset": offset}


def ir_rng_fill_i32(out, n, seed, stream=0, offset=0, lo=0, hi=100, mode="bits"):
    """Counter-based RNG fill -> int32 lanes in [lo, hi) (chunkable=true).

    CORE (Philox4x32-10): lane i uses counter offset + i, so chunked
    (offset-adjusted) == unchunked bit-exact. width 1 -> constant lo
    (counter still advances). mode='bits' only (Lemire r*width>>32 map).
    n: non-negative int. Errors surface at execute as -2 range / -3 geometry.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise _err(f"rng_fill_i32 n must be a non-negative int, got {n!r}.", "pass n>=0")
    if isinstance(lo, bool) or not isinstance(lo, int) or isinstance(hi, bool) \
            or not isinstance(hi, int):
        raise _err(f"rng_fill_i32 lo/hi must be ints, got {lo!r}/{hi!r}.", "pass lo=0 hi=100")
    if mode not in _RNG_MODES:
        raise _err(f"rng_fill_i32 mode must be one of {list(_RNG_MODES)}, got {mode!r}.",
                   "pass mode='bits'")
    params = {"n": n, "lo": lo, "hi": hi, "mode": mode}
    params.update(_rng_domain(seed, stream, offset, "rng_fill_i32"))
    return {"op": "rng_fill_i32", "inputs": [], "params": params, "out": out}


def ir_rng_fill_f64(out, n, seed, stream=0, offset=0, lo=0.0, hi=1.0):
    """Counter-based RNG fill -> float64 lanes in [lo, hi) (chunkable=true).

    CORE: 53-bit mantissa draws (u/2^53 in [0,1), span-scaled). Same
    counter domain as rng_fill_i32. lo/hi must be finite with lo < hi
    (else -2 at execute). n: non-negative int.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise _err(f"rng_fill_f64 n must be a non-negative int, got {n!r}.", "pass n>=0")
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        raise _err(f"rng_fill_f64 lo/hi must be numbers, got {lo!r}/{hi!r}.",
                   "pass lo=0.0 hi=1.0")
    params = {"n": n, "lo": lo_f, "hi": hi_f}
    params.update(_rng_domain(seed, stream, offset, "rng_fill_f64"))
    return {"op": "rng_fill_f64", "inputs": [], "params": params, "out": out}


def ir_rng_sample_no_replace(out, n, k, seed, stream=0, offset=0):
    """Fisher-Yates first-k sample -> int32 lanes (chunkable=false).

    CORE: partial Yates over 0..n-1, output in DRAW ORDER (not sorted).
    Draws are sequential from the counter domain (offset + draw#), so the
    draw stream is NOT row-chunkable (chunkable=false). k > n is an
    explicit error at execute (-2), never wrap.
    """
    for name, v in (("n", n), ("k", k)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise _err(f"rng_sample_no_replace {name} must be a non-negative int, got {v!r}.",
                       "pass non-negative n/k with k<=n")
    params = {"n": n, "k": k}
    params.update(_rng_domain(seed, stream, offset, "rng_sample_no_replace"))
    return {"op": "rng_sample_no_replace", "inputs": [], "params": params, "out": out}


def ir_rng_permutation(out, n, seed, stream=0, offset=0):
    """Full Yates permutation of 0..n -> int32 lanes (chunkable=false).

    CORE: same draw stream as rng_sample_no_replace with k == n.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise _err(f"rng_permutation n must be a non-negative int, got {n!r}.", "pass n>=0")
    params = {"n": n}
    params.update(_rng_domain(seed, stream, offset, "rng_permutation"))
    return {"op": "rng_permutation", "inputs": [], "params": params, "out": out}


def ir_rng_compat(out, n, seed, kind="runif", lo=0.0, hi=1.0, m=None):
    """R-compatible RNG (R-COMPAT MT19937, CPU-only, chunkable=false).

    kind='runif': float64 lanes, out[i] = lo + (hi-lo)*unif(), strict
    sequential (Ripley init + twist/temper + fixup). kind='sample':
    int32 index draws (0-based lanes), rejection chunks, m draws
    (m required). seed: R int32 range (wraps like R's seed coercion).
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise _err(f"rng_compat n must be a non-negative int, got {n!r}.", "pass n>=0")
    if kind not in ("runif", "sample"):
        raise _err(f"rng_compat kind must be runif/sample, got {kind!r}.", "pass kind='runif'")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise _err(f"rng_compat seed must be an int, got {seed!r}.", "pass seed=42")
    params = {"n": n, "seed": seed, "kind": kind}
    if kind == "runif":
        try:
            lo_f, hi_f = float(lo), float(hi)
        except (TypeError, ValueError):
            raise _err(f"rng_compat lo/hi must be numbers, got {lo!r}/{hi!r}.",
                       "pass lo=0.0 hi=1.0")
        params.update({"lo": lo_f, "hi": hi_f})
    else:
        if isinstance(m, bool) or not isinstance(m, int) or m < 0:
            raise _err(f"rng_compat sample needs m>=0 draws, got {m!r}.", "pass m=<draw count>")
        params["m"] = m
    return {"op": "rng_compat", "inputs": [], "params": params, "out": out}


def ir_map_round(out, values, ndigits=0):
    """Half-even round (banker's) -> float64 + validity (chunkable=true).

    ndigits 0..15 (else explicit error). NaN -> NaN, +-Inf -> +-Inf
    (validity kept); invalid input rows -> 0.0 + valid 0; -0.0 signbit
    preserved; scaled overflow -> invalid. Scaled-int ticks comparison.
    """
    if isinstance(ndigits, bool) or not isinstance(ndigits, int):
        raise _err(f"map_round ndigits must be an int, got {ndigits!r}.", "pass ndigits=0..15")
    if not 0 <= ndigits <= 15:
        raise _err(f"map_round ndigits must be in [0, 15], got {ndigits!r}.", "pass ndigits=0..15")
    return {"op": "map_round", "inputs": [values],
            "params": {"ndigits": ndigits}, "out": out}


def ir_lookup(out, build_keys, probe_keys):
    """Build-probe lookup -> (positions, hit-mask) pair (chunkable=false).

    Semantics: build side MUST hold unique keys (unique-keys contract);
    duplicate build keys are an explicit error at execute (never collapse:
    collapse is the GPU semi-lookup set semantics in Drivers/GPU lookup_mask
    / LookupTable, which is mask-only and NOT this node). Probe keys may
    repeat freely (each probe row independent). Output: positions int64
    (index into the SORTED-UNIQUE build order, -1 on miss) + sidecar
    `<out>#hit` bool mask (True = match). Payload take is composition:
    gather(payload, positions[hit]) -- lookup never carries payloads.
    Keys are int32 (like Join); invalid rows never match (miss, not error).
    chunkable=false like sort/groupby: the probe needs the GLOBAL build
    table U[K] (per-chunk builds would miss cross-chunk keys), so chunk
    planning keeps the graph whole and the driver shards the probe only.
    """
    for name, v in (("build_keys", build_keys), ("probe_keys", probe_keys)):
        if not isinstance(v, str) or not v:
            raise _err(f"lookup {name} must be a non-empty series name, got {v!r}.",
                       "pass ir_lookup(out, 'build', 'probe')")
    if build_keys == probe_keys:
        raise _err("lookup build_keys and probe_keys must differ.",
                   "pass two distinct series (build side is the unique table)")
    return {"op": "lookup", "inputs": [build_keys, probe_keys],
            "params": {}, "out": out}


def ir_unique_inverse(out, values):
    """Sorted-order unique + inverse -> (uniq, inv) pair (chunkable=false).

    uniq: sorted ascending distinct values; inv: sorted-position codes
    with uniq[inv] == values (np.unique semantics, NOT first-appearance).
    Invalid input rows -> inv -1 (excluded from uniq); ng (distinct
    count) rides buffer metadata `<out>#ng`. Global order => chunkable=false.
    """
    return {"op": "unique_inverse", "inputs": [values], "params": {}, "out": out}


def ir_unique(out, values):
    """Sorted distinct set without inverse (chunkable=false).

    Set-only path (uniq sorted ascending + `#ng`; invalid rows excluded
    from uniq). Callers needing only the set/count read `bufs[out]`
    (+ `#ng`); callers needing position codes use `ir_unique_inverse`
    explicitly. `CPU` executes op `unique` without the N-sized inverse
    build (generic: all scalar DISTINCT save ~half the sort cost).
    """
    return {"op": "unique", "inputs": [values], "params": {}, "out": out}


def ir_count_distinct(out, values, keys=None):
    """Thin alias for DISTINCT counts, no new engine (chunkable=false).

    keys=None (global): same node as unique (uniq+#ng, no inverse);
    the scalar count is `<out>#ng` (== uniq.size, invalid-excluded).
    keys given (grouped): `group_count_distinct` node with
    inputs [values, keys] executed by the CPU sorted_dedup kernel
    (_valid_mask -> compact -> sort(keys,values) -> single vector
    scan; NULL/invalid-excluded). Co-aggregates stay separate nodes.
    """
    if keys is None:
        return ir_unique(out, values)
    return {"op": "group_count_distinct", "inputs": [values, keys],
            "params": {}, "out": out}
