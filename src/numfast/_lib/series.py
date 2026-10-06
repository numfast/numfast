# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Series: one logical column (values + validity sidecar + Schema record).

Null contract (DELTA-3): missing rows live in `validity` (bool array),
never as sentinels in data. Float NaN on ingest means missing ->
validity False, data filled 0.0; to_numpy() restores NaN at invalid rows.
Int/bool invalid rows hold fill 0/False; read them via .validity.
"""

import numpy as np


_MAP_FNS = {"add": "add", "sub": "sub", "mul": "mul", "div": "div",
            "pow": "pow", "floordiv": "floor_div", "mod": "mod"}
_REDUCE_OPS = ("sum", "count", "mean", "min", "max", "var", "std")
_CMP_OPS = ("==", "!=", "<", "<=", ">", ">=")


def _materialize(buf):
    """Engine buffer -> C-contiguous numpy (BitPack included via __array__)."""
    if isinstance(buf, np.ndarray):
        return np.ascontiguousarray(buf)
    return np.ascontiguousarray(np.asarray(buf))


class Series:
    def __init__(self, kernel, name, values, logical, validity=None, schema=None,
                 sidecar=None, backend=None):
        alias = kernel.alias
        self._kernel = kernel
        self._name = str(name)
        # WHICH BACKEND PRODUCED THIS COLUMN. None means no engine backend ran
        # -- the column was ingested on the host (nf.from_numpy / from_pandas /
        # from_arrow) -- and that is a fact, not a missing value: reporting
        # 'cpu' there would claim a CPU execution that never happened, which
        # is the same lie in the other direction.
        self._backend = backend
        if logical == "text":
            # Dictionary TEXT: int32 codes + sorted-unique sidecar values.
            # Compute on codes would be meaningless; only order/filter flow.
            codes = _materialize(values)
            if codes.dtype != np.dtype("int32"):
                raise ValueError(
                    f"Text Series '{self._name}' needs int32 codes, got {codes.dtype}. "
                    "Fix: build via nf.from_pandas/from_arrow (dictionary path)."
                )
            if not isinstance(sidecar, dict) or not isinstance(
                    sidecar.get("values"), list):
                raise ValueError(
                    f"Text Series '{self._name}' needs sidecar "
                    "{'values': [str, ...]}. Fix: build via the dictionary path."
                )
            self._logical = "text"
            self._values = np.ascontiguousarray(codes)
            self._sidecar = {"values": list(sidecar["values"])}
        else:
            info = alias["canonical_dtype"](logical)  # Schema layer validates dtype
            self._logical = info["logical"]
            self._values = _materialize(values)
            self._sidecar = None
            if self._values.ndim != 1:
                raise ValueError(
                    f"Series '{self._name}' must be rank-1, got shape {self._values.shape}. "
                    "Fix: pass a flat column (2D -> nf.table/from_numpy 2D)."
                )
        if self._values.ndim != 1:
            raise ValueError(
                f"Series '{self._name}' must be rank-1, got shape {self._values.shape}. "
                "Fix: pass a flat column (2D -> nf.table/from_numpy 2D)."
            )
        if validity is None:
            self._validity = None
        else:
            mask = np.ascontiguousarray(np.asarray(validity, dtype=bool))
            if mask.shape != self._values.shape:
                raise ValueError(
                    f"Series '{self._name}' validity shape {mask.shape} != "
                    f"values shape {self._values.shape}. "
                    "Fix: pass validity matching values length."
                )
            self._validity = mask
        if schema is None:
            if self._logical == "text":
                schema = alias["column_schema"](self._name, "int32")
                schema = dict(schema, logical="text")
            else:
                schema = alias["column_schema"](self._name, self._logical)
        self._schema = schema

    # -- accessors ------------------------------------------------------
    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._logical

    @property
    def schema(self):
        return dict(self._schema)

    @property
    def validity(self):
        return None if self._validity is None else self._validity.copy()

    @property
    def backend(self):
        """'cpu' / 'gpu' -- the backend that produced this column.

        None when no engine backend produced it (host ingest via
        nf.from_numpy / from_pandas / from_arrow).
        """
        return self._backend

    def __len__(self):
        return int(self._values.shape[0])

    def __repr__(self):
        n = len(self)
        inv = "" if self._validity is None else f", invalid={int((~self._validity).sum())}"
        # The backend is in the repr, not only in a property: a result held in
        # a REPL has to say where it came from without a second call. Without
        # it, `repr` reported shape and dtype -- everything about the data and
        # nothing about which of the two backends produced it.
        be = "host" if self._backend is None else self._backend
        return f"Series({self._name!r}, {self._logical}[{n}][{be}]{inv})"

    # -- exit -----------------------------------------------------------
    def to_numpy(self):
        """Logical values; float invalid rows restored as NaN (copy).

        Text decodes via Dictionary (display-only boundary restore).
        """
        if self._logical == "text":
            out = self._kernel.alias["dictionary_decode"](
                self._values, self._sidecar["values"], self._validity)
            return np.asarray(out, dtype=object)
        out = self._values.copy()
        if self._validity is not None and out.dtype.kind == "f":
            out[~self._validity] = np.nan
        return out

    def to_masked(self):
        """np.ma.MaskedArray (mask = invalid rows)."""
        mask = (np.zeros(len(self), dtype=bool) if self._validity is None
                else ~self._validity)
        return np.ma.MaskedArray(self._values.copy(), mask=mask)

    def to_pandas(self):
        """Thin wrapper over nf.to_pandas (no duplicate logic)."""
        import numfast as nf
        return nf.to_pandas(self)

    def to_arrow(self):
        """Thin wrapper over nf.to_arrow (no duplicate logic)."""
        import numfast as nf
        return nf.to_arrow(self)

    # -- engine plumbing ------------------------------------------------
    def _source(self, tag):
        kw = {"values": self._values,
              "dtype": "int32" if self._logical == "text" else self._logical}
        if self._validity is not None:
            kw["validity"] = self._validity
        return self._kernel.alias["ir_series"](tag, **kw)

    def _run_array(self, jobs, out):
        a = self._kernel.alias
        graph = a["optimize"](a["compile"](jobs))
        res = a["evaluate"](graph, "cpu", len(self))
        info = res["execution_info"]
        if info["actual"] != "cpu":
            raise RuntimeError(
                f"Series op routed to '{info['actual']}', cpu required. "
                "Fix: use explicit kernel paths for non-cpu backends."
            )
        buf = res["result"]
        values = _materialize(buf)
        sidecar = None
        if self._logical == "text" and values.dtype == np.dtype("int32"):
            logical, sidecar = "text", self._sidecar
        elif values.dtype.kind == "b" or getattr(buf, "width", 0) == 1:
            logical = "bool"
            values = np.ascontiguousarray(values.astype(bool, copy=False))
        elif values.dtype == self._values.dtype:
            logical = self._logical
        else:
            logical = "float64" if values.dtype.kind == "f" else "int32"
        valid = None
        if self._validity is not None or any(
                j.get("params", {}).get("validity") is not None for j in jobs):
            bufs = a["cpu_execute"](graph["nodes"])  # nullable path: sidecar only
            side = bufs.get(out + "#validity")
            if side is not None:
                valid = np.ascontiguousarray(np.asarray(side, dtype=bool))
        return Series(self._kernel, out, values, logical, validity=valid,
                      sidecar=sidecar, backend=info["actual"])

    def _elementwise(self, fn, other):
        if self._logical == "text":
            raise ValueError(
                f"Text Series '{self._name}' has no arithmetic (codes are not values). "
                "Fix: filter/sort/to_numpy only."
            )
        a = self._kernel.alias
        jobs = [self._source("s")]
        if isinstance(other, Series):
            if len(other) != len(self):
                raise ValueError(
                    f"Series length mismatch {len(self)} != {len(other)}. "
                    "Fix: align lengths before elementwise ops."
                )
            jobs.append(other._source("o"))
            jobs.append(a["ir_map"]("m", "s", fn, "o"))
        else:
            jobs.append(a["ir_map"]("m", "s", fn, other))
        return self._run_array(jobs, "m")

    # -- arithmetic -----------------------------------------------------
    def __add__(self, other):
        return self._elementwise("add", other)

    def __sub__(self, other):
        return self._elementwise("sub", other)

    def __mul__(self, other):
        return self._elementwise("mul", other)

    def __truediv__(self, other):
        return self._elementwise("div", other)

    # -- compare -> bool Series ----------------------------------------
    def compare(self, op, other):
        if self._logical == "text":
            raise ValueError(
                f"Text Series '{self._name}' has no compare (dictionaries are "
                "per-column; codes are not cross-comparable). "
                "Fix: filter/sort/to_numpy only."
            )
        if op not in _CMP_OPS:
            raise ValueError(
                f"unknown compare op {op!r}: use one of {list(_CMP_OPS)}."
            )
        a = self._kernel.alias
        jobs = [self._source("s")]
        if isinstance(other, Series):
            jobs.append(other._source("o"))
            jobs.append(a["ir_compare"]("c", "s", "o", op))
        else:
            jobs.append(a["ir_compare"]("c", "s", other, op))
        return self._run_array(jobs, "c")

    def __eq__(self, other):
        return self.compare("==", other)

    def __ne__(self, other):
        return self.compare("!=", other)

    def __lt__(self, other):
        return self.compare("<", other)

    def __le__(self, other):
        return self.compare("<=", other)

    def __gt__(self, other):
        return self.compare(">", other)

    def __ge__(self, other):
        return self.compare(">=", other)

    # -- filter / sort --------------------------------------------------
    def filter(self, mask):
        if not isinstance(mask, Series) or mask.dtype != "bool":
            raise ValueError(
                "filter needs a bool Series mask (e.g. s.compare('>', 0)). "
                "Fix: pass a compare/mask output."
            )
        a = self._kernel.alias
        jobs = [self._source("s"), mask._source("mk"),
                a["ir_filter"]("f", "s", "mk")]
        return self._run_array(jobs, "f")

    def sort(self, descending=False):
        a = self._kernel.alias
        jobs = [self._source("s"), a["ir_sort"]("p", "s", descending=descending),
                a["ir_gather"]("g", "s", "p")]
        return self._run_array(jobs, "g")

    def shift(self, periods, name=None):
        """Positional shift N->N right by periods (v1, CPU).

        Head rows (0:periods) become invalid (validity False, fill 0);
        values + validity travel together (NaN rides as a value).
        periods=0 is identity; negative raises; periods>=N -> all invalid.
        """
        if isinstance(periods, bool) or not isinstance(periods, int):
            raise ValueError(
                f"shift periods must be a non-negative int, got {periods!r}. "
                "Fix: pass periods>=0."
            )
        if periods < 0:
            raise ValueError(
                f"shift negative periods={periods} unsupported in v1. "
                "Fix: pass periods>=0."
            )
        if self._logical == "text":
            raise ValueError(
                f"Text Series '{self._name}' has no shift in v1. "
                "Fix: shift numeric series only."
            )
        a = self._kernel.alias
        jobs = [self._source("s"), a["ir_shift"]("h", "s", periods)]
        graph = a["optimize"](a["compile"](jobs))
        bufs = a["cpu_execute"](graph["nodes"])
        valid = bufs.get("h#validity")
        return Series(self._kernel, name or self._name, bufs["h"],
                      self._logical,
                      validity=None if valid is None else np.ascontiguousarray(
                          np.asarray(valid, dtype=bool)),
                      backend="cpu")

    def cumsum(self, name=None):
        """Inclusive prefix sum N->N (v1, CPU, cumsum only).

        Contract: out[i] = sum(x[0..i]) in the Series dtype
        (int32 wraps mod 2**32, floats keep their dtype); invalid
        input rows contribute 0 and stay invalid in the output
        (per-row validity carry, downstream resumes); NaN is a value
        with IEEE forward propagation; empty -> empty same dtype.
        """
        if self._logical == "text":
            raise ValueError(
                f"Text Series '{self._name}' has no cumsum in v1. "
                "Fix: cumsum numeric series only."
            )
        if self._logical == "bool":
            raise ValueError(
                f"Bool Series '{self._name}' has no cumsum in v1. "
                "Fix: cumsum int32/float32/float64 series only."
            )
        if self._logical not in ("int32", "float32", "float64"):
            raise ValueError(
                f"cumsum needs int32/float32/float64, got {self._logical!r}. "
                "Fix: cumsum numeric series only."
            )
        a = self._kernel.alias
        jobs = [self._source("s"), a["ir_cumsum"]("h", "s")]
        graph = a["optimize"](a["compile"](jobs))
        bufs = a["cpu_execute"](graph["nodes"])
        valid = bufs.get("h#validity")
        return Series(self._kernel, name or self._name, bufs["h"],
                      self._logical,
                      validity=None if valid is None else np.ascontiguousarray(
                          np.asarray(valid, dtype=bool)),
                      backend="cpu")

    def rolling_mean(self, window, min_periods=None, name=None):
        """Rolling mean N->N (v1, CPU, composition only, no new IR).

        Contract (specs/07 + 02): mean = RollingSum + MapBinary(div)
        by the FULL window: out[i] = rolling_sum(x, window, min_periods)[i]
        / window. NaN in window -> NaN; rows with count < min_periods ->
        NaN (default min_periods = window, full windows only); invalid
        input rows act as NaN (validity sidecar honoured by rolling_sum,
        missing rides as NaN in data, no output sidecar); empty -> empty.
        Partial windows (min_periods < window) divide by the full window,
        NOT by the row count (composition-exact per spec, pandas differs
        there: pandas divides partials by count). int input with leading
        NaN (default) yields float64; all-valid int (window=1, or
        min_periods=1 full coverage) follows the map div int rule.
        """
        if self._logical == "text":
            raise ValueError(
                f"Text Series '{self._name}' has no rolling_mean in v1. "
                "Fix: rolling_mean numeric series only."
            )
        if self._logical == "bool":
            raise ValueError(
                f"Bool Series '{self._name}' has no rolling_mean in v1. "
                "Fix: rolling_mean int32/float32/float64 series only."
            )
        if self._logical not in ("int32", "float32", "float64"):
            raise ValueError(
                f"rolling_mean needs int32/float32/float64, got {self._logical!r}. "
                "Fix: rolling_mean numeric series only."
            )
        a = self._kernel.alias
        if min_periods is None:
            rs = a["ir_rolling_sum"]("rs", "s", window)
        else:
            rs = a["ir_rolling_sum"]("rs", "s", window, min_periods=min_periods)
        jobs = [self._source("s"), rs,
                a["ir_map"]("rm", "rs", "div", float(window))]
        graph = a["optimize"](a["compile"](jobs))
        bufs = a["cpu_execute"](graph["nodes"])
        out = _materialize(bufs["rm"])
        if out.dtype.kind == "f":
            logical = "float32" if out.dtype == np.dtype("float32") else "float64"
        elif out.dtype == self._values.dtype:
            logical = self._logical
        else:
            logical = "int32"
        valid = bufs.get("rm#validity")
        return Series(self._kernel, name or self._name, out,
                      logical,
                      validity=None if valid is None else np.ascontiguousarray(
                          np.asarray(valid, dtype=bool)),
                      backend="cpu")

    def returns(self, name=None):
        """Simple returns N->N (v1, CPU, composition only, no new IR).

        Contract (specs/02 + delta-11): out = Shift(1) +
        MapBinary(div, sub): out[0] is invalid (shift head, fill 0);
        out[i] = x[i]/x[i-1] - 1 for i>=1, no x100 factor (vs roc).
        NaN is a value with IEEE propagation; invalid input rows
        propagate via validity AND (head + gaps stay invalid,
        downstream resumes). Zeros in previous inherit MapBinary div
        as-is (float: IEEE +-inf/NaN; int32: deterministic INT_MIN
        cast, sub wraps mod 2**32 -- no new behavior invented);
        empty -> empty same composition dtype.
        """
        if self._logical == "text":
            raise ValueError(
                f"Text Series '{self._name}' has no returns in v1. "
                "Fix: returns numeric series only."
            )
        if self._logical == "bool":
            raise ValueError(
                f"Bool Series '{self._name}' has no returns in v1. "
                "Fix: returns int32/float32/float64 series only."
            )
        if self._logical not in ("int32", "float32", "float64"):
            raise ValueError(
                f"returns needs int32/float32/float64, got {self._logical!r}. "
                "Fix: returns numeric series only."
            )
        a = self._kernel.alias
        jobs = [self._source("s"), a["ir_shift"]("h", "s", 1),
                a["ir_map"]("d", "s", "div", "h"),
                a["ir_map"]("r", "d", "sub", 1)]
        graph = a["optimize"](a["compile"](jobs))
        bufs = a["cpu_execute"](graph["nodes"])
        out = _materialize(bufs["r"])
        if out.dtype.kind == "f":
            logical = "float32" if out.dtype == np.dtype("float32") else "float64"
        elif out.dtype == self._values.dtype:
            logical = self._logical
        else:
            logical = "int32"
        valid = bufs.get("r#validity")
        return Series(self._kernel, name or self._name, out,
                      logical,
                      validity=None if valid is None else np.ascontiguousarray(
                          np.asarray(valid, dtype=bool)),
                      backend="cpu")

    # -- reduce -> scalar ------------------------------------------------
    def reduce(self, op="sum"):
        if self._logical == "text":
            raise ValueError(
                f"Text Series '{self._name}' has no reduce. "
                "Fix: filter/sort/to_numpy only."
            )
        if op not in _REDUCE_OPS:
            raise ValueError(
                f"unknown reduce op {op!r}: use one of {list(_REDUCE_OPS)}."
            )
        a = self._kernel.alias
        jobs = [self._source("s"), a["ir_reduce"]("r", "s", op)]
        graph = a["optimize"](a["compile"](jobs))
        res = a["evaluate"](graph, "cpu", len(self))
        out = res["result"]
        return out.item() if isinstance(out, np.generic) else out

    # -- unique -> {uniq, inv, ng} --------------------------------------
    def unique(self, name=None):
        """Sorted unique + inverse (chains via ir_unique_inverse).

        Returns {"uniq": Series, "inv": Series, "ng": int} with
        uniq[inv] == self (valid rows); invalid rows -> inv -1.
        """
        import numfast as nf
        return nf.unique(self, name=name)
