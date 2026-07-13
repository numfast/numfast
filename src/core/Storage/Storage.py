"""Storage Extension -- Column, PackingPlan, Table, Optimize.
Builder entry point.
"""


def table(data, columns=None):
    from Storage._lib.table import Table as _Table
    return _Table(data, columns)


def column(name, dtype="float"):
    from Storage._lib.column import Column as _Column
    return _Column(name, dtype)


def compute_plan(columns):
    from Storage._lib.packing import compute_packing_plan as _plan
    return _plan(columns)


def optimize_table(tbl):
    from Storage._lib.optimize import optimize as _opt
    return _opt(tbl)


def column_layout(plan, col_name):
    from Storage._lib.packing import ColumnLayout as _CL
    return _CL(plan, col_name)


def packing_plan(columns):
    from Storage._lib.packing import PackingPlan as _PP
    return _PP(columns)


def Table(data, columns=None):
    from Storage._lib.table import Table as _T
    return _T(data, columns)


def Column(name, dtype="float"):
    from Storage._lib.column import Column as _C
    return _C(name, dtype)


def ColumnType():
    from Storage._lib.column import ColumnType as _CT
    return _CT


def PackingPlan(columns):
    from Storage._lib.packing import PackingPlan as _PP
    return _PP(columns)


def ColumnLayout(plan, col_name):
    from Storage._lib.packing import ColumnLayout as _CL
    return _CL(plan, col_name)


def setup(kernel):
    kernel.metadata.setdefault("Storage", {})
    kernel.metadata["Storage"]["version"] = "1.0.0"
