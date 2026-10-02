# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""TableExpr -- the GAP-1 consumer facade (Semantic/TableExpr, 07:25).

PUBLIC surface is exactly the four manifest aliases; `_lib/` is PRIVATE
wholesale (07:36). Entry file: imports + PUBLIC + setup(kernel) only.
"""

from _lib.chain import app as tableexpr_app
from _lib.chain import chain as tableexpr_chain
from _lib.expr import ref as tableexpr_expr
from _lib.names import v0_names as tableexpr_names

from _lib import plan


def setup(kernel):
    plan.set_kernel(kernel)
    kernel.metadata.setdefault("TableExpr", {})["version"] = "0.1.0"
    kernel.metadata["TableExpr"]["types"] = ["App", "Chain", "Expr"]
    kernel.metadata["TableExpr"]["surface"] = list(tableexpr_names())


PUBLIC = {
    "tableexpr_app": tableexpr_app,
    "tableexpr_chain": tableexpr_chain,
    "tableexpr_expr": tableexpr_expr,
    "tableexpr_names": tableexpr_names,
}