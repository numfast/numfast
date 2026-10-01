# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Core/errors — single error-format contract (what + how-to-fix + doc-link).

All ValueErrors across Extensions use this shape. Other Extensions receive
it via kernel.alias (depends), never via private imports (import-guard).
"""


def format_error(what, fix="", doc="specs/00-principles.md"):
    """format_error(what, fix, doc) -> ValueError with contract shape."""
    msg = str(what)
    if fix:
        msg += f" Fix: {fix}."
    if doc:
        msg += f" See {doc}"
    return ValueError(msg)
