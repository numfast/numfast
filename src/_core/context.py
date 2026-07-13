# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import uuid

_ALIGNMENT = 256


def create_context(name: str = "default") -> dict:
    return {"_id": uuid.uuid4().hex, "_name": name}


def context_id(ctx: dict) -> str:
    return ctx["_id"]


def context_name(ctx: dict) -> str:
    return ctx["_name"]


def check_context_compatible(*series: dict) -> None:
    ids = {s["_context_id"] for s in series if s is not None}
    if len(ids) > 1:
        raise TypeError(
            f"Cannot mix Series from different Contexts: {ids}"
        )
