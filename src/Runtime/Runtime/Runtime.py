# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.runtime import evaluate_impl as _evaluate_impl

_box = {}


def evaluate(graph, backend="auto", n=0, hints=None, profile=None):
    alias = _box["kernel"].alias

    def _chunk_plan(op, nn, backend="cpu", graph=None):
        return alias["chunk_plan"](op, nn, backend=backend, graph=graph)

    res = _evaluate_impl(graph, backend, n, alias["select_backend"],
                         alias["cpu_execute"], alias["gpu_execute"],
                         _chunk_plan, hints, profile)
    _box["kernel"].metadata.setdefault("Runtime", {})["last_execution_info"] = res["execution_info"]
    return res


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("Runtime", {})["version"] = "0.1.0"


PUBLIC = {"evaluate": evaluate}
