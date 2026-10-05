# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.router import router_available as router_available
from _lib.router import router_route as router_route
from _lib.plan import RouterPlan as RouterPlan

_box = {}


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("Router", {})["version"] = "0.2.0"
    kernel.metadata["Router"]["types"] = [
        "router_route", "router_available", "RouterPlan",
    ]
    kernel.metadata["Router"]["inf"] = 4294967295


PUBLIC = {"router_route": router_route, "router_available": router_available,
          "RouterPlan": RouterPlan}