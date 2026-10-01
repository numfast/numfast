# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.fused import fused_build as fused_build
from _lib.fused import fused_oracle as fused_oracle
from _lib.fused import fused_run as fused_run
from _lib.fused import fused_source as fused_source

_box = {}


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("Fused", {})["version"] = "0.1.0"


PUBLIC = {"fused_build": fused_build, "fused_run": fused_run,
          "fused_source": fused_source, "fused_oracle": fused_oracle}
