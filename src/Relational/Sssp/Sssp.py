# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.sssp import sssp_batch as sssp_batch
from _lib.sssp import sssp_csr as sssp_csr

_box = {}


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("Sssp", {})["version"] = "0.1.0"
    kernel.metadata["Sssp"]["types"] = [
        "sssp_csr", "sssp_batch",
    ]
    kernel.metadata["Sssp"]["inf"] = 4294967295


PUBLIC = {"sssp_csr": sssp_csr, "sssp_batch": sssp_batch}
