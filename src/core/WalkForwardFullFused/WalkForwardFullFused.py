# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""WalkForwardFullFused Extension -- fused walk-forward evaluation kernel.

Builder entry point. All logic in _lib/.
"""

# Public API -- everything from _lib/
from _lib.walkforward_full_fused import (
    SEED_WGSL,
    WGSL,
    WalkForwardFullFused,
    build_tuples,
    cpu,
    describe,
    describe_seed,
    evaluate,
    bind_kernel,
    last_profile,
    mean_ret,
    register,
)


def setup(kernel):
    """Register kernel in Runtime kernel_table + fill metadata.

    Reuses descriptor.register() -- alias, describe, cpu, wgsl,
    abi_version, capabilities defined in one place (Compute._lib.
    walkforward_full_fused.descriptor).
    """
    bind_kernel(kernel)
    register_fn = kernel.alias.get("register_kernel")
    if register_fn is not None:
        register(_RegisterProxy(register_fn))

    kernel.metadata.setdefault("WalkForwardFullFused", {})
    kernel.metadata["WalkForwardFullFused"]["version"] = "0.1.0"
    kernel.metadata["WalkForwardFullFused"]["kernel"] = "WalkForwardFullFused"
    kernel.metadata["WalkForwardFullFused"]["output"] = "u32[2*ntuples*folds] interleaved sum/count"


class _RegisterProxy:
    """Adapts Runtime register_kernel alias to descriptor.register(runtime)."""

    def __init__(self, fn):
        self._fn = fn

    def register_kernel(self, *args, **kw):
        return self._fn(*args, **kw)
