# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""C LOCAL HASH + RADIX PARTITION: P=8 multiplicative-hash partitions, unique
each part, global sort -> unique. Overflow: part rows > cap recorded as
would-spill with modeled cost (implementation stays bit-exact).
"""

import time

import numpy as np

from ..contract import empty_state, merge_states
from ..kernels import radix_partition_local

NAME = "C_hash_partition"
P_BITS = 3


def local(keys, vals):
    t0 = time.perf_counter()
    k = np.asarray(keys)
    if k.size == 0:
        return empty_state(), {"local_ms": 0.0, "parts": []}
    st, info = radix_partition_local(k, vals, P_BITS)
    ms = (time.perf_counter() - t0) * 1000.0
    n = k.size
    cap = max(4096, 2 * (n // info["p"]))
    spill = sum(max(0, s - cap) for s in info["parts"])
    return st, {
        "local_ms": ms, "parts": info["parts"], "p": info["p"],
        "max_part": int(max(info["parts"])), "cap": int(cap),
        "overflow": bool(spill > 0), "spill_rows": int(spill),
        # modeled extra level cost if spilled partitions were re-partitioned
        "spill_model_ms": float(spill) / max(n, 1) * ms,
        "imbalance": float(max(info["parts"])) / max(min(info["parts"]), 1),
    }


def merge(states):
    return merge_states(states)
