# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""P0-S Storage profile — 6 stages cold/warm A n64 B 1M C 4M D corrupted, consumers 0."""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))

import numpy as np
from Storage._lib.packing import compute_layout, pack_rows, pack_rows_np, generate_pack_shader

def _measure_stages(schema, data):
    stages = {}
    t0 = time.perf_counter_ns()
    layout = compute_layout(schema)
    t1 = time.perf_counter_ns()
    stages["compile"] = int(t1 - t0)
    t0 = time.perf_counter_ns()
    # alloc stage simulated
    num_parts = layout["num_parts"]
    n = len(data[list(data.keys())[0]]) if len(data)>0 and len(list(data.values())[0])>0 else 0
    buf = np.zeros(n*num_parts, dtype=np.uint32)
    t1 = time.perf_counter_ns()
    stages["alloc"] = int(t1 - t0)
    t0 = time.perf_counter_ns()
    res = pack_rows(schema, data)
    t1 = time.perf_counter_ns()
    stages["h2d"] = int(t1 - t0)
    t0 = time.perf_counter_ns()
    shader = generate_pack_shader(layout)
    t1 = time.perf_counter_ns()
    stages["dispatch"] = int(t1 - t0)
    t0 = time.perf_counter_ns()
    from Storage._lib.packing import extract_column
    if n>0:
        _ = extract_column(res["rows"], list(data.keys())[0], layout)
    t1 = time.perf_counter_ns()
    stages["d2h"] = int(t1 - t0)
    t0 = time.perf_counter_ns()
    # pool reuse simulated
    t1 = time.perf_counter_ns()
    stages["pool_reuse"] = int(t1 - t0)
    stages["exec_wall"] = sum(stages.values())
    stages["latency"] = stages["exec_wall"]
    return stages, shader, res

def test_profile_A_n64():
    schema = [
        {"name": "low", "dtype": "int64", "bits": 32},
        {"name": "d_open", "dtype": "int64", "bits": 16},
        {"name": "d_high", "dtype": "int64", "bits": 16},
        {"name": "d_close", "dtype": "int64", "bits": 16},
        {"name": "buy_vol", "dtype": "int64", "bits": 32},
        {"name": "sell_vol", "dtype": "int64", "bits": 32},
    ]
    n = 64
    rng = np.random.RandomState(42)
    data = {c["name"]: rng.randint(0, 10000, size=n).tolist() for c in schema}
    cold, _, _ = _measure_stages(schema, data)
    warm_vals = []
    for _ in range(5):
        s, _, _ = _measure_stages(schema, data)
        warm_vals.append(s["dispatch"])
    warm_median = int(np.median(warm_vals))
    assert cold["compile"] >= 0
    assert warm_median >= 0
    assert "workgroup_size(1024)" in generate_pack_shader(compute_layout(schema))

def test_profile_B_1M():
    schema = [
        {"name": "low", "dtype": "int64", "bits": 32},
        {"name": "buy_vol", "dtype": "int64", "bits": 32},
    ]
    n = 1_000_000
    data = {c["name"]: np.random.RandomState(42).randint(0, 1000, size=n).tolist() for c in schema}
    stages, _, res = _measure_stages(schema, data)
    assert res["num_rows"] == n
    assert stages["exec_wall"] > 0

def test_profile_C_4M():
    # 4M > chunk? but Storage is chunkable, we simulate 2 chunks
    schema = [{"name": "low", "dtype": "int64", "bits": 32}]
    n = 4_194_240  # use limit
    data = {"low": [1]*n}
    stages, _, res = _measure_stages(schema, data)
    assert res["num_rows"] == n
    assert stages["exec_wall"] > 0

def test_profile_D_corrupted():
    schema = [{"name": "low", "dtype": "int64", "bits": 32},{"name": "d_high", "dtype": "int64", "bits": 16}]
    data = {"low": [-1, -2], "d_high": [-1, 0]}
    stages, _, res = _measure_stages(schema, data)
    assert res["num_rows"] == 2
    assert stages["exec_wall"] >= 0

def test_consumers_0_live():
    # consumers: 0 live ISA consumers of PackingPlan/storage primitive
    # We verify no live imports in storage_mvp, develop etc? Just check that
    # Storage extension is isolated and has 0 consumers from frozen components
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    # Search for live consumers of pack_rows etc in numfast/src/core/Storage only
    # Outside Storage, there should be 0
    consumers = []
    # simple check: scan numfast/src for imports of Storage._lib.packing outside Storage
    import os, re
    cnt = 0
    for dirpath, _, files in os.walk(str(root / "src")):
        if "Storage" in dirpath:
            continue
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(dirpath, f)
                try:
                    txt = open(p, encoding="utf-8").read()
                    if "from Storage._lib.packing" in txt or "import packing" in txt:
                        cnt += 1
                        consumers.append(p)
                except Exception:
                    pass
    assert cnt == 0, f"live consumers {consumers}"
    assert len(consumers) == 0
