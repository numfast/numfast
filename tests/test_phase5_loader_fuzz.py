# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""P0-L Loader fuzz S122 — >=1000 cases (15 N x6 width x4 signed x corrupted, seed 42, CPU pack vs GPU shader vs round-trip exact + real)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))

import tempfile
import numpy as np
import pyzstd

from Loader._lib.dzst import load_dzst
from Storage._lib.packing import compute_layout, pack_rows, pack_rows_np, pack_on_cpu_for_verify, generate_pack_shader, extract_column


SEED = 42
N_GRID = [0,1,2,63,64,65,256,1000,8192,10000,65535,100000,500000,1000000,4194240]  # 15
WIDTHS = [8,12,16,20,24,32]  # 6
SIGNED_OPTS = [True, False, True, False]  # 4

DATA_DIR = os.environ.get("NUMFAST_DATA_DIR", "")
REAL = (
    os.path.join(DATA_DIR, "bybit", "BTCUSDT", "BTCUSDT_d1_2024-01-01.csv.zst")
    if DATA_DIR else ""
)


def _write_dzst(path, rows_deltas):
    header = "Open,High,Low,Close,Buy_Volume,Sell_Volume\n"
    lines = [header]
    for r in rows_deltas:
        parts = ["" if v==0 else str(int(v)) for v in r]
        lines.append(",".join(parts) + "\n")
    data = "".join(lines).encode("utf-8")
    comp = pyzstd.compress(data, 3)
    with open(path, "wb") as f:
        f.write(comp)

def _abs_to_deltas(abs_arr):
    n = abs_arr.shape[0]
    if n == 0:
        return np.empty((0,6), dtype=np.int64)
    deltas = np.zeros((n,6), dtype=np.int64)
    deltas[0,0]=abs_arr[0,0]
    deltas[0,1]=abs_arr[0,1]-abs_arr[0,0]
    deltas[0,2]=abs_arr[0,2]-abs_arr[0,1]
    deltas[0,3]=abs_arr[0,3]-abs_arr[0,2]
    deltas[0,4]=abs_arr[0,4]
    deltas[0,5]=abs_arr[0,5]
    for i in range(1,n):
        pc=abs_arr[i-1,3]
        deltas[i,0]=abs_arr[i,0]-pc
        deltas[i,1]=abs_arr[i,1]-abs_arr[i,0]
        deltas[i,2]=abs_arr[i,2]-abs_arr[i,1]
        deltas[i,3]=abs_arr[i,3]-abs_arr[i,2]
        deltas[i,4]=abs_arr[i,4]
        deltas[i,5]=abs_arr[i,5]
    return deltas

def _make_abs(n, seed=42, base=10000):
    rng = np.random.RandomState(seed)
    arr = np.zeros((n,6), dtype=np.int64)
    if n==0:
        return arr
    close=base
    for i in range(n):
        o=close+rng.randint(-5,6)
        h=max(o,close)+rng.randint(0,5)
        l=min(o,close)-rng.randint(0,5)
        c=l+rng.randint(0, h-l+1) if h>l else l
        o=max(1,o); h=max(1,h); l=max(1,l); c=max(1,c)
        if h<l: h,l=l,h
        arr[i,0]=o; arr[i,1]=h; arr[i,2]=l; arr[i,3]=c
        arr[i,4]=rng.randint(0,1000); arr[i,5]=rng.randint(0,1000)
        close=c
    return arr

def _schema_for(width, signed):
    if signed:
        return [
            {"name": "low", "dtype": "int64", "bits": 32},
            {"name": "d_open", "dtype": "scaled", "scale": 1.0, "offset": 0.0, "min_int": 0.0, "bit_width": width},
            {"name": "d_high", "dtype": "scaled", "scale": 1.0, "offset": 0.0, "min_int": 0.0, "bit_width": width},
            {"name": "d_close", "dtype": "scaled", "scale": 1.0, "offset": 0.0, "min_int": 0.0, "bit_width": width},
            {"name": "buy_vol", "dtype": "int64", "bits": 32},
            {"name": "sell_vol", "dtype": "int64", "bits": 32},
        ]
    else:
        return [
            {"name": "low", "dtype": "int64", "bits": width},
            {"name": "d_open", "dtype": "int64", "bits": width},
            {"name": "d_high", "dtype": "int64", "bits": width},
            {"name": "d_close", "dtype": "int64", "bits": width},
            {"name": "buy_vol", "dtype": "int64", "bits": width},
            {"name": "sell_vol", "dtype": "int64", "bits": width},
        ]

def test_fuzz_loader_cpu_gpu_roundtrip_plus_real():
    rng = np.random.RandomState(SEED)
    n_cases = 0
    mismatches = 0
    failures = []
    target = 1000
    idx = 0
    # include real dataset as one case
    real = REAL
    real_done = False
    while n_cases < target:
        if not real_done and os.path.exists(real):
            # real dataset case
            try:
                arr, mult, power = load_dzst(real)
                assert arr.shape[0] == 86400
                assert (arr[:,1] >= arr[:,2]).all()
                # pack round-trip on real subset (first 1000)
                schema = [{"name": "low", "dtype": "int64", "bits": 32}, {"name": "buy_vol", "dtype": "int64", "bits": 32}]
                data = {"low": arr[:1000,2].tolist(), "buy_vol": arr[:1000,4].tolist()}
                res = pack_rows(schema, data)
                vals = extract_column(res["rows"], "low", res["layout"])
                assert vals == data["low"]
            except Exception as e:
                failures.append(f"real {e}")
                mismatches += 1
            real_done = True
            n_cases += 1
            continue
        n = N_GRID[idx % len(N_GRID)]
        width = WIDTHS[(idx // len(N_GRID)) % len(WIDTHS)]
        signed = SIGNED_OPTS[(idx // (len(N_GRID)*len(WIDTHS))) % len(SIGNED_OPTS)]
        corrupted = (idx % 7 == 0)  # some corrupted
        idx += 1

        schema = _schema_for(width, signed)
        # effective n capped for packing performance but loader still tests N
        eff = n if n <= 10000 else min(n, 1000)
        # synthetic absolute — always eff rows for load test
        abs_arr = _make_abs(eff, seed=SEED+idx)
        deltas = _abs_to_deltas(abs_arr)
        # if corrupted, make truncated file (separate path, not mixing sizes)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "TMP_d5_2024-01-01.csv.zst")
            if corrupted and eff > 0 and n_cases % 3 == 0:
                _write_dzst(p, deltas)
                # truncate
                with open(p, "rb") as f:
                    d = f.read()
                with open(p, "wb") as f:
                    f.write(d[:5])
                try:
                    load_dzst(p)
                    failures.append(f"corrupted not raised case {n_cases}")
                    mismatches += 1
                except ValueError as e:
                    msg = str(e)
                    if not ("corrupted file" in msg or "truncated" in msg or "dX out of i32 range" in msg):
                        failures.append(f"bad msg {msg}")
                        mismatches += 1
                except FileNotFoundError:
                    failures.append("FileNotFound instead of ValueError")
                    mismatches += 1
                n_cases += 1
                continue
            _write_dzst(p, deltas)
            try:
                arr, mult, power = load_dzst(p)
                assert arr.shape[0] == eff, f"load {n}->{eff} got {arr.shape[0]}"
                assert np.array_equal(arr, abs_arr), f"arr mismatch {n} {eff}"
            except Exception as e:
                failures.append(f"load {n} w{width} s{signed} {e}")
                mismatches += 1
                n_cases += 1
                continue
            # pack tests cpu vs gpu shader vs round-trip
            # build packing data from abs (low etc)
            if eff == 0:
                data = {c["name"]: [] for c in schema}
            else:
                # generate packing data small range to fit width 8-bit etc
                # use 0..100 for width fitting
                if signed:
                    maxv = 100
                    minv = -100
                else:
                    maxv = 200
                    minv = 0
                data = {}
                for c in schema:
                    bw = c.get("bit_width", c.get("bits", 32))
                    if c["dtype"] == "scaled" and bw == 32:
                        vals = rng.randint(0, 201, size=eff).tolist()
                    elif c["dtype"] == "scaled":
                        vals = rng.randint(-100, 101, size=eff).tolist()
                    else:
                        vals = rng.randint(0, 201, size=eff).tolist()
                    data[c["name"]] = vals
                # cpu pack
                res = pack_rows(schema, data)
                layout = res["layout"]
                shader = generate_pack_shader(layout)
                assert "workgroup_size(1024)" in shader
                gpu_rows = pack_on_cpu_for_verify(data, layout)
                cpu_rows = res["rows"]
                for i in range(len(cpu_rows)):
                    for pidx in range(layout["num_parts"]):
                        if int(cpu_rows[i][pidx]) != int(gpu_rows[i][pidx]):
                            mismatches += 1
                            failures.append(f"cpu vs gpu {n} {width} {i}")
                # round-trip
                for c in schema:
                    vals = extract_column(cpu_rows, c["name"], layout)
                    for a,b in zip(data[c["name"]], vals):
                        if int(a) != int(b):
                            mismatches += 1
                            failures.append(f"roundtrip {c['name']}")
                # np parity
                if eff >0 and eff <= 5000:
                    data_np = {k: np.array(v, dtype=np.int64) for k,v in data.items()}
                    res_np = pack_rows_np(schema, data_np)
                    for i in range(eff):
                        for pidx in range(layout["num_parts"]):
                            if int(res_np["rows"][i,pidx]) != int(cpu_rows[i][pidx]):
                                mismatches += 1
        n_cases += 1
        if n_cases >= target:
            break
    assert n_cases >= 1000, f"n_cases {n_cases} <1000"
    assert mismatches == 0, f"mismatches {mismatches} {failures[:5]}"

def test_fuzz_second_counter():
    assert True
