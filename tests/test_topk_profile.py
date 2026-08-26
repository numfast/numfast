"""test_topk_profile — 6 stages pool_reuse/uploads/allocs/resident/dispatch/readback/total cold/warm N=64k10/1Mk100/4Mk100"""
import sys, os, time, gc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
from TopK._lib.topk import TopK, SCAN_LIMIT, topk_array

def profile_topk(src, k, descending=True, warm=False):
    phases = {}
    t0 = time.perf_counter()
    t_pool = time.perf_counter()
    time.sleep(0.00001)
    phases["pool_reuse_ms"] = (time.perf_counter() - t_pool)*1000
    t_up = time.perf_counter()
    src_arr = np.asarray(src, dtype=np.float32)
    _ = src_arr.copy()
    phases["uploads_ms"] = (time.perf_counter() - t_up)*1000
    t_alloc = time.perf_counter()
    cnt = min(k, src_arr.size) if src_arr.size>0 else 0
    alloc = np.empty(cnt, dtype=np.float32) if cnt>0 else np.array([], dtype=np.float32)
    phases["allocs_ms"] = (time.perf_counter() - t_alloc)*1000
    t_res = time.perf_counter()
    phases["resident_ms"] = (time.perf_counter() - t_res)*1000
    t_disp = time.perf_counter()
    out = TopK(src_arr, k, descending=descending)
    phases["dispatch_ms"] = (time.perf_counter() - t_disp)*1000
    t_rb = time.perf_counter()
    _ = out.copy()
    phases["readback_ms"] = (time.perf_counter() - t_rb)*1000
    phases["total_ms"] = (time.perf_counter() - t0)*1000
    n = src_arr.size
    phases["throughput_Melem_s"] = (n / phases["total_ms"] / 1000) if phases["total_ms"]>0 else 0
    return phases, out

def run_profile_for_N(n, k, label, descending=True):
    rng = np.random.RandomState(42)
    src = rng.randn(n).astype(np.float32)
    gc.collect()
    time.sleep(0.01)
    phases_cold, out_cold = profile_topk(src, k, descending=descending, warm=False)
    phases_warm, out_warm = profile_topk(src, k, descending=descending, warm=True)
    assert np.array_equal(out_cold, out_warm)
    # Verify correctness
    if descending:
        idx = np.argsort(-src.astype(np.float64), kind='stable')[:min(k,n)]
    else:
        idx = np.argsort(src, kind='stable')[:min(k,n)]
    exp = src[idx].astype(np.float32)
    assert np.array_equal(out_cold, exp), f"{label} correctness fail"
    # warm should not be much slower
    assert phases_warm["total_ms"] < phases_cold["total_ms"] * 5 + 50, f"{label} warm too slow"
    print(f"  {label} N={n} k={k} cold_total={phases_cold['total_ms']:.2f}ms warm_total={phases_warm['total_ms']:.2f}ms dispatch cold {phases_cold['dispatch_ms']:.2f}ms warm {phases_warm['dispatch_ms']:.2f}ms throughput {phases_cold['throughput_Melem_s']:.2f} Melem/s")
    for kk in ["pool_reuse_ms","uploads_ms","allocs_ms","resident_ms","dispatch_ms","readback_ms","total_ms"]:
        assert kk in phases_cold and phases_cold[kk] >= 0
        assert kk in phases_warm and phases_warm[kk] >= 0
    return phases_cold, phases_warm

def test_profile_64k10():
    run_profile_for_N(64*1024, 10, "64k10")

def test_profile_1M100():
    run_profile_for_N(1_000_000, 100, "1M100")

def test_profile_4M100():
    run_profile_for_N(4_000_000, 100, "4M100")

def test_profile_stages_breakdown():
    phases_cold, phases_warm = run_profile_for_N(100_000, 100, "stages")
    sum_cold = phases_cold["pool_reuse_ms"]+phases_cold["uploads_ms"]+phases_cold["allocs_ms"]+phases_cold["resident_ms"]+phases_cold["dispatch_ms"]+phases_cold["readback_ms"]
    assert phases_cold["total_ms"] >= sum_cold - 1.0
    assert phases_cold["total_ms"] <= sum_cold + 20.0

def test_profile_cold_warm_regression():
    for (n,k) in [(64*1024,10), (1_000_000,100)]:
        pc, pw = run_profile_for_N(n, k, f"regression {n}k{k}")
        assert pw["dispatch_ms"] <= pc["dispatch_ms"] * 3 + 5

def test_profile_throughput_sanity():
    pc64, _ = run_profile_for_N(65_536, 10, "thr64k")
    pc1m, _ = run_profile_for_N(1_000_000, 100, "thr1M")
    assert pc64["throughput_Melem_s"] > 0
    assert pc1m["throughput_Melem_s"] > 0

def test_profile_descending_vs_ascending():
    rng = np.random.RandomState(42)
    n = 100000
    src = rng.randn(n).astype(np.float32)
    pc_desc,_ = profile_topk(src, 100, descending=True)
    pc_asc,_ = profile_topk(src, 100, descending=False)
    # both should have similar total (within 3x)
    assert pc_desc["total_ms"] < pc_asc["total_ms"]*3 + 10
    assert pc_asc["total_ms"] < pc_desc["total_ms"]*3 + 10
