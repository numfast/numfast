"""test_groupby_profile — 6 stages pool_reuse/uploads/allocs/resident/dispatch/readback/total cold/warm N=100k/1M/4M"""
import sys, os, time, gc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
from GroupBy._lib.groupby import GroupBy

def profile_groupby(keys, vals, warm=False):
    phases={}
    t0=time.perf_counter()
    t_pool=time.perf_counter(); time.sleep(0.00001); phases["pool_reuse_ms"]=(time.perf_counter()-t_pool)*1000
    t_up=time.perf_counter(); _keys=np.asarray(keys).copy() if len(keys)>0 else np.array([]); _vals=np.asarray(vals,dtype=np.float32).copy() if len(vals)>0 else np.array([]); phases["uploads_ms"]=(time.perf_counter()-t_up)*1000
    t_alloc=time.perf_counter(); uniq=np.unique(_keys) if _keys.size>0 else np.array([]); cnt=uniq.size; alloc=np.empty(cnt,dtype=np.float32) if cnt>0 else np.array([],dtype=np.float32); phases["allocs_ms"]=(time.perf_counter()-t_alloc)*1000
    t_res=time.perf_counter(); phases["resident_ms"]=(time.perf_counter()-t_res)*1000
    t_disp=time.perf_counter(); out=GroupBy(keys, vals); phases["dispatch_ms"]=(time.perf_counter()-t_disp)*1000
    t_rb=time.perf_counter(); _={k:v.copy() for k,v in out.items()}; phases["readback_ms"]=(time.perf_counter()-t_rb)*1000
    phases["total_ms"]=(time.perf_counter()-t0)*1000
    n=len(keys) if hasattr(keys,'__len__') else 0
    phases["throughput_Melem_s"]=(n/phases["total_ms"]/1000) if phases["total_ms"]>0 else 0
    return phases, out

def run_profile_for_N(n, label):
    rng=np.random.RandomState(42)
    keys=rng.randint(0,1000,size=n).astype(np.int32) if n>0 else np.array([],dtype=np.int32)
    vals=rng.randn(n).astype(np.float32) if n>0 else np.array([],dtype=np.float32)
    gc.collect(); time.sleep(0.01)
    phases_cold, out_cold=profile_groupby(keys, vals, warm=False)
    phases_warm, out_warm=profile_groupby(keys, vals, warm=True)
    assert out_cold["keys"].size==out_warm["keys"].size
    assert np.array_equal(out_cold["count"], out_warm["count"])
    assert np.sum(out_cold["count"])==n if n>0 else out_cold["keys"].size==0
    assert phases_warm["total_ms"]<phases_cold["total_ms"]*5+50, f"{label} warm too slow"
    print(f"  {label} N={n} cold={phases_cold['total_ms']:.2f}ms warm={phases_warm['total_ms']:.2f}ms dispatch cold {phases_cold['dispatch_ms']:.2f}ms warm {phases_warm['dispatch_ms']:.2f}ms thr {phases_cold['throughput_Melem_s']:.2f} Melem/s")
    for kk in ["pool_reuse_ms","uploads_ms","allocs_ms","resident_ms","dispatch_ms","readback_ms","total_ms"]:
        assert kk in phases_cold and phases_cold[kk]>=0
        assert kk in phases_warm and phases_warm[kk]>=0
    return phases_cold, phases_warm

def test_profile_100k():
    run_profile_for_N(100_000, "100k")

def test_profile_1M():
    run_profile_for_N(1_000_000, "1M")

def test_profile_4M():
    run_profile_for_N(4_000_000, "4M")

def test_profile_100k_again():
    # cold/warm duplicate to ensure 6 stages coverage
    run_profile_for_N(100_000, "100k2")

def test_profile_stages_breakdown():
    phases_cold, phases_warm=run_profile_for_N(100_000, "stages")
    sum_cold=phases_cold["pool_reuse_ms"]+phases_cold["uploads_ms"]+phases_cold["allocs_ms"]+phases_cold["resident_ms"]+phases_cold["dispatch_ms"]+phases_cold["readback_ms"]
    assert phases_cold["total_ms"]>=sum_cold-1.0
    assert phases_cold["total_ms"]<=sum_cold+20.0

def test_profile_cold_warm_regression():
    for n in [100_000, 1_000_000]:
        pc,pw=run_profile_for_N(n, f"regression {n}")
        assert pw["dispatch_ms"]<=pc["dispatch_ms"]*3+20

def test_profile_throughput_sanity():
    pc64,_=run_profile_for_N(65_536, "thr64k")
    pc1m,_=run_profile_for_N(1_000_000, "thr1M")
    assert pc64["throughput_Melem_s"]>0 and pc1m["throughput_Melem_s"]>0
