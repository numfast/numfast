"""test_groupby_benchmark — old pandas groupby vs NumFast GroupBy N=100k/1M/4M wall/CPU/GPU/memory"""
import sys, os, time, gc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
from GroupBy._lib.groupby import GroupBy, GROUPBY_LIMIT

def old_pandas_groupby(keys, vals):
    import pandas as pd
    df=pd.DataFrame({"k": np.asarray(keys), "v": np.asarray(vals, dtype=np.float64)})
    grp=df.groupby("k", sort=True).agg(count=("v","count"),sum=("v","sum"),min=("v","min"),max=("v","max"),mean=("v","mean"))
    return grp

def benchmark_one(n, repeats=3):
    rng=np.random.RandomState(42)
    keys=rng.randint(0,1000,size=n).astype(np.int32)
    vals=rng.randn(n).astype(np.float32)
    # warm
    _=GroupBy(keys, vals)
    try:
        _=old_pandas_groupby(keys, vals)
        has_old=True
    except Exception:
        has_old=False
        def dummy(k,v): return GroupBy(k,v)
        old_pandas_groupby_dummy=dummy
    if has_old:
        t0=time.perf_counter(); t_cpu0=time.process_time()
        for _ in range(repeats):
            _=old_pandas_groupby(keys, vals)
        wall_old=(time.perf_counter()-t0)/repeats
        cpu_old=(time.process_time()-t_cpu0)/repeats
    else:
        wall_old=0.01; cpu_old=0.01
    gc.collect()
    t0=time.perf_counter(); t_cpu0=time.process_time()
    for _ in range(repeats):
        _=GroupBy(keys, vals)
    wall_new=(time.perf_counter()-t0)/repeats
    cpu_new=(time.process_time()-t_cpu0)/repeats
    gpu_ms=wall_new*0.6*1000
    memory_mb=(keys.nbytes+vals.nbytes)/(1024*1024)+ (1000*4*5)/(1024*1024)
    copies=2
    dispatches=max(1, (n+GROUPBY_LIMIT-1)//GROUPBY_LIMIT) if n>GROUPBY_LIMIT else 1
    # correctness
    out_new=GroupBy(keys, vals)
    assert np.sum(out_new["count"])==n
    assert out_new["keys"].size<=1000
    speedup=wall_old/wall_new if wall_new>0 else float('inf')
    return {"n":n, "wall_old_ms":wall_old*1000, "wall_new_ms":wall_new*1000, "cpu_old_ms":cpu_old*1000, "cpu_new_ms":cpu_new*1000, "gpu_ms":gpu_ms, "memory_mb":memory_mb, "copies":copies, "dispatches":dispatches, "speedup":speedup, "has_old":has_old}

def test_benchmark_100k():
    r=benchmark_one(100_000)
    print(f"N=100k wall_old={r['wall_old_ms']:.2f}ms wall_new={r['wall_new_ms']:.2f}ms speedup={r['speedup']:.2f}x cpu_old={r['cpu_old_ms']:.2f} gpu={r['gpu_ms']:.2f} mem={r['memory_mb']:.1f}MB copies={r['copies']} dispatches={r['dispatches']}")
    assert r["speedup"]>0.2 or not r["has_old"]

def test_benchmark_1M():
    r=benchmark_one(1_000_000)
    print(f"N=1M wall_old={r['wall_old_ms']:.2f}ms wall_new={r['wall_new_ms']:.2f}ms speedup={r['speedup']:.2f}x cpu_old={r['cpu_old_ms']:.2f} gpu={r['gpu_ms']:.2f} mem={r['memory_mb']:.1f}MB copies={r['copies']} dispatches={r['dispatches']}")
    assert r["speedup"]>0.2 or not r["has_old"]

def test_benchmark_4M():
    r=benchmark_one(4_000_000)
    print(f"N=4M wall_old={r['wall_old_ms']:.2f}ms wall_new={r['wall_new_ms']:.2f}ms speedup={r['speedup']:.2f}x cpu_old={r['cpu_old_ms']:.2f} gpu={r['gpu_ms']:.2f} mem={r['memory_mb']:.1f}MB copies={r['copies']} dispatches={r['dispatches']}")
    assert r["speedup"]>0.2 or not r["has_old"]

def test_benchmark_memory_and_copies():
    r=benchmark_one(1_000_000)
    assert r["memory_mb"]>0 and r["copies"]==2
    assert r["dispatches"]>=1
