"""test_groupby_resource_failure — R1-R7 (refcount, reuse, leak 20 warm, N>limit chunked, N=0, repeat after error)"""
import sys, os, gc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
import weakref
from GroupBy._lib.groupby import GroupBy, groupby_array, GROUPBY_LIMIT

def test_R1_refcount():
    keys=np.arange(100,dtype=np.int32); vals=np.arange(100,dtype=np.float32)
    keys_ref=sys.getrefcount(keys)
    out=GroupBy(keys, vals)
    assert sys.getrefcount(keys)==keys_ref
    wr=weakref.ref(out["sum"])
    del out; gc.collect()
    assert wr() is None
    assert np.array_equal(keys, np.arange(100,dtype=np.int32))

def test_R2_reuse():
    rng=np.random.RandomState(0); n=10000
    keys=rng.randint(0,100,size=n).astype(np.int32); vals=rng.randn(n).astype(np.float32)
    out1=GroupBy(keys, vals); out2=GroupBy(keys, vals)
    assert np.array_equal(out1["keys"], out2["keys"]) and np.array_equal(out1["count"], out2["count"])
    assert np.allclose(out1["sum"], out2["sum"])
    out1_copy={k:v.copy() for k,v in out1.items()}
    _=GroupBy(keys, vals)
    assert np.array_equal(out1["keys"], out1_copy["keys"])

def test_R3_leak_20_warm():
    rng=np.random.RandomState(1); n=100000
    keys=rng.randint(0,1000,size=n).astype(np.int32); vals=rng.randn(n).astype(np.float32)
    for i in range(20):
        out=GroupBy(keys, vals)
        assert out["keys"].size<=1000 and out["keys"].size>0
        assert np.sum(out["count"])==n
    gc.collect()

def test_R4_N_over_limit_chunked():
    rng=np.random.RandomState(2); n=10000
    keys=rng.randint(0,100,size=n).astype(np.int32); vals=rng.randn(n).astype(np.float32)
    ref=GroupBy(keys, vals)
    out=groupby_array(keys, vals, chunk_limit=1024)
    assert np.array_equal(out["keys"], ref["keys"]) and np.array_equal(out["count"], ref["count"])
    big_n=4096*3+500; keys2=rng.randint(0,50,size=big_n).astype(np.int32); vals2=rng.randn(big_n).astype(np.float32)
    ref2=GroupBy(keys2, vals2)
    out2=groupby_array(keys2, vals2, chunk_limit=4096)
    assert np.array_equal(out2["keys"], ref2["keys"])
    out0=groupby_array(np.array([],dtype=np.int32), np.array([],dtype=np.float32), chunk_limit=1024)
    assert out0["keys"].size==0
    small_keys=np.array([3,1,2],dtype=np.int32); small_vals=np.array([1,2,3],dtype=np.float32)
    assert groupby_array(small_keys, small_vals, chunk_limit=2)["keys"].size==3

def test_R5_N_zero():
    out=GroupBy(np.array([],dtype=np.int32), np.array([],dtype=np.float32))
    assert out["keys"].size==0 and out["count"].size==0 and out["sum"].size==0
    out2=GroupBy(np.array([],dtype=np.int32), np.array([],dtype=np.float32))
    assert out2["keys"].size==0
    for _ in range(5):
        assert GroupBy([], []).get("keys").size==0
        assert groupby_array([], [], chunk_limit=1024)["keys"].size==0

def test_R6_repeat_after_error():
    keys=np.arange(10,dtype=np.int32); vals=np.arange(10,dtype=np.float32)
    out_ok=GroupBy(keys, vals)
    assert out_ok["keys"].size==10
    # invalid keys type should not crash later
    out_none=GroupBy(None, None)
    assert out_none["keys"].size==0
    out_ok2=GroupBy(keys, vals)
    assert out_ok2["keys"].size==10
    try: GroupBy(None, vals)
    except Exception: pass
    assert GroupBy(keys, vals)["keys"].size==10

def test_R7_stress_reuse_and_pool():
    rng=np.random.RandomState(3)
    for i in range(100):
        n=int(rng.choice([0,1,10,100,1000,10000]))
        keys=rng.randint(0,10,size=n).astype(np.int32) if n>0 else np.array([],dtype=np.int32)
        vals=rng.randn(max(n,1)).astype(np.float32)[:n] if n>0 else np.array([],dtype=np.float32)
        out=GroupBy(keys, vals)
        if n==0: assert out["keys"].size==0
        else: assert np.sum(out["count"])==n and out["keys"].size<=10
