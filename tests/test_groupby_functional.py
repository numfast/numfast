"""test_groupby_functional — empty/single/one group/all unique, stable order, M=count distinct"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
from GroupBy._lib.groupby import GroupBy, groupby_array, groupby

def ref_groupby(keys, vals):
    keys = np.asarray(keys)
    vals = np.asarray(vals, dtype=np.float32)
    if keys.size == 0:
        return {"keys": np.array([], dtype=np.int32), "count": np.array([], dtype=np.int64),
                "sum": np.array([], dtype=np.float32), "min": np.array([], dtype=np.float32),
                "max": np.array([], dtype=np.float32), "mean": np.array([], dtype=np.float32)}
    # pandas-like: sort keys stable
    # handle NaN: NaN last
    perm = np.argsort(keys, kind='stable')
    sk = keys[perm]; sv = vals[perm]
    # pandas would also sort but naive: use python grouping
    uniq = []
    counts=[]; sums=[]; mins=[]; maxs=[]
    i=0
    n=len(sk)
    while i<n:
        k=sk[i]; j=i
        # handle NaN equality
        if isinstance(k, float) and np.isnan(k):
            while j<n and np.isnan(sk[j]): j+=1
        else:
            while j<n and sk[j]==k: j+=1
        seg = sv[i:j]
        uniq.append(k); counts.append(j-i); sums.append(float(np.sum(seg))); mins.append(float(np.min(seg))); maxs.append(float(np.max(seg)))
        i=j
    uniq=np.array(uniq)
    return {"keys": uniq, "count": np.array(counts, dtype=np.int64),
            "sum": np.array(sums, dtype=np.float32), "min": np.array(mins, dtype=np.float32),
            "max": np.array(maxs, dtype=np.float32), "mean": np.array([s/c for s,c in zip(sums,counts)], dtype=np.float32)}

def assert_groupby_equal(out, exp):
    assert out["keys"].size == exp["keys"].size, f"M mismatch {out['keys'].size} vs {exp['keys'].size}"
    # keys may be int32/float32 with NaN; compare with equal or both nan
    if out["keys"].size>0:
        for a,b in zip(out["keys"], exp["keys"]):
            if isinstance(a, float) and np.isnan(a) and np.isnan(b): continue
            if isinstance(b,float) and np.isnan(b) and np.isnan(a): continue
            assert a==b, f"keys {a} vs {b}"
        assert np.array_equal(out["count"], exp["count"])
        assert np.allclose(out["sum"], exp["sum"], atol=1e-5)
        assert np.allclose(out["min"], exp["min"], atol=1e-5, equal_nan=True)
        assert np.allclose(out["max"], exp["max"], atol=1e-5, equal_nan=True)
        assert np.allclose(out["mean"], exp["mean"], atol=1e-5, equal_nan=True)

def test_empty():
    out = GroupBy([], [])
    assert out["keys"].size==0 and out["count"].size==0 and out["sum"].size==0
    out2 = groupby(np.array([], dtype=np.int32), np.array([], dtype=np.float32))
    assert out2["keys"].size==0
    exp = ref_groupby([], [])
    assert_groupby_equal(out, exp)

def test_single():
    keys = np.array([5], dtype=np.int32); vals = np.array([10.0], dtype=np.float32)
    out = GroupBy(keys, vals)
    exp = ref_groupby(keys, vals)
    assert_groupby_equal(out, exp)
    assert out["keys"][0]==5 and out["count"][0]==1 and out["sum"][0]==10 and out["mean"][0]==10

def test_one_group():
    keys = np.array([7,7,7,7], dtype=np.int32); vals = np.array([1,2,3,4], dtype=np.float32)
    out = GroupBy(keys, vals)
    exp = ref_groupby(keys, vals)
    assert_groupby_equal(out, exp)
    assert out["keys"].size==1 and out["count"][0]==4 and out["sum"][0]==10 and out["mean"][0]==2.5
    assert out["min"][0]==1 and out["max"][0]==4

def test_all_unique():
    keys = np.array([3,1,4,2], dtype=np.int32); vals = np.array([10,20,30,40], dtype=np.float32)
    out = GroupBy(keys, vals)
    exp = ref_groupby(keys, vals)
    assert_groupby_equal(out, exp)
    assert out["keys"].size==4
    # sorted ascending
    assert list(out["keys"]) == [1,2,3,4]

def test_stable_order():
    # keys stable: groups sorted, but vals within group preserves order not affecting sum etc
    # test that keys sorted stable still groups correct regardless of input order
    keys = np.array([2,1,2,1,2], dtype=np.int32)
    vals = np.array([100,1,200,2,300], dtype=np.float32)
    out = GroupBy(keys, vals)
    # group 1: vals [1,2] sum 3, group 2: [100,200,300] sum 600
    assert list(out["keys"]) == [1,2]
    assert list(out["count"]) == [2,3]
    assert list(out["sum"]) == [3,600]
    # M=count distinct
    assert out["keys"].size == len(np.unique(keys))

def test_nan_last():
    keys = np.array([1, np.nan, 0, np.nan, 2], dtype=np.float32)
    vals = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    out = GroupBy(keys, vals)
    # NaN group last
    assert np.isnan(out["keys"][-1])
    assert not np.isnan(out["keys"][0])
    # NaN group sum: 20+40=60
    nan_idx = np.where(np.isnan(out["keys"]))[0][0]
    assert out["count"][nan_idx]==2 and out["sum"][nan_idx]==60
    # order: 0,1,2, NaN
    assert out["keys"][0]==0 and out["keys"][1]==1 and out["keys"][2]==2

def test_m_count_distinct():
    rng = np.random.RandomState(42)
    for n in [0,1,10,100]:
        keys = rng.randint(0,5,size=n).astype(np.int32)
        vals = rng.randn(n).astype(np.float32)
        out = GroupBy(keys, vals)
        exp_keys = np.unique(keys) if n>0 else np.array([])
        assert out["keys"].size == exp_keys.size

def test_mean_mapbinary():
    keys = np.array([0,0,1,1,1], dtype=np.int32); vals = np.array([2,4,3,6,9], dtype=np.float32)
    out = GroupBy(keys, vals)
    assert np.allclose(out["mean"], np.array([3,6], dtype=np.float32))
