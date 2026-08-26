"""test_groupby_consumers — 2-3 real Colossus: aggregate PnL per group, tuple group, flat panels parity"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
from GroupBy._lib.groupby import GroupBy, groupby_array

def test_aggregate_pnl_per_group():
    """Real PnL: N=50000 trades, keys=strategy_id 0..9, vals=pnl, verify sum/count/mean parity vs pandas."""
    rng=np.random.RandomState(42)
    n=50000
    strat=rng.randint(0,10,size=n).astype(np.int32)
    pnl=(rng.randn(n)*100).astype(np.float32)
    # add some large PnL spikes
    pnl[::500]+=1000
    out=GroupBy(strat, pnl)
    # pandas ref if available
    try:
        import pandas as pd
        df=pd.DataFrame({"k": strat, "v": pnl.astype(np.float64)})
        grp=df.groupby("k", sort=True).agg(count=("v","count"),sum=("v","sum"),min=("v","min"),max=("v","max"),mean=("v","mean")).sort_index()
        assert out["keys"].size==10
        assert np.array_equal(np.sort(out["keys"]), np.arange(10))
        for i,k in enumerate(out["keys"]):
            row=grp.loc[int(k)]
            assert out["count"][i]==row["count"]
            assert np.isclose(out["sum"][i], row["sum"], atol=1e-2)
            assert np.isclose(out["min"][i], row["min"], atol=1e-4)
            assert np.isclose(out["max"][i], row["max"], atol=1e-4)
            assert np.isclose(out["mean"][i], row["mean"], atol=1e-4)
    except ImportError:
        # fallback: total sum parity
        assert np.isclose(np.sum(out["sum"]), np.sum(pnl), atol=1e-2)
        assert np.sum(out["count"])==n
    # also tuple group simulation: multi-key via combined int key
    print(f"PnL per group keys {out['keys'][:3]} sums {out['sum'][:3]}")

def test_tuple_group():
    """Tuple group: composite keys (a,b) encoded as a*100 + b, verify GroupBy equals manual tuple group."""
    rng=np.random.RandomState(123)
    n=20000
    a=rng.randint(0,5,size=n).astype(np.int32)
    b=rng.randint(0,5,size=n).astype(np.int32)
    vals=rng.randn(n).astype(np.float32)*10
    # encode tuple as single int32 key
    keys=(a*100 + b).astype(np.int32)
    out=GroupBy(keys, vals)
    # manual tuple dict
    from collections import defaultdict
    d=defaultdict(list)
    for ka,kb,v in zip(a,b,vals):
        d[(int(ka),int(kb))].append(float(v))
    assert out["keys"].size==len(d)
    # check a few groups
    for k, cnt, s in zip(out["keys"][:5], out["count"][:5], out["sum"][:5]):
        ak=int(k)//100; bk=int(k)%100
        seg=d[(ak,bk)]
        assert cnt==len(seg) and np.isclose(s, sum(seg), atol=1e-3)

def test_flat_panels_parity():
    """Flat panels parity: time groups, GroupBy vs numpy bincount-like parity."""
    rng=np.random.RandomState(99)
    n=100000
    # panel ids 0..99 repeated
    panel=rng.randint(0,100,size=n).astype(np.int32)
    vals=rng.randn(n).astype(np.float32)
    out=GroupBy(panel, vals)
    # parity via numpy unique + bincount manual
    uniq, inv = np.unique(panel, return_inverse=True)
    # sort uniq corresponds to out keys sorted ascending -> should match
    assert np.array_equal(np.sort(uniq), out["keys"])
    # verify per group sum via np.add.at
    sums_manual=np.zeros(uniq.size, dtype=np.float64)
    np.add.at(sums_manual, inv, vals.astype(np.float64))
    # map uniq sorted order to sums_manual order: uniq is sorted, out keys sorted -> directly compare
    # reorder sums_manual to match sorted uniq (already sorted)
    assert np.allclose(out["sum"].astype(np.float64), sums_manual, atol=1e-3)
    # also chunk_limit parity
    out_chunk=groupby_array(panel, vals, chunk_limit=8192)
    assert np.array_equal(out_chunk["keys"], out["keys"])
    assert np.allclose(out_chunk["mean"], out["mean"], atol=1e-5)

def test_pnl_flat_and_tuple_combined():
    rng=np.random.RandomState(7)
    n=10000
    strat=rng.randint(0,4,size=n).astype(np.int32)
    pnl=rng.randn(n).astype(np.float32)
    out=GroupBy(strat, pnl)
    assert np.sum(out["count"])==n
    assert out["keys"].size==4
    # mean parity sum/count
    for s,c,m in zip(out["sum"], out["count"], out["mean"]):
        assert np.isclose(s/c, m, atol=1e-5)
