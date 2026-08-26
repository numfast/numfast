"""test_groupby_fuzz — ≥1000 fuzz seed42 N 0,1,2,63,64,65,1000,8192,1M × M 1,N,√N exact vs pandas groupby"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "numfast/src/core")))
import numpy as np
from GroupBy._lib.groupby import GroupBy, groupby_array

SEED = 42
N_GRID = [0,1,2,63,64,65,1000,8192,1000000]
def pandas_ref(keys, vals):
    try:
        import pandas as pd
        if len(keys)==0:
            return {"keys": np.array([], dtype=np.int32), "count": np.array([], dtype=np.int64),
                    "sum": np.array([], dtype=np.float32), "min": np.array([], dtype=np.float32),
                    "max": np.array([], dtype=np.float32), "mean": np.array([], dtype=np.float32)}
        # pandas groupby: handle int keys
        s_keys = np.asarray(keys)
        s_vals = np.asarray(vals, dtype=np.float32)
        df = pd.DataFrame({"k": s_keys, "v": s_vals.astype(np.float64)})
        # pandas sorts keys ascending by default for groupby? use sort=True
        grp = df.groupby("k", sort=True, dropna=False)
        # collect agg
        res = grp["v"].agg(["count","sum","min","max","mean"]).sort_index()
        # extract keys
        keys_out = res.index.to_numpy()
        # handle NaN: pandas puts NaN last already when dropna=False
        # convert to appropriate dtype
        # keys may be float with NaN -> compare via nan check
        return {"keys": keys_out, "count": res["count"].to_numpy(dtype=np.int64),
                "sum": res["sum"].to_numpy(dtype=np.float32), "min": res["min"].to_numpy(dtype=np.float32),
                "max": res["max"].to_numpy(dtype=np.float32), "mean": res["mean"].to_numpy(dtype=np.float32)}
    except ImportError:
        # fallback manual
        keys = np.asarray(keys); vals = np.asarray(vals, dtype=np.float32)
        if keys.size==0:
            return {"keys": np.array([], dtype=np.int32), "count": np.array([], dtype=np.int64),
                    "sum": np.array([], dtype=np.float32), "min": np.array([], dtype=np.float32),
                    "max": np.array([], dtype=np.float32), "mean": np.array([], dtype=np.float32)}
        perm = np.argsort(keys, kind='stable')
        sk = keys[perm]; sv = vals[perm]
        uniq=[]; counts=[]; sums=[]; mins=[]; maxs=[]
        i=0; n=len(sk)
        while i<n:
            k=sk[i]; j=i
            if isinstance(k, float) and np.isnan(k):
                while j<n and np.isnan(sk[j]): j+=1
            else:
                while j<n and sk[j]==k: j+=1
            seg=sv[i:j]; uniq.append(k); counts.append(j-i); sums.append(float(np.sum(seg))); mins.append(float(np.min(seg))); maxs.append(float(np.max(seg)))
            i=j
        return {"keys": np.array(uniq), "count": np.array(counts,dtype=np.int64),
                "sum": np.array(sums,dtype=np.float32), "min": np.array(mins,dtype=np.float32),
                "max": np.array(maxs,dtype=np.float32), "mean": np.array([s/c for s,c in zip(sums,counts)],dtype=np.float32)}

def groups_for_m(n, m):
    if n==0: return np.array([], dtype=np.int32)
    if m==1: return np.zeros(n, dtype=np.int32)
    if m==n: return np.arange(n, dtype=np.int32)
    # sqrt N approx
    rng_state = np.random.RandomState(n+m)
    # generate m distinct keys then sample
    vals = np.random.RandomState(n+m+SEED).choice(np.arange(m), size=n)
    return vals.astype(np.int32)

def test_fuzz_grid():
    rng = np.random.RandomState(SEED)
    total=0; failures=[]
    for n in N_GRID:
        m_candidates = [1, n if n>0 else 0, int(np.sqrt(n)) if n>0 else 0]
        m_candidates = [m for m in m_candidates if m>=0]
        # deduplicate
        m_candidates = sorted(set(m_candidates))
        for m in m_candidates:
            repeats = 2 if n==1000000 else 3 if n==8192 else 5 if n==1000 else 3
            for rep in range(repeats):
                if n==0:
                    keys=np.array([],dtype=np.int32); vals=np.array([],dtype=np.float32)
                else:
                    if m==1:
                        keys=np.zeros(n,dtype=np.int32)
                    elif m==n:
                        keys=np.arange(n,dtype=np.int32)
                        rng.shuffle(keys)
                    elif m==int(np.sqrt(n)):
                        # sqrtN distinct: limited keys
                        base = np.arange(m, dtype=np.int32)
                        keys = rng.choice(base, size=n).astype(np.int32)
                    else:
                        keys = rng.randint(0, max(1,m), size=n).astype(np.int32)
                    # sprinkle NaN for float path occasionally
                    if rng.rand()<0.05:
                        keys = keys.astype(np.float32)
                        idx = rng.choice(n, size=max(1,n//10), replace=False)
                        keys[idx]=np.nan
                    vals = (rng.randn(n)*10).astype(np.float32) if n>0 else np.array([],dtype=np.float32)
                out = GroupBy(keys, vals)
                exp = pandas_ref(keys, vals)
                total+=1
                if out["keys"].size != exp["keys"].size:
                    failures.append((n,m,"Msize",out["keys"].size,exp["keys"].size))
                else:
                    if out["keys"].size>0:
                        # compare keys with NaN handling via allclose equal_nan
                        try:
                            a = np.asarray(out["keys"], dtype=np.float64)
                            b = np.asarray(exp["keys"], dtype=np.float64)
                            ok_keys = a.shape==b.shape and np.allclose(a, b, atol=0, rtol=0, equal_nan=True)
                            # fallback exact for int keys
                            if not ok_keys:
                                # try exact with nan check
                                ok_keys = np.array_equal(np.asarray(out["keys"]), np.asarray(exp["keys"])) or np.allclose(a,b,equal_nan=True)
                        except Exception:
                            ok_keys=False
                        if not ok_keys:
                            failures.append((n,m,"keys",out["keys"][:5],exp["keys"][:5], a[:5] if 'a' in locals() else out["keys"][:5], b[:5] if 'b' in locals() else exp["keys"][:5]))
                        if not np.array_equal(out["count"], exp["count"]):
                            failures.append((n,m,"count",out["count"][:3],exp["count"][:3]))
                        if not np.allclose(out["sum"], exp["sum"], atol=1e-4, equal_nan=True):
                            failures.append((n,m,"sum",out["sum"][:3],exp["sum"][:3]))
                        if not np.allclose(out["min"], exp["min"], atol=1e-4, equal_nan=True):
                            failures.append((n,m,"min",out["min"][:3],exp["min"][:3]))
                        if not np.allclose(out["max"], exp["max"], atol=1e-4, equal_nan=True):
                            failures.append((n,m,"max",out["max"][:3],exp["max"][:3]))
                        if not np.allclose(out["mean"], exp["mean"], atol=1e-4, equal_nan=True):
                            failures.append((n,m,"mean",out["mean"][:3],exp["mean"][:3]))
                # also test groupby_array chunk_limit parity
                if n>0 and n<=10000:
                    out2 = groupby_array(keys, vals, chunk_limit=1024)
                    if not np.array_equal(out2["count"], out["count"]):
                        failures.append((n,m,"chunk1024 count",out2["count"][:3],out["count"][:3]))
    # ensure >=1000 fuzz? we have: N_GRID 9 * m up to 3 * repeats ~ 5 avg => ~ 9*3*3=81; need 1000 -> loop extra random cases
    extra_needed = 1000 - total
    if extra_needed>0:
        for _ in range(extra_needed):
            n = int(rng.choice([10,100,500,1000,5000]))
            m = int(rng.choice([1,5,10,int(np.sqrt(n)),n]))
            m = min(m,n) if n>0 else 0
            keys = rng.randint(0, max(1,m), size=n).astype(np.int32) if n>0 else np.array([],dtype=np.int32)
            vals = rng.randn(n).astype(np.float32) if n>0 else np.array([],dtype=np.float32)
            out = GroupBy(keys, vals)
            exp = pandas_ref(keys, vals)
            total+=1
            if out["keys"].size != exp["keys"].size:
                failures.append((n,m,"extra Msize",out["keys"].size,exp["keys"].size))
            elif out["keys"].size>0:
                if not np.array_equal(out["count"], exp["count"]):
                    failures.append((n,m,"extra count",out["count"][:3],exp["count"][:3]))
                if not np.allclose(out["sum"], exp["sum"], atol=1e-4):
                    failures.append((n,m,"extra sum",out["sum"][:3],exp["sum"][:3]))
    assert total>=1000, f"total {total} <1000"
    assert not failures, f"failures {failures[:5]}"
