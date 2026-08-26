"""WalkForward Full Fused CPU oracle -- same result as GPU kernel.

Mirrors the reference CPU path (bench_full_wf_stage1.measure_old_wf):
- SMA f32 via shifted-add window sums, warmup 0.0
- RSI f32 Wilder EMA, seed i==1, np.round ties-to-even on rsi*1000
- conditions gt / and-or combine, fold segments
- ret in f64: close[i]/close[i-1]-1 (prev==0 -> 0)
- per (tuple, fold): sum_ret f64, count

Cold runs <= 1000 rows only (reference check, not mass compute).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from Compute._lib.walkforward_full_fused.wgsl import build_tuples


def sma_f32(c_f32, period):
    n = len(c_f32)
    out = np.zeros(n, dtype=np.float32)
    if n >= period:
        s = np.zeros(n, dtype=np.float32)
        for j in range(period):
            s[period - 1:] += c_f32[period - 1 - j:n - j]
        out[period - 1:] = (s[period - 1:] / np.float32(period)).astype(np.float32)
    return out


def rsi_f32(c_f32, period):
    n = len(c_f32)
    d = np.zeros(n, dtype=np.float32)
    d[1:] = c_f32[1:] - c_f32[:-1]
    g = np.maximum(d, 0).astype(np.float32)
    l = np.maximum(-d, 0).astype(np.float32)
    ag = np.zeros(n, dtype=np.float32)
    al = np.zeros(n, dtype=np.float32)
    a = np.float32(1.0 / period)
    if n > 1:
        ag[1] = g[1]
        al[1] = l[1]
    for i in range(2, n):
        ag[i] = a * g[i] + np.float32(1 - a) * ag[i - 1]
        al[i] = a * l[i] + np.float32(1 - a) * al[i - 1]
    rs = np.where(al > 1e-10, ag / np.maximum(al, np.float32(1e-10)), np.float32(100))
    return np.round((100 - 100 / (1 + rs)) * 1000).astype(np.float32)


def _cmp(a, b, op):
    if op == "gt":
        return a > b
    if op == "ge":
        return a >= b
    if op == "lt":
        return a < b
    if op == "le":
        return a <= b
    if op == "eq":
        return a == b
    return a != b


def evaluate(close, tuples=None, folds=4, fold_sz=None):
    """CPU oracle -> (sums[80] f64, counts[80] i64), index (t*folds+fold)."""
    close_f32 = np.asarray(close, dtype=np.float32)
    n = len(close_f32)
    if tuples is None:
        tuples = build_tuples()
    if fold_sz is None:
        fold_sz = max(1, n // folds) if folds > 0 else n
    feats = {}
    for p in [10, 20, 30, 50, 100, 200]:
        feats[f"sma_{p}"] = sma_f32(close_f32, p)
    for p in [14, 28]:
        feats[f"rsi_{p}"] = rsi_f32(close_f32, p)
    sh = np.zeros(n, dtype=np.float64)
    sh[1:] = close_f32[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        div = np.where(sh == 0, 0, close_f32.astype(np.float64) / np.where(sh == 0, 1, sh))
    ret = div - 1.0
    nt = len(tuples)
    sums = np.zeros(nt * folds, dtype=np.float64)
    cnts = np.zeros(nt * folds, dtype=np.int64)
    for t, tp in enumerate(tuples):
        mask = None
        for c in tp["cond"]:
            av = feats[c["a"]] if c["a"] in feats else close_f32
            bv = c["b"]
            if isinstance(bv, str):
                bv = feats[bv] if bv in feats else close_f32
            else:
                bv = np.float32(bv)
            cur = _cmp(av, bv, c["op"])
            mask = cur if mask is None else (
                mask & cur if tp.get("logic", "and") == "and" else mask | cur)
        for fold in range(folds):
            s = fold * fold_sz
            e = min(s + fold_sz, n)
            if s >= n:
                continue
            seg = mask[s:e].astype(np.float64) if mask is not None else np.zeros(e - s)
            cnt = int(seg.sum())
            sums[t * folds + fold] = float((ret[s:e] * seg).sum())
            cnts[t * folds + fold] = cnt
    return sums, cnts


def mean_ret(sums, cnts, folds=4):
    """sums/counts -> mean_ret with nan where count==0 (oracle semantics)."""
    sums = np.asarray(sums, dtype=np.float64)
    cnts = np.asarray(cnts, dtype=np.float64)
    out = np.full(len(sums), float("nan"))
    nz = cnts > 0
    out[nz] = sums[nz] / cnts[nz]
    return out


def cpu(ctx):
    """Runtime CPU driver adapter: ctx.inputs[0]=close, ctx.outputs[0]=u32[160]."""
    n = int(ctx.uniforms.get("n", 0))
    fold_sz = int(ctx.uniforms.get("fold_sz", max(1, n // 4)))
    folds = int(ctx.uniforms.get("folds", 4))
    close = np.asarray(ctx.inputs[0].view._raw[:n], dtype=np.float32)
    sums, cnts = evaluate(close, folds=folds, fold_sz=fold_sz)
    out = np.zeros(len(cnts) * 2, dtype=np.uint32)
    out[0::2] = np.asarray(sums, dtype=np.float32).view(np.uint32)
    out[1::2] = cnts.astype(np.uint32)
    ov = ctx.outputs[0].view
    ov._raw[:len(out)] = out


__all__ = ["evaluate", "mean_ret", "cpu", "sma_f32", "rsi_f32", "build_tuples"]
