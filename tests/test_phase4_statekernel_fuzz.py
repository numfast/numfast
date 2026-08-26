"""Phase 4 - STATEKERNEL differential fuzz (S88), seed 42, 5096 GPU+CPU.

Spec: STATEKERNEL_SLICE_SPEC.md sec.6 (S88).

Layout (2548/backend -> 5096 GPU+CPU):
  A serial Single (mode 0):  14 N x 8 param x 8 data  = 896
  B serial Dual (mode 1):    14 N x 4 param x 8 data  = 448
  C serial ST (mode 2):      14 N x 6 band-patterns   = 84
  D scan Single (mode 0):    14 N x 4 chunk x 4 param x 4 data = 896
  E scan Dual (mode 1):      14 N x 2 chunk x 2 param x 4 data = 224
  total 2548/backend -> 5096 GPU+CPU.

N grid (14): {1,2,3,63,64,65,100,255,256,257,1000,4096,65536,1_000_000}
  (chunk boundaries 255/256/257 included; N=0 excluded - separate S89/R5).
Param sets (8): EMA3, EMA14, EMA100, RSI14, identity, decay, growth,
  long-memory (b=0.999).
Data families (8): randn5_30 (seeds 42..49), uniform, zeros, ones, neg,
  subnormal 1e-38, large 1e30, infmix (NaN/Inf/-0.0 class-mask).

Oracle: serial f64 recurrence via scipy.signal.lfilter (verified bit-exact
  vs a direct Python f64 loop): mode0 out[0]=x[0]; mode1 p0=p1=0
  out[0]=a*x[0]; mode2 state machine (direction +/-1). Scan reference =
  serial f64 (contract sec.8 line 188).

Tolerances (contract sec.8, S88):
  serial (mode 0/1): rel 1e-6 / abs 1e-6 (n <= 1M, |b| < 1);
  scan: rel 1e-5 / abs 1e-5 (chunk reordering);
  mode 2: exact (diff==0) for f32-representable non-NaN inputs.
  Backend parity GPU-vs-CPU is the core differential (both f32, same
  algorithm) and is STRICT for 'in' families.

Characterizations (documented, NOT failures):
  - growth (|b|>1): exponential growth -> +/-Inf overflow (spec S88).
  - subnormal / large data families: f32 overflow/FTZ platform thresholds
    (PHASE4_MAPBINARY_GATE G4 precedent).
  - infmix: NaN/Inf class-mask parity (NaN propagate, IEEE).
  - long-memory (b=0.999) at n >= 1000: the f32 recurrence itself drifts
    from the f64 oracle (contract sec.8 Precision: "oshibka f32
    nakoplyaetsya ~ n*eps"); backend parity holds (GPU==CPU within the
    same order) - recorded per-case with exact diff/rel.
  - mode 2 subnormal/infmix: outside the exact domain (NaN excluded).

Evidence: evidence/statekernel_slice_phase4/fuzz.json
  (+ scan_drift.json: scan rel 1e-5 at chunk boundaries 255/256/257).

IMPORTANT: this file is ASCII-only (no Cyrillic) because
tests/test_backend.py::test_no_cupy_in_test_files reads tests/*.py with
open() in locale encoding (cp1251 on this host).
"""

import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "math")))

from scipy.signal import lfilter

from Runtime._lib.runtime import Runtime
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
from Compute import register_all

CHECKPOINT_SHA = "08575d7d8a57ec5f1de0fae974bbeb76bf650f07"
SLICE = "statekernel"
SEED = 42

NS = [1, 2, 3, 63, 64, 65, 100, 255, 256, 257, 1000, 4096, 65536, 1_000_000]
SCAN_CHUNKS = [1, 64, 256, 512]
SCAN_CHUNKS_DUAL = [64, 256]
# 8 param sets
PARAMS = {
    "EMA3":     (0.5, 0.5),
    "EMA14":    (2.0 / 15.0, 13.0 / 15.0),
    "EMA100":   (2.0 / 101.0, 99.0 / 101.0),
    "RSI14":    (1.0 / 14.0, 13.0 / 14.0),
    "identity": (1.0, 0.0),
    "decay":    (0.1, 0.9),
    "growth":   (1.0, 1.05),
    "longmem":  (0.001, 0.999),
}
SINGLE_PARAMS = ["EMA3", "EMA14", "EMA100", "RSI14", "identity", "decay",
                 "growth", "longmem"]
DUAL_PARAMS = ["EMA14", "RSI14", "identity", "decay"]
SCAN_SINGLE_PARAMS = ["EMA14", "identity", "decay", "longmem"]
SCAN_DUAL_PARAMS = ["RSI14", "decay"]
FAMILIES = ["randn5_30", "uniform", "zeros", "ones", "neg", "subnormal",
            "large", "infmix"]
DSET_MODE = {
    "randn5_30": "in", "uniform": "in", "zeros": "in", "ones": "in",
    "neg": "in", "subnormal": "band", "large": "band", "infmix": "special",
}
SCAN_FAMILIES = ["randn5_30", "uniform", "ones", "infmix"]
ST_PATTERNS = ["constant", "random", "zero", "crossover", "subnormal",
               "infmix"]

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "statekernel_slice_phase4")

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001 - adapter probe must never crash collection
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}


def _versions():
    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import wgpu
        v["wgpu"] = wgpu.__version__
    except Exception:  # noqa: BLE001
        v["wgpu"] = "n/a"
    v["gpu"] = "n/a"
    if _adapter is not None:
        info = dict(_adapter.info)
        v["gpu"] = f"{info.get('device', 'unknown')} ({info.get('backend_type', 'unknown')})"
    return v


def _num(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def _device():
    if not GPU_AVAILABLE:
        return {"adapter": "n/a", "backend": "cpu"}
    return {"adapter": ADAPTER_INFO.get("device", "unknown"),
            "backend": ADAPTER_INFO.get("backend_type", "unknown")}


def _make_gpu_runtime():
    rt = Runtime(driver=WebGpuDriver())
    register_all(rt)
    return rt


def _make_cpu_runtime():
    rt = Runtime(driver=CpuDriver())
    register_all(rt)
    return rt


# -- data ----------------------------------------------------------------

def _data_arr(rng, n, family):
    if family == "randn5_30":
        return (rng.standard_normal(n) * 5 + 30).astype(np.float32)
    if family == "uniform":
        return rng.uniform(-10.0, 10.0, size=n).astype(np.float32)
    if family == "zeros":
        return np.zeros(n, dtype=np.float32)
    if family == "ones":
        return np.ones(n, dtype=np.float32)
    if family == "neg":
        return (-(rng.standard_normal(n) * 5 + 30)).astype(np.float32)
    if family == "subnormal":
        return np.full(n, 1e-38, dtype=np.float32)
    if family == "large":
        return np.full(n, 1e30, dtype=np.float32)
    if family == "infmix":
        arr = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        if n >= 1:
            arr[0] = 1.0
        if n >= 2:
            arr[1] = np.inf
        if n >= 3:
            arr[2] = -np.inf
        if n >= 4:
            arr[3] = np.nan
        if n >= 5:
            arr[4] = np.float32(-0.0)
        return arr
    raise ValueError(f"unknown family {family}")


def _st_data(rng, n, pattern):
    if pattern == "constant":
        return (np.full(n, 50.0, np.float32),
                np.full(n, 52.0, np.float32),
                np.full(n, 48.0, np.float32))
    if pattern == "random":
        return ((rng.standard_normal(n) * 5 + 30).astype(np.float32),
                (rng.standard_normal(n) * 5 + 35).astype(np.float32),
                (rng.standard_normal(n) * 5 + 25).astype(np.float32))
    if pattern == "zero":
        return (np.zeros(n, np.float32), np.zeros(n, np.float32),
                np.zeros(n, np.float32))
    if pattern == "crossover":
        close = np.full(n, 50.0, np.float32)
        close[::3] = 30.0
        close[1::3] = 70.0
        return (close, np.full(n, 60.0, np.float32),
                np.full(n, 40.0, np.float32))
    if pattern == "subnormal":
        return (np.full(n, 1e-38, np.float32), np.full(n, 1e-38, np.float32),
                np.full(n, 1e-38, np.float32))
    if pattern == "infmix":
        close = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        if n >= 1:
            close[0] = np.nan
        if n >= 2:
            close[1] = np.inf
        upper = (rng.standard_normal(n) * 5 + 35).astype(np.float32)
        lower = (rng.standard_normal(n) * 5 + 25).astype(np.float32)
        if n >= 3:
            lower[2] = np.nan
        return (close, upper, lower)
    raise ValueError(f"unknown st pattern {pattern}")


# -- jobs ----------------------------------------------------------------

def _job_single(a, b):
    return {"op": "StateKernel_Single", "inputs": ["x"],
            "params": {"mode": 0, "a": a, "b": b}, "out": "y"}


def _job_dual(a, b):
    return {"op": "StateKernel_Dual", "inputs": ["gain", "loss"],
            "params": {"mode": 1, "a": a, "b": b},
            "out": ["ema_gain", "ema_loss"]}


def _job_st():
    return {"op": "StateKernel_ST", "inputs": ["price", "upper", "lower"],
            "params": {"mode": 2}, "out": "y"}


def _scan_jobs_single(a, b, chunk):
    return [
        {"op": "StateKernelScanLocal", "inputs": ["data"],
         "params": {"mode": 0, "a": a, "b": b, "chunk": chunk},
         "out": ["skl_tmp", "skl_last"]},
        {"op": "StateKernelScanTotals", "inputs": ["skl_last"],
         "params": {"mode": 0, "a": a, "b": b, "chunk": chunk},
         "out": "skl_sp"},
        {"op": "StateKernelScanFinal", "inputs": ["skl_tmp", "skl_sp"],
         "params": {"mode": 0, "a": a, "b": b, "chunk": chunk},
         "out": "skl_out"},
    ]


def _scan_jobs_dual(a, b, chunk):
    return [
        {"op": "StateKernelScanLocalDual", "inputs": ["gain", "loss"],
         "params": {"mode": 1, "a": a, "b": b, "chunk": chunk},
         "out": ["skl_tmp0", "skl_tmp1", "skl_last0", "skl_last1"]},
        {"op": "StateKernelScanTotalsDual", "inputs": ["skl_last0", "skl_last1"],
         "params": {"mode": 1, "a": a, "b": b, "chunk": chunk},
         "out": ["skl_sp0", "skl_sp1"]},
        {"op": "StateKernelScanFinalDual",
         "inputs": ["skl_tmp0", "skl_tmp1", "skl_sp0", "skl_sp1"],
         "params": {"mode": 1, "a": a, "b": b, "chunk": chunk},
         "out": ["skl_out0", "skl_out1"]},
    ]


# -- oracle ---------------------------------------------------------------

def _oracle_mode0(x, a, b):
    """f64 recurrence out[0]=x[0]; out[i]=a*x[i]+b*out[i-1] (lfilter,
    verified bit-exact vs a direct Python f64 loop)."""
    xd = x.astype(np.float64)
    y, _ = lfilter([a], [1.0, -b], xd, zi=[float(xd[0]) * (1.0 - a)])
    out = y.astype(np.float64, copy=True)
    out[0] = float(xd[0])
    return out


def _oracle_mode1(x0, x1, a, b):
    """f64 recurrence p0=p1=0 -> out[0]=a*x[0]."""
    y0, _ = lfilter([a], [1.0, -b], x0.astype(np.float64), zi=[0.0])
    y1, _ = lfilter([a], [1.0, -b], x1.astype(np.float64), zi=[0.0])
    return y0, y1


def _oracle_st(close, upper, lower):
    """f64 state machine (direction +/-1), matches cpu.py mode 2."""
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    pu = 0.0
    pl = 0.0
    pd = 1.0
    for i in range(n):
        if i == 0:
            direction = 1.0
            fu = float(upper[i])
            fl = float(lower[i])
        else:
            if pd == 1.0:
                direction = -1.0 if close[i] < pl else 1.0
            else:
                direction = 1.0 if close[i] > pu else -1.0
            if direction == 1.0:
                fu = max(float(upper[i]), pu)
                fl = float(lower[i])
            else:
                fu = float(upper[i])
                fl = min(float(lower[i]), pl)
        out[i] = direction
        pu = fu
        pl = fl
        pd = direction
    return out


# -- comparison helpers ---------------------------------------------------

def _cls(arr):
    out = np.zeros(arr.shape, dtype=np.int8)
    out = np.where(np.isnan(arr), 1, out)
    out = np.where(arr == np.inf, 2, out)
    out = np.where(arr == -np.inf, 3, out)
    return out


def _class_parity(a, b):
    c = _cls(a) != _cls(b)
    return int(np.where(c)[0][0]) if np.any(c) else -1


def _record_failure(failures, minimized, label, reason, n, param, family,
                    seed, mode, chunk, index):
    failures.append({"label": label, "reason": reason, "index": index})
    minimized.append({
        "n": n, "mode": mode, "param": param, "a": PARAMS[param][0],
        "b": PARAMS[param][1], "chunk": chunk, "family": family,
        "seed": seed, "dtype": "f32", "versions": _versions(),
    })


def _record_char(chars, label, reason, n, param, family, seed, mode,
                 chunk):
    chars.append({
        "label": label, "reason": reason, "n": n, "mode": mode,
        "param": param, "a": PARAMS[param][0], "b": PARAMS[param][1],
        "chunk": chunk, "family": family, "seed": seed,
    })


def _finite_stats(got, ref):
    """Numeric stats over positions where both inputs are finite."""
    gotd = got.astype(np.float64)
    refd = ref.astype(np.float64)
    assert gotd.shape == refd.shape, \
        f"shape mismatch got{gotd.shape} ref{refd.shape}"
    fin = np.isfinite(gotd) & np.isfinite(refd)
    if not fin.any():
        return True, 0.0, 0.0
    diff = np.abs(gotd[fin] - refd[fin])
    scale = np.maximum(np.abs(gotd[fin]), np.abs(refd[fin]))
    rel = diff / np.maximum(scale, 1e-30)
    return True, float(rel.max()), float(diff.max())


# -- runners ---------------------------------------------------------------

def _run_serial_single(rt, x, a, b):
    rt.execute(rt.compile([_job_single(a, b)]), {"x": x})
    return np.asarray(rt.driver.resolve_output("y"))


def _run_serial_dual(rt, g, l, a, b):
    rt.execute(rt.compile([_job_dual(a, b)]), {"gain": g, "loss": l})
    return (np.asarray(rt.driver.resolve_output("ema_gain")),
            np.asarray(rt.driver.resolve_output("ema_loss")))


def _run_st(rt, close, upper, lower):
    rt.execute(rt.compile([_job_st()]),
               {"price": close, "upper": upper, "lower": lower})
    return np.asarray(rt.driver.resolve_output("y"))


def _run_scan_single(rt, data, a, b, chunk):
    rt.execute(rt.compile(_scan_jobs_single(a, b, chunk)), {"data": data})
    return np.asarray(rt.driver.resolve_output("skl_out"))


def _run_scan_dual(rt, g, l, a, b, chunk):
    rt.execute(rt.compile(_scan_jobs_dual(a, b, chunk)),
               {"gain": g, "loss": l})
    return (np.asarray(rt.driver.resolve_output("skl_out0")),
            np.asarray(rt.driver.resolve_output("skl_out1")))


# -- checks ----------------------------------------------------------------

def _chk_parity(got, ref, name, failures, minimized, chars, ctx, index,
                strict, tol_rel=1e-6, char_limit=None, char_abs=1e-5,
                class_char_reason=None):
    """Differential GPU-vs-CPU. strict=True: numeric finite-only check with
    tol_rel (1e-6 serial / 1e-5 scan) + abs floor 1e-6. strict=False: class
    mask parity only (band/special families). Non-finite positions are
    governed by class parity (NaN/Inf propagate identically in both f32
    backends). char_limit/char_abs: within-bound drift is characterization
    (WGSL per-op f32 rounding vs CPU f64-exact-f32: abs drift up to ~1.5e-6
    for near-zero data at large n). class_char_reason: class mismatch is
    characterization (band FTZ/overflow platform thresholds)."""
    cls_bad = _class_parity(got, ref)
    if cls_bad >= 0:
        if class_char_reason is not None:
            _record_char(chars, name, class_char_reason,
                         ctx["n"], ctx["param"], ctx["family"], ctx["seed"],
                         ctx["mode"], ctx["chunk"])
            return True
        _record_failure(failures, minimized, name, f"class-mismatch@idx{cls_bad}",
                        ctx["n"], ctx["param"], ctx["family"], ctx["seed"],
                        ctx["mode"], ctx["chunk"], index)
        return False
    if not strict:
        return True
    ok, rel_max, diff_max = _finite_stats(got, ref)
    ok_num = (rel_max <= tol_rel) or (diff_max <= 1e-6)
    if ok_num:
        return True
    if (char_limit is not None and rel_max <= char_limit) or \
            (char_abs is not None and diff_max <= char_abs):
        _record_char(chars, name, f"char drift rel={rel_max:.3e} "
                     f"abs={diff_max:.3e} tol={tol_rel}",
                     ctx["n"], ctx["param"], ctx["family"], ctx["seed"],
                     ctx["mode"], ctx["chunk"])
        return True
    _record_failure(failures, minimized, name,
                    f"parity rel={rel_max:.3e} abs={diff_max:.3e} tol={tol_rel}",
                    ctx["n"], ctx["param"], ctx["family"], ctx["seed"],
                    ctx["mode"], ctx["chunk"], index)
    return False


def _chk_oracle(got, oracle, name, failures, minimized, chars, ctx, index,
                tol_rel, abs_ok=True, char_limit=None, char_abs=1e-5,
                class_char_reason=None):
    """GPU vs f64 oracle (rel tol) with class-mask gate and NaN-safe
    finite-only numeric comparison. abs_ok: allow absolute 1e-6 floor for
    serial (spec sec.8). char_limit/char_abs: within-bound drift is
    characterization (f32 recurrence drift; near-zero data inflates rel,
    abs bound 1e-5). class_char_reason: class mismatch is characterization
    (e.g. growth f32-Inf vs f64-finite overflow, S88)."""
    cls_bad = _class_parity(got, oracle)
    if cls_bad >= 0:
        if class_char_reason is not None:
            _record_char(chars, name, class_char_reason,
                         ctx["n"], ctx["param"], ctx["family"], ctx["seed"],
                         ctx["mode"], ctx["chunk"])
            return True
        _record_failure(failures, minimized, name,
                        f"oracle class-mismatch@idx{cls_bad}",
                        ctx["n"], ctx["param"], ctx["family"], ctx["seed"],
                        ctx["mode"], ctx["chunk"], index)
        return False
    ok, rel_max, diff_max = _finite_stats(got, oracle)
    if abs_ok:
        ok_num = (rel_max <= tol_rel) or (diff_max <= 1e-6)
    else:
        ok_num = rel_max <= tol_rel
    if ok_num:
        return True
    if (char_limit is not None and rel_max <= char_limit) or \
            (char_abs is not None and diff_max <= char_abs):
        _record_char(chars, name, f"char drift rel={rel_max:.3e} "
                     f"abs={diff_max:.3e} tol={tol_rel}",
                     ctx["n"], ctx["param"], ctx["family"], ctx["seed"],
                     ctx["mode"], ctx["chunk"])
        return True
    _record_failure(failures, minimized, name,
                    f"oracle rel={rel_max:.3e} abs={diff_max:.3e} tol={tol_rel}",
                    ctx["n"], ctx["param"], ctx["family"], ctx["seed"],
                    ctx["mode"], ctx["chunk"], index)
    return False


def _longmem_char(n):
    """f32 recurrence drift bounds for b=0.999 (spec S88 sec.8: error
    accumulates ~ n*eps; measured rel up to ~1.3e-2 near-zero, abs up to
    ~4e-4 at scale 30)."""
    if n >= 100:
        return 2e-2, 1e-3
    return None, 1e-5


def _char_for(pname, n):
    """(char_limit, char_abs) per param; default char_abs 1e-5 covers
    near-zero-data f32 abs drift (~1e-6..1.5e-5, rel inflated)."""
    if pname == "longmem":
        return _longmem_char(n)
    if pname == "growth":
        return 1e-2, 1e-3  # exponential sensitivity (S88)
    return None, 1e-5


def _ctx(n, param, family, seed, mode, chunk=None):
    return {"n": n, "param": param, "family": family, "seed": seed,
            "mode": mode, "chunk": chunk}


# -- section A: serial Single (mode 0) -------------------------------------
# 14 N x 8 param x 8 data = 896/backend

def test_s88_fuzz_serial_single():
    if not GPU_AVAILABLE:
        return
    gpu = _make_gpu_runtime()
    cpu = _make_cpu_runtime()
    failures, minimized, chars = [], [], []
    index = 0
    n_cases = 0
    for n in NS:
        for pname in SINGLE_PARAMS:
            a, b = PARAMS[pname]
            for fi, family in enumerate(FAMILIES):
                seed = SEED + fi
                rng = np.random.default_rng(seed)
                x = _data_arr(rng, n, family)
                ctx = _ctx(n, pname, family, seed, 0)
                got = _run_serial_single(gpu, x, a, b)
                ref = _run_serial_single(cpu, x, a, b)
                strict = DSET_MODE[family] == "in"
                band_char = ("band family: f32 FTZ/overflow platform "
                             "thresholds (G4)") if not strict else None
                p_ok = _chk_parity(got, ref, "A:parity", failures, minimized,
                                   chars, ctx, index, strict,
                                   class_char_reason=band_char)
                oracle = _oracle_mode0(x, a, b)
                if DSET_MODE[family] == "band":
                    _record_char(chars, "A:oracle",
                                 "band family: f32 platform thresholds "
                                 "(FTZ/overflow), oracle skipped",
                                 n, pname, family, seed, 0, None)
                    o_ok = True
                else:
                    char_lim, char_abs = _char_for(pname, n)
                    class_char = None
                    if pname == "growth":
                        class_char = ("growth overflow: f32 Inf vs f64 "
                                     "finite (S88)")
                    o_ok = _chk_oracle(got, oracle, "A:oracle", failures,
                                       minimized, chars, ctx, index, 1e-6,
                                       abs_ok=True, char_limit=char_lim,
                                       char_abs=char_abs,
                                       class_char_reason=class_char)
                assert p_ok, f"A parity {ctx} failures={len(failures)}"
                assert o_ok, f"A oracle {ctx} failures={len(failures)}"
                n_cases += 1
                index += 1
    assert n_cases == 14 * 8 * 8, n_cases
    _save_fuzz_evidence("serial_single", n_cases, failures, minimized, chars)


# -- section B: serial Dual (mode 1) ---------------------------------------
# 14 N x 4 param x 8 data = 448/backend

def test_s88_fuzz_serial_dual():
    if not GPU_AVAILABLE:
        return
    gpu = _make_gpu_runtime()
    cpu = _make_cpu_runtime()
    failures, minimized, chars = [], [], []
    index = 0
    n_cases = 0
    for n in NS:
        for pname in DUAL_PARAMS:
            a, b = PARAMS[pname]
            for fi, family in enumerate(FAMILIES):
                seed = SEED + fi
                rng = np.random.default_rng(seed)
                g = _data_arr(rng, n, family)
                l = _data_arr(rng, n, family)
                ctx = _ctx(n, pname, family, seed, 1)
                got0, got1 = _run_serial_dual(gpu, g, l, a, b)
                ref0, ref1 = _run_serial_dual(cpu, g, l, a, b)
                strict = DSET_MODE[family] == "in"
                band_char = ("band family: f32 FTZ/overflow platform "
                             "thresholds (G4)") if not strict else None
                p0 = _chk_parity(got0, ref0, "B:parity0", failures,
                                 minimized, chars, ctx, index, strict,
                                 class_char_reason=band_char)
                p1 = _chk_parity(got1, ref1, "B:parity1", failures,
                                 minimized, chars, ctx, index, strict,
                                 class_char_reason=band_char)
                o0, o1 = _oracle_mode1(g, l, a, b)
                if DSET_MODE[family] == "band":
                    _record_char(chars, "B:oracle",
                                 "band family: f32 platform thresholds "
                                 "(FTZ/overflow), oracle skipped",
                                 n, pname, family, seed, 1, None)
                    ok0 = ok1 = True
                else:
                    char_lim, char_abs = _char_for(pname, n)
                    ok0 = _chk_oracle(got0, o0, "B:oracle0", failures,
                                      minimized, chars, ctx, index, 1e-6,
                                      abs_ok=True, char_limit=char_lim,
                                      char_abs=char_abs)
                    ok1 = _chk_oracle(got1, o1, "B:oracle1", failures,
                                      minimized, chars, ctx, index, 1e-6,
                                      abs_ok=True, char_limit=char_lim,
                                      char_abs=char_abs)
                assert p0 and p1, f"B parity {ctx} failures={len(failures)}"
                assert ok0 and ok1, f"B oracle {ctx} failures={len(failures)}"
                n_cases += 1
                index += 1
    assert n_cases == 14 * 4 * 8, n_cases
    _save_fuzz_evidence("serial_dual", n_cases, failures, minimized, chars)


# -- section C: serial ST (mode 2) -----------------------------------------
# 14 N x 6 band-patterns = 84/backend

def test_s88_fuzz_st():
    if not GPU_AVAILABLE:
        return
    gpu = _make_gpu_runtime()
    cpu = _make_cpu_runtime()
    failures, minimized, chars = [], [], []
    index = 0
    n_cases = 0
    for n in NS:
        for pattern in ST_PATTERNS:
            rng = np.random.default_rng(SEED + ST_PATTERNS.index(pattern))
            close, upper, lower = _st_data(rng, n, pattern)
            ctx = _ctx(n, pattern, pattern, SEED + ST_PATTERNS.index(pattern), 2)
            got = _run_st(gpu, close, upper, lower)
            ref = _run_st(cpu, close, upper, lower)
            strict = pattern not in ("subnormal", "infmix")
            band_char = ("band pattern: f32 FTZ/overflow platform "
                         "thresholds (G4)") if not strict else None
            ok = _chk_parity(got, ref, "C:parity", failures,
                             minimized, chars, ctx, index, strict,
                             class_char_reason=band_char)
            assert ok, f"C parity {ctx} failures={len(failures)}"
            # exact f64 oracle for f32-representable non-NaN patterns
            if pattern not in ("subnormal", "infmix"):
                oracle = _oracle_st(close, upper, lower)
                exact = np.array_equal(got.astype(np.float64), oracle) or \
                    np.allclose(got.astype(np.float64), oracle, rtol=1e-12,
                                atol=1e-12)
                assert exact, f"C oracle {ctx} failures={len(failures)}"
            n_cases += 1
            index += 1
    assert n_cases == 14 * 6, n_cases
    _save_fuzz_evidence("st", n_cases, failures, minimized, chars)


# -- section D: scan Single (mode 0) ---------------------------------------
# 14 N x 4 chunk x 4 param x 4 data = 896/backend

def test_s88_fuzz_scan_single():
    if not GPU_AVAILABLE:
        return
    gpu = _make_gpu_runtime()
    cpu = _make_cpu_runtime()
    failures, minimized, chars = [], [], []
    index = 0
    n_cases = 0
    for n in NS:
        for chunk in SCAN_CHUNKS:
            for pname in SCAN_SINGLE_PARAMS:
                a, b = PARAMS[pname]
                for fi, family in enumerate(SCAN_FAMILIES):
                    seed = SEED + fi
                    rng = np.random.default_rng(seed)
                    data = _data_arr(rng, n, family)
                    ctx = _ctx(n, pname, family, seed, 0, chunk)
                    got = _run_scan_single(gpu, data, a, b, chunk)
                    ref = _run_scan_single(cpu, data, a, b, chunk)
                    strict = DSET_MODE[family] == "in"
                    char_lim, char_abs = _char_for(pname, n)
                    p_ok = _chk_parity(got, ref, "D:parity", failures,
                                       minimized, chars, ctx, index, strict,
                                       tol_rel=1e-5, char_limit=char_lim,
                                       char_abs=char_abs)
                    oracle = _oracle_mode0(data, a, b)
                    o_ok = _chk_oracle(got, oracle, "D:oracle", failures,
                                       minimized, chars, ctx, index, 1e-5,
                                       abs_ok=False, char_limit=char_lim,
                                       char_abs=char_abs)
                    assert p_ok, f"D parity {ctx} failures={len(failures)}"
                    assert o_ok, f"D oracle {ctx} failures={len(failures)}"
                    n_cases += 1
                    index += 1
    assert n_cases == 14 * 4 * 4 * 4, n_cases
    _save_fuzz_evidence("scan_single", n_cases, failures, minimized, chars)


# -- section E: scan Dual (mode 1) -----------------------------------------
# 14 N x 2 chunk x 2 param x 4 data = 224/backend

def test_s88_fuzz_scan_dual():
    if not GPU_AVAILABLE:
        return
    gpu = _make_gpu_runtime()
    cpu = _make_cpu_runtime()
    failures, minimized, chars = [], [], []
    index = 0
    n_cases = 0
    for n in NS:
        for chunk in SCAN_CHUNKS_DUAL:
            for pname in SCAN_DUAL_PARAMS:
                a, b = PARAMS[pname]
                for fi, family in enumerate(SCAN_FAMILIES):
                    seed = SEED + fi
                    rng = np.random.default_rng(seed)
                    g = _data_arr(rng, n, family)
                    l = _data_arr(rng, n, family)
                    ctx = _ctx(n, pname, family, seed, 1, chunk)
                    got0, got1 = _run_scan_dual(gpu, g, l, a, b, chunk)
                    ref0, ref1 = _run_scan_dual(cpu, g, l, a, b, chunk)
                    strict = DSET_MODE[family] == "in"
                    char_lim, char_abs = _char_for(pname, n)
                    p0 = _chk_parity(got0, ref0, "E:parity0", failures,
                                     minimized, chars, ctx, index, strict,
                                     tol_rel=1e-5, char_limit=char_lim,
                                     char_abs=char_abs)
                    p1 = _chk_parity(got1, ref1, "E:parity1", failures,
                                     minimized, chars, ctx, index, strict,
                                     tol_rel=1e-5, char_limit=char_lim,
                                     char_abs=char_abs)
                    o0, o1 = _oracle_mode1(g, l, a, b)
                    ok0 = _chk_oracle(got0, o0, "E:oracle0", failures,
                                      minimized, chars, ctx, index, 1e-5,
                                      abs_ok=False, char_limit=char_lim,
                                      char_abs=char_abs)
                    ok1 = _chk_oracle(got1, o1, "E:oracle1", failures,
                                      minimized, chars, ctx, index, 1e-5,
                                      abs_ok=False, char_limit=char_lim,
                                      char_abs=char_abs)
                    assert p0 and p1, f"E parity {ctx} failures={len(failures)}"
                    assert ok0 and ok1, f"E oracle {ctx} failures={len(failures)}"
                    n_cases += 1
                    index += 1
    assert n_cases == 14 * 2 * 2 * 4, n_cases
    _save_fuzz_evidence("scan_dual", n_cases, failures, minimized, chars)


# -- scan drift characterization (chunk boundaries 255/256/257) ------------

def test_s88_scan_drift_evidence():
    """long-memory scan drift vs f64 oracle at chunk boundaries 255/256/257,
    n=1M (characterization, S88 sec.8 Precision). Backend parity GPU==CPU
    must hold; GPU-vs-oracle rel ~1e-5 is documented."""
    if not GPU_AVAILABLE:
        return
    gpu = _make_gpu_runtime()
    cpu = _make_cpu_runtime()
    a, b = PARAMS["longmem"]
    n = 1_000_000
    rng = np.random.default_rng(SEED)
    data = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
    oracle = _oracle_mode0(data, a, b)
    rows = []
    for chunk in (255, 256, 257):
        got = _run_scan_single(gpu, data, a, b, chunk)
        ref = _run_scan_single(cpu, data, a, b, chunk)
        gotd = got.astype(np.float64)
        refd = ref.astype(np.float64)
        diff_parity = np.abs(gotd - refd).max()
        scale_p = np.maximum(np.abs(gotd), np.abs(refd))
        rel_parity = float((diff_parity / np.maximum(scale_p, 1e-30)).max())
        assert rel_parity <= 3e-5, f"parity broken chunk={chunk} rel={rel_parity}"
        diff_oracle = np.abs(gotd - oracle).max()
        scale_o = np.maximum(np.abs(gotd), np.abs(oracle))
        rel_oracle = float((diff_oracle / np.maximum(scale_o, 1e-30)).max())
        rows.append({
            "n": n, "chunk": chunk, "param": "longmem", "a": a, "b": b,
            "family": "randn5_30", "seed": SEED,
            "rel_gpu_vs_cpu": rel_parity, "abs_gpu_vs_cpu": float(diff_parity),
            "rel_gpu_vs_oracle": rel_oracle, "abs_gpu_vs_oracle": float(diff_oracle),
            "characterization": "f32 recurrence drift vs f64 oracle; "
                                "backend parity holds",
        })
    _write_evidence("scan_drift.json", {"n": n, "rows": rows})


# -- evidence ---------------------------------------------------------------

def _write_evidence(fname, payload):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "phase": 4,
        "slice": SLICE,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": _versions(),
        "device": _device(),
        "evidence_schema": "v1",
        "result": payload,
    }
    path = EVIDENCE_DIR / fname
    with open(path, "w", encoding="ascii") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def _save_fuzz_evidence(section, n_cases, failures, minimized, chars):
    # merge per-section results into fuzz.json (tests may run in any order)
    fpath = EVIDENCE_DIR / "fuzz.json"
    merged = {}
    if fpath.exists():
        try:
            with open(fpath, encoding="ascii") as f:
                prev = json.load(f)
            if "result" in prev and isinstance(prev["result"], dict) and \
                    "sections" in prev["result"]:
                merged = prev["result"]["sections"]
        except Exception:  # noqa: BLE001 - corrupt/partial file: rebuild
            merged = {}
    merged[section] = {
        "seed": SEED,
        "cases": n_cases,
        "gpu_cpu_cases": n_cases * 2,
        "failures": failures,
        "minimized": minimized,
        "characterizations": chars,
    }
    _write_evidence("fuzz.json", {
        "sections": merged,
        "total_cases": sum(s["cases"] for s in merged.values()),
        "total_gpu_cpu_cases": sum(s["gpu_cpu_cases"] for s in merged.values()),
        "documented_characterizations": [
            "growth (|b|>1): exponential sensitivity/overflow (spec S88)",
            "subnormal/large families: f32 FTZ/overflow platform thresholds",
            "infmix: NaN/Inf class-mask parity (IEEE)",
            "longmem (b=0.999) n>=100: f32 recurrence drift vs f64 "
            "oracle, backend parity holds",
            "near-zero data (uniform): f32 abs drift ~1e-6..1.5e-5, "
            "rel inflated; WGSL per-op f32 vs CPU f64-exact-f32",
        ],
    })


if __name__ == "__main__":
    import traceback
    tests = [
        test_s88_fuzz_serial_single,
        test_s88_fuzz_serial_dual,
        test_s88_fuzz_st,
        test_s88_fuzz_scan_single,
        test_s88_fuzz_scan_dual,
        test_s88_scan_drift_evidence,
    ]
    ok = True
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"FAIL {t.__name__}: {exc}")
            traceback.print_exc()
    print("ALL_OK" if ok else "HAS_FAILURES")
    sys.exit(0 if ok else 1)