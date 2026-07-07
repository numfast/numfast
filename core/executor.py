"""Execution — Planner + Kernel (CPU Runtime).

Жизненный цикл:
  User API → Planner (merge by OpID+Input) → ExecutionGraph → Kernel → Driver

Правила:
  I3: Planner не знает алгоритмы (только OpID, Inputs, Params).
  I4: Driver не знает операции.
  Kernel — единственное место, где алгоритмы живут.
  - .meta — SeriesMeta (offset, scale, valid_from, signed)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from numfast.core.nseries import NumericSeries
from numfast.core.driver import CpuDriver


# ── Job ──────────────────────────────────────────────────────────────

@dataclass
class Job:
    """Одна вычислительная единица.

    Planner ничего не знает про алгоритмы — для Planner Job это просто:
      - Что делать (op_id)
      - Над чем (inputs)
      - С какими параметрами (params)
      - Куда писать (outputs)

    Args:
        op_id: идентификатор операции ("sma", "ema", "rsi", ...).
        inputs: список входных рядов.
        params: параметры операции (w, alpha, ...).
        outputs: список выходных рядов (заполняются Kernel).
        valid_from: индекс первой полноценной метки результата.
    """
    op_id: str
    inputs: List[NumericSeries]
    params: Dict[str, Any]
    outputs: List[NumericSeries] = field(default_factory=list)
    valid_from: int = 0

    def __repr__(self) -> str:
        return (
            f"Job('{self.op_id}', "
            f"inputs={len(self.inputs)}, "
            f"params={self.params}, "
            f"outputs={len(self.outputs)}, "
            f"valid_from={self.valid_from})"
        )


# ── ExecutionGraph ──────────────────────────────────────────────────

@dataclass
class ExecutionGraph:
    """Группа Job для одного прохода.

    Planner собирает Job, ExecutionGraph передаёт Kernel.
    """
    jobs: List[Job] = field(default_factory=list)

    def add(self, job: Job) -> None:
        self.jobs.append(job)

    def __len__(self) -> int:
        return len(self.jobs)

    def __repr__(self) -> str:
        return f"ExecutionGraph({len(self.jobs)} jobs)"


# ── Merge functions registry ─────────────────────────────────────────

def merge_sma(existing: dict, new: dict) -> dict:
    """Combine w params из нескольких SMA вызовов.

    SMA(close, w=14) + SMA(close, w=28) → {w: [14, 28]}
    """
    merged = dict(existing)
    for k, v in new.items():
        if k in existing:
            old = existing[k]
            if not isinstance(old, list):
                merged[k] = [old]
            if isinstance(v, list):
                merged[k].extend(v)
            else:
                merged[k].append(v)
        else:
            merged[k] = v
    return merged


merge_window = merge_sma  # alias для любых оконных операций


def _ensure_list(val):
    """Привести к списку, если ещё не."""
    if isinstance(val, list):
        return val
    return [val]


# ── Planner ──────────────────────────────────────────────────────────

class Planner:
    """Building Execution Graph — мерджит по (OpID + Input).

    I3: Planner не знает алгоритмы. Знает только:
      - OpID (строка)
      - Inputs (список Series)
      - Params (словарь)

    Слияние:
      - Операция регистрирует merge функцию через register_merge().
      - Если merge функция есть — Planner вызывает её при совпадении (op_id, inputs).
      - Если merge функции нет — каждый вызов register() создаёт отдельный Job.

    Planner НЕ знает:
      - что такое SMA, EMA, window, alpha.
      - как считать скользящую среднюю.
      - что params значат.
    """

    def __init__(self):
        self._pending: Dict[tuple, Job] = {}
        self._merge_fns: Dict[str, callable] = {}

    def register_merge(self, op_id: str, merge_fn) -> None:
        """Зарегистрировать merge функцию для операции.

        Args:
            op_id: идентификатор операции.
            merge_fn: fn(existing_params, new_params) → merged_params.
        """
        self._merge_fns[op_id] = merge_fn

    def register(
        self,
        op_id: str,
        inputs: List[NumericSeries],
        params: Dict[str, Any],
    ) -> Job:
        """Зарегистрировать операцию для планирования.

        Если для op_id есть merge функция — вызовы с одним (op_id, inputs)
        сливаются в один Job. Без merge функции — каждый вызов отдельный Job.

        Args:
            op_id: идентификатор операции.
            inputs: входные ряды.
            params: параметры.

        Returns:
            Job (может быть слит с другими).
        """
        input_key = tuple(id(s) for s in inputs)
        key = (op_id, input_key)

        if key in self._pending and op_id in self._merge_fns:
            # Merge через функцию операции
            existing = self._pending[key]
            existing.params = self._merge_fns[op_id](existing.params, params)
            return existing
        elif key in self._pending:
            # Нет merge функции — создаём отдельную запись с уникальным ключом
            counter = 1
            while True:
                unique_key = (op_id, input_key, counter)
                if unique_key not in self._pending:
                    break
                counter += 1
            key = unique_key

        job = Job(op_id=op_id, inputs=inputs, params=dict(params))
        self._pending[key] = job
        return job

    def build(self) -> ExecutionGraph:
        """Собрать ExecutionGraph из зарегистрированных операций."""
        graph = ExecutionGraph()
        for job in self._pending.values():
            graph.add(job)
        self._pending.clear()
        return graph

    def reset(self) -> None:
        """Очистить очередь (для тестов)."""
        self._pending.clear()


# ── Kernel (CPU) ────────────────────────────────────────────────────

class CpuKernel:
    """Исполняет Job через Driver (CPU или GPU).
    
    Если driver — WebGpuDriver, пытается диспатчить на GPU.
    Fallback на CPU loop если GPU недоступен или операция не портирована.

    Kernel знает алгоритмы (SMA, EMA, RSI).
    Driver не знает операции (I4).
    """

    def __init__(self, driver=None):
        from numfast.core.driver import CpuDriver
        self.driver = driver or CpuDriver()
        self._wgsl_cache = {}
        # Реестр операций: op_id → функция
        self._ops = {
            "sma": self._run_sma,
            "rolling_min": self._run_rolling_min,
            "rolling_max": self._run_rolling_max,
            "rolling_stddev": self._run_rolling_stddev,
            "roc": self._run_roc,
            "atr": self._run_atr,
            "ema": self._run_ema,
            "wma": self._run_wma,
            "trange": self._run_trange,
            "rsi": self._run_rsi,
            "stoch": self._run_stoch,
            "median": self._run_median,
            "obv": self._run_obv,
            "vwap": self._run_vwap,
            "cci": self._run_cci,
            "pointwise": self._run_pointwise,
        }

    def run(self, job: Job) -> None:
        """Исполнить один Job.

        Args:
            job: Job с op_id, inputs, params, outputs (заполняются здесь).
        """
        fn = self._ops.get(job.op_id)
        if fn is None:
            raise ValueError(f"Unknown op_id: {job.op_id}")
        fn(job)

    def run_graph(self, graph: ExecutionGraph) -> None:
        """Исполнить все Job в ExecutionGraph."""
        for job in graph.jobs:
            self.run(job)

    def _gpu_available(self) -> bool:
        """GPU доступен для dispatch? (WebGpuDriver)."""
        return hasattr(self.driver, 'dispatch') and self.driver.name == "webgpu"

    def _dispatch_sma_gpu(self, job, w: int) -> bool:
        """Dispatch SMA to GPU. Returns True if successful."""
        if not self._gpu_available():
            return False

        inp = job.inputs[0]
        out = job.outputs[0]
        n = inp.n

        # Generate or retrieve WGSL
        shader_key = f"sma_{w}"
        if shader_key not in self._wgsl_cache:
            from numfast.core.gpu.shaders import sma_wgsl
            self._wgsl_cache[shader_key] = sma_wgsl(w)
        wgsl = self._wgsl_cache[shader_key]

        # Get GPU buffers
        inp_key = getattr(inp, '_gpu_buf_key', None)
        out_key = getattr(out, '_gpu_buf_key', None)
        if out_key is None:
            return False
        if inp_key is None:
            inp_key = self.driver.attach_buffer(inp)
            self._gpu_upload(inp, inp_key)

        inp_buf = self.driver._buffers[inp_key]['gpu']
        out_buf = self.driver._buffers[out_key]['gpu']

        # Upload CPU shadow → GPU buffer (batch)
        self._gpu_upload(inp, inp_key)

        # Dispatch
        workgroups = ((n + 255) // 256, 1, 1)
        self.driver.dispatch(wgsl, {0: inp_buf, 1: out_buf}, workgroups)

        # Read back: copy GPU → CPU shadow (batch)
        self._gpu_download(out, out_key)

        return True

    def _dispatch_roc_gpu(self, job, period: int) -> bool:
        """Dispatch ROC to GPU. Returns True if successful."""
        if not self._gpu_available():
            return False

        inp = job.inputs[0]
        out = job.outputs[0]
        n = inp.n

        shader_key = f"roc_{period}"
        if shader_key not in self._wgsl_cache:
            from numfast.core.gpu.shaders import roc_wgsl
            self._wgsl_cache[shader_key] = roc_wgsl(period)
        wgsl = self._wgsl_cache[shader_key]

        from numfast.core.gpu.device import get_device
        device = get_device()
        import numpy as np

        inp_key = getattr(inp, '_gpu_buf_key', None)
        out_key = getattr(out, '_gpu_buf_key', None)
        if out_key is None:
            return False
        if inp_key is None:
            inp_key = self.driver.attach_buffer(inp)
            self._gpu_upload(inp, inp_key)

        inp_buf = self.driver._buffers[inp_key]['gpu']
        out_buf = self.driver._buffers[out_key]['gpu']

        # Upload CPU → GPU (batch)
        self._gpu_upload(inp, inp_key)

        workgroups = ((n + 255) // 256, 1, 1)
        self.driver.dispatch(wgsl, {0: inp_buf, 1: out_buf}, workgroups)

        # Readback GPU → CPU (batch)
        self._gpu_download(out, out_key)

        return True

    def _dispatch_sliding_gpu(self, job, w: int, op: str) -> bool:
        """Dispatch sliding window (MIN/MAX) to GPU."""
        if not self._gpu_available():
            return False

        inp = job.inputs[0]
        out = job.outputs[0]
        n = inp.n

        shader_key = f"sliding_{op}_{w}"
        if shader_key not in self._wgsl_cache:
            from numfast.core.gpu.shaders import sliding_wgsl
            self._wgsl_cache[shader_key] = sliding_wgsl(op, w)
        wgsl = self._wgsl_cache[shader_key]

        from numfast.core.gpu.device import get_device
        import numpy as np

        device = get_device()
        inp_key = getattr(inp, '_gpu_buf_key', None)
        out_key = getattr(out, '_gpu_buf_key', None)
        if out_key is None:
            return False
        if inp_key is None:
            inp_key = self.driver.attach_buffer(inp)
            self._gpu_upload(inp, inp_key)

        inp_buf = self.driver._buffers[inp_key]['gpu']
        out_buf = self.driver._buffers[out_key]['gpu']

        self._gpu_upload(inp, inp_key)

        workgroups = ((n + 255) // 256, 1, 1)
        self.driver.dispatch(wgsl, {0: inp_buf, 1: out_buf}, workgroups)

        self._gpu_download(out, out_key)

        return True

    def _dispatch_pointwise_gpu(self, job, op: str) -> bool:
        """Dispatch pointwise operation (add/sub/mul/div) to GPU."""
        if not self._gpu_available():
            return False

        inp_a = job.inputs[0]
        inp_b = job.inputs[1]
        out = job.outputs[0]
        n = inp_a.n

        shader_key = f"pointwise_{op}"
        if shader_key not in self._wgsl_cache:
            from numfast.core.gpu.shaders import pointwise_wgsl
            self._wgsl_cache[shader_key] = pointwise_wgsl(op)
        wgsl = self._wgsl_cache[shader_key]

        from numfast.core.gpu.device import get_device
        import numpy as np

        device = get_device()
        a_key = getattr(inp_a, '_gpu_buf_key', None)
        b_key = getattr(inp_b, '_gpu_buf_key', None)
        out_key = getattr(out, '_gpu_buf_key', None)
        if out_key is None:
            return False
        if a_key is None:
            a_key = self.driver.attach_buffer(inp_a)
            self._gpu_upload(inp_a, a_key)
        if b_key is None:
            b_key = self.driver.attach_buffer(inp_b)
            self._gpu_upload(inp_b, b_key)

        a_buf = self.driver._buffers[a_key]['gpu']
        b_buf = self.driver._buffers[b_key]['gpu']
        out_buf = self.driver._buffers[out_key]['gpu']

        self._gpu_upload(inp_a, a_key)
        self._gpu_upload(inp_b, b_key)

        workgroups = ((n + 255) // 256, 1, 1)
        self.driver.dispatch(wgsl, {0: a_buf, 1: b_buf, 2: out_buf}, workgroups)

        self._gpu_download(out, out_key)

        return True

    def _gpu_upload(self, series, buf_key) -> None:
        """Batch upload series data to GPU buffer (fast path).
        
        series._values is numpy int32 array — use directly.
        """
        self.driver.upload_batch(buf_key, series._values.tobytes())

    def _gpu_download(self, series, buf_key) -> None:
        """Batch download GPU buffer to series (fast path)."""
        data = self.driver.read_batch(buf_key)
        import numpy as np
        arr = np.frombuffer(data, dtype=np.int32)
        series._values[:] = arr

    # ── Pointwise ─────────────────────────────────────────

    def _run_pointwise(self, job: Job) -> None:
        """Pointwise operation (add/sub/mul/div)."""
        inp_a = job.inputs[0]
        inp_b = job.inputs[1]
        op = job.params.get("op", "add")
        n = inp_a.n
        input_valid = max(inp_a.valid_from, inp_b.valid_from)

        out = self.driver.allocate(n, meta=inp_a.meta, valid_from=input_valid)
        job.outputs.append(out)

        if self._dispatch_pointwise_gpu(job, op):
            return

        op_map = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a // b,
        }
        fn = op_map.get(op)
        if fn is None:
            raise ValueError(f"Unknown pointwise op: {op}")
        for i in range(n):
            val = fn(inp_a.read(i), inp_b.read(i))
            self.driver.write(out, i, val)

    # ── SMA ────────────────────────────────────────────────

    def _run_sma(self, job: Job) -> None:
        """Simple Moving Average."""
        inp = job.inputs[0]
        windows = job.params.get("w", [14])
        if not isinstance(windows, list):
            windows = [windows]

        n = inp.n
        input_valid = inp.valid_from

        # Create output series (same as before)
        for w in windows:
            valid = input_valid + w - 1 if w > 0 else input_valid
            out = self.driver.allocate(n, meta=inp.meta, valid_from=valid)
            job.outputs.append(out)

        # Try GPU dispatch for single-window case
        if len(windows) == 1 and self._dispatch_sma_gpu(job, windows[0]):
            return  # GPU handled it

        # CPU fallback (same as before)
        for wi, w in enumerate(windows):
            out = job.outputs[wi]
            partial_end = min(input_valid + w - 1, n)
            for i in range(input_valid, partial_end):
                self.driver.write(out, i, 0)

            if w > n:
                continue

            running = 0
            for i in range(n):
                running += self.driver.read(inp, i)
                if i >= w:
                    running -= self.driver.read(inp, i - w)
                if i >= input_valid + w - 1:
                    val = running // w
                    self.driver.write(out, i, val)

    # ── Rolling MIN/MAX ───────────────────────────────────

    def _run_rolling_min(self, job: Job) -> None:
        """Rolling Minimum — скользящее окно без prefix sum."""
        inp = job.inputs[0]
        windows = _ensure_list(job.params.get("w", [14]))
        n = inp.n
        input_valid = inp.valid_from

        for w in windows:
            valid = input_valid + w - 1 if w > 0 else input_valid
            out = self.driver.allocate(n, meta=inp.meta, valid_from=valid)
            job.outputs.append(out)

        if len(windows) == 1 and self._dispatch_sliding_gpu(job, windows[0], "min"):
            return

        for wi, w in enumerate(windows):
            out = job.outputs[wi]
            for i in range(input_valid, min(input_valid + w - 1, n)):
                self.driver.write(out, i, 0)
            if w > n or w <= 0:
                continue
            for i in range(input_valid + w - 1, n):
                win_min = min(
                    self.driver.read(inp, j)
                    for j in range(i - w + 1, i + 1)
                )
                self.driver.write(out, i, win_min)

    def _run_rolling_max(self, job: Job) -> None:
        """Rolling Maximum — скользящее окно без prefix sum."""
        inp = job.inputs[0]
        windows = _ensure_list(job.params.get("w", [14]))
        n = inp.n
        input_valid = inp.valid_from

        for w in windows:
            valid = input_valid + w - 1 if w > 0 else input_valid
            out = self.driver.allocate(n, meta=inp.meta, valid_from=valid)
            job.outputs.append(out)

        if len(windows) == 1 and self._dispatch_sliding_gpu(job, windows[0], "max"):
            return

        for wi, w in enumerate(windows):
            out = job.outputs[wi]
            for i in range(input_valid, min(input_valid + w - 1, n)):
                self.driver.write(out, i, 0)
            if w > n or w <= 0:
                continue
            for i in range(input_valid + w - 1, n):
                win_max = max(
                    self.driver.read(inp, j)
                    for j in range(i - w + 1, i + 1)
                )
                self.driver.write(out, i, win_max)

    # ── Rolling STDDEV ─────────────────────────────────

    def _run_rolling_stddev(self, job: Job) -> None:
        """Rolling Standard Deviation.

        Формула: σ = sqrt( (w·Σx² - (Σx)²) / w² )
        Использует int64 для аккумуляторов, int32 для результата.

        Args:
            job:
                inputs[0]: NumericSeries.
                params['w']: int или list[int] — окна.
                outputs: [NumericSeries...] — по одному на окно.
        """
        import math
        inp = job.inputs[0]
        windows = _ensure_list(job.params.get("w", [14]))
        n = inp.n
        input_valid = inp.valid_from

        for w in windows:
            valid = input_valid + w - 1 if w > 0 else input_valid
            out = self.driver.allocate(n, meta=inp.meta, valid_from=valid)
            job.outputs.append(out)

            for i in range(input_valid, min(input_valid + w - 1, n)):
                self.driver.write(out, i, 0)

            if w > n or w <= 0:
                continue

            w_64 = int(w)
            for i in range(input_valid + w - 1, n):
                win_sum = 0
                win_sum2 = 0
                for j in range(i - w + 1, i + 1):
                    val = self.driver.read(inp, j)
                    win_sum += val
                    win_sum2 += val * val

                # population variance: (w·sum2 - sum²) / w²
                numerator = w_64 * win_sum2 - win_sum * win_sum
                if numerator < 0:
                    numerator = 0  # rounding correction
                variance = numerator // (w_64 * w_64)
                stddev = math.isqrt(variance)
                self.driver.write(out, i, stddev)

    # ── ROC (Rate of Change) ───────────────────────────

    ROC_SCALE = 100_000  # 5 десятичных знаков для процентов

    def _run_roc(self, job: Job) -> None:
        """Rate of Change.

        roc(i) = (curr - prev) * SCALE / prev

        Storage Invariant: prev ≥ 1 → не нужен guard на деление.

        Args:
            job:
                inputs[0]: NumericSeries.
                params['period']: int или list[int] — периоды.
                outputs: [NumericSeries...] — по одному на период.
        """
        inp = job.inputs[0]
        periods = _ensure_list(job.params.get("period", [1]))
        n = inp.n
        input_valid = inp.valid_from
        SCALE = self.ROC_SCALE

        for period in periods:
            valid = input_valid + period if period > 0 else input_valid
            # ROC — безразмерная величина, offset=0
            out = self.driver.allocate(n, offset=0, scale=SCALE, valid_from=valid)
            job.outputs.append(out)

        # Try GPU for single-period case
        if len(periods) == 1 and self._dispatch_roc_gpu(job, periods[0]):
            return

        # CPU fallback (same formula as before)
        for pi, period in enumerate(periods):
            out = job.outputs[pi]
            for i in range(input_valid, min(input_valid + period, n)):
                self.driver.write(out, i, 0)

            if period > n or period <= 0:
                continue

            for i in range(input_valid + period, n):
                prev = self.driver.read(inp, i - period)
                curr = self.driver.read(inp, i)
                # prev ≥ 1 (Storage Invariant) → division safe
                # Truncation toward zero (matches WGSL / semantics)
                diff = curr - prev
                if diff >= 0:
                    roc = diff * SCALE // prev
                else:
                    roc = -((-diff) * SCALE // prev)
                self.driver.write(out, i, roc)

    # ── Registry ───────────────────────────────────────────

    # ── ATR (Average True Range) ────────────────────────

    def _run_atr(self, job: Job) -> None:
        """Average True Range.

        Первая операция с несколькими входными сериями.
        Шаги:
          1. True Range = max(H-L, |H-prevC|, |L-prevC|)
          2. SMA(TR, w) для каждого окна

        Args:
            job:
                inputs[0]: High.
                inputs[1]: Low.
                inputs[2]: Close.
                params['w']: int или list[int] — окна.
                outputs: [NumericSeries...] — по одному на окно.
        """
        high = job.inputs[0]
        low = job.inputs[1]
        close = job.inputs[2]
        windows = _ensure_list(job.params.get("w", [14]))
        n = high.n
        input_valid = max(high.valid_from, low.valid_from, close.valid_from)

        # 1. True Range (internal, не экспортируется)
        tr = self.driver.allocate(n, meta=high.meta, valid_from=input_valid)
        # TR[0] = H[0] - L[0] (без prev close)
        tr0 = max(high.read(0) - low.read(0), 0)
        self.driver.write(tr, 0, tr0)
        for i in range(1, n):
            hl = high.read(i) - low.read(i)
            hc = abs(high.read(i) - close.read(i - 1))
            lc = abs(low.read(i) - close.read(i - 1))
            tr_i = max(hl, hc, lc)
            self.driver.write(tr, i, tr_i)

        # 2. SMA на TR
        for w in windows:
            valid = input_valid + w - 1 if w > 0 else input_valid
            out = self.driver.allocate(n, meta=high.meta, valid_from=valid)
            job.outputs.append(out)

            for i in range(input_valid, min(input_valid + w - 1, n)):
                self.driver.write(out, i, 0)

            if w > n or w <= 0:
                continue

            running = 0
            for i in range(n):
                running += self.driver.read(tr, i)
                if i >= w:
                    running -= self.driver.read(tr, i - w)
                if i >= input_valid + w - 1:
                    val = running // w
                    self.driver.write(out, i, val)

    # ── EMA (Exponential Moving Average) ────────────────

    EMA_SCALE = 1_000_000  # точность alpha

    def _run_ema(self, job: Job) -> None:
        """Exponential Moving Average — рекурсия.

        EMA[0] = close[0]
        EMA[i] = α·close[i] + (1-α)·EMA[i-1]

        где α = 2/(w+1), выраженное через EMA_SCALE.

        Args:
            job:
                inputs[0]: NumericSeries (close).
                params['w']: int или list[int] — окна (периоды EMA).
                outputs: [NumericSeries...] — по одному на окно.
        """
        inp = job.inputs[0]
        windows = _ensure_list(job.params.get("w", [14]))
        n = inp.n
        input_valid = inp.valid_from
        SCALE = self.EMA_SCALE

        for w in windows:
            alpha = int(2 * SCALE / (w + 1))
            # EMA[0] = first close — валидна с первого элемента
            out = self.driver.allocate(n, meta=inp.meta, valid_from=input_valid)
            job.outputs.append(out)

            if n == 0:
                continue

            ema = self.driver.read(inp, 0)
            self.driver.write(out, 0, ema)

            for i in range(1, n):
                val = self.driver.read(inp, i)
                # α·val + (1-α)·prev
                ema = (alpha * val + (SCALE - alpha) * ema) // SCALE
                self.driver.write(out, i, ema)

    # ── WMA (Weighted Moving Average) ──────────────────────

    def _run_wma(self, job: Job) -> None:
        """Weighted Moving Average — convolution с линейными весами.

        WMA[i] = Σ(S[i-w+1+j] * (w-j)) / (w*(w+1)/2)

        Args:
            job:
                inputs[0]: NumericSeries.
                params['w']: int или list[int] — окна.
                outputs: [NumericSeries...].
        """
        inp = job.inputs[0]
        windows = _ensure_list(job.params.get("w", [14]))
        n = inp.n
        input_valid = inp.valid_from

        for w in windows:
            valid = input_valid + w - 1 if w > 0 else input_valid
            out = self.driver.allocate(n, meta=inp.meta, valid_from=valid)
            job.outputs.append(out)

            if w > n or w <= 0:
                continue

            weight_sum = w * (w + 1) // 2
            for i in range(input_valid + w - 1, n):
                total = 0
                for j in range(w):
                    idx = i - w + 1 + j
                    val = self.driver.read(inp, idx)
                    total += val * (w - j)
                wma = total // weight_sum
                self.driver.write(out, i, wma)

    # ── TRANGE (True Range) ────────────────────────────────

    def _run_trange(self, job: Job) -> None:
        """True Range — отдельная операция (ранее внутри ATR).

        TR[i] = max(H[i]-L[i], |H[i]-C[i-1]|, |L[i]-C[i-1]|)
        TR[0] = H[0] - L[0]

        Args:
            job:
                inputs[0]: High.
                inputs[1]: Low.
                inputs[2]: Close.
                params: {} — без параметров окна.
                outputs: [NumericSeries].
        """
        high = job.inputs[0]
        low = job.inputs[1]
        close = job.inputs[2]
        n = high.n
        input_valid = max(high.valid_from, low.valid_from, close.valid_from)

        out = self.driver.allocate(n, meta=high.meta, valid_from=input_valid)
        job.outputs.append(out)

        tr0 = max(high.read(0) - low.read(0), 0)
        self.driver.write(out, 0, tr0)
        for i in range(1, n):
            hl = high.read(i) - low.read(i)
            hc = abs(high.read(i) - close.read(i - 1))
            lc = abs(low.read(i) - close.read(i - 1))
            tr_i = max(hl, hc, lc)
            self.driver.write(out, i, tr_i)

    # ── RSI (Relative Strength Index) ──────────────────────

    RSI_SCALE = 1000  # 0..100000 = 0.0..100.0

    def _run_rsi(self, job: Job) -> None:
        """Relative Strength Index.

        gain = max(C[i]-C[i-1], 0)
        loss = max(C[i-1]-C[i], 0)
        avg_gain = EMA-like smooth of gain
        avg_loss = EMA-like smooth of loss
        RSI[i] = 100 * avg_gain / (avg_gain + avg_loss)

        Использует Wilders' smoothing: α = 1/w (в EMA_SCALE).
        """
        inp = job.inputs[0]
        windows = _ensure_list(job.params.get("w", [14]))
        n = inp.n
        input_valid = inp.valid_from
        ESCALE = self.EMA_SCALE  # 1_000_000

        for w in windows:
            alpha = int(ESCALE // w)  # α = 1/w (Wilders')
            valid = input_valid  # RSI валидна с первого gain/loss
            out = self.driver.allocate(n, meta=inp.meta, valid_from=valid)
            job.outputs.append(out)

            if n < 2:
                continue

            # gain[i] = max(close[i] - close[i-1], 0)
            # loss[i] = max(close[i-1] - close[i], 0)
            gain = self.driver.read(inp, 1) - self.driver.read(inp, 0)
            loss = self.driver.read(inp, 0) - self.driver.read(inp, 1)
            avg_gain = max(gain, 0)
            avg_loss = max(loss, 0)

            rsi = self._rsi_value(avg_gain, avg_loss)
            self.driver.write(out, 0, rsi)

            for i in range(1, n):
                diff = self.driver.read(inp, i) - self.driver.read(inp, i - 1)
                g = max(diff, 0)
                l_val = max(-diff, 0)
                # EMA smooth: avg = (α·val + (ESCALE-α)·prev) // ESCALE
                avg_gain = (alpha * g + (ESCALE - alpha) * avg_gain) // ESCALE
                avg_loss = (alpha * l_val + (ESCALE - alpha) * avg_loss) // ESCALE
                rsi = self._rsi_value(avg_gain, avg_loss)
                self.driver.write(out, i, rsi)

    def _rsi_value(self, avg_gain: int, avg_loss: int) -> int:
        """Вычислить RSI из avg_gain/avg_loss (raw)."""
        denom = avg_gain + avg_loss
        if denom == 0:
            return 50 * self.RSI_SCALE  # нейтральное
        # RSI = 100 * RSI_SCALE * avg_gain / denom
        return (100 * self.RSI_SCALE * avg_gain) // denom

    # ── STOCH (Stochastic Oscillator) ──────────────────────

    STOCH_SCALE = 1000  # 0..100000 = 0.0..100.0

    def _run_stoch(self, job: Job) -> None:
        """Stochastic Oscillator %K.

        %K = 100 * (close - L(w)) / (H(w) - L(w))

        Args:
            job:
                inputs[0]: High.
                inputs[1]: Low.
                inputs[2]: Close.
                params['w']: int или list[int] — окна.
                outputs: [NumericSeries...].
        """
        high = job.inputs[0]
        low = job.inputs[1]
        close = job.inputs[2]
        windows = _ensure_list(job.params.get("w", [14]))
        n = high.n
        input_valid = max(high.valid_from, low.valid_from, close.valid_from)
        SCALE = self.STOCH_SCALE

        for w in windows:
            valid = input_valid + w - 1 if w > 0 else input_valid
            out = self.driver.allocate(n, meta=close.meta, valid_from=valid)
            job.outputs.append(out)

            if w > n or w <= 0:
                continue

            for i in range(input_valid + w - 1, n):
                hh = self.driver.read(high, i)
                ll = self.driver.read(low, i)
                for j in range(i - w + 1, i + 1):
                    hv = self.driver.read(high, j)
                    lv = self.driver.read(low, j)
                    if hv > hh:
                        hh = hv
                    if lv < ll:
                        ll = lv

                c = self.driver.read(close, i)
                denom = hh - ll
                if denom > 0:
                    k = 100 * SCALE * (c - ll) // denom
                    k = max(0, min(100 * SCALE, k))  # clamp to [0, 100]
                else:
                    k = 50 * SCALE
                self.driver.write(out, i, k)

    # ── MEDIAN (Rolling Median) ────────────────────────────

    def _run_median(self, job: Job) -> None:
        """Rolling Median — сортировка окна.

        Args:
            job:
                inputs[0]: NumericSeries.
                params['w']: int или list[int] — окна.
                outputs: [NumericSeries...].
        """
        inp = job.inputs[0]
        windows = _ensure_list(job.params.get("w", [14]))
        n = inp.n
        input_valid = inp.valid_from

        for w in windows:
            valid = input_valid + w - 1 if w > 0 else input_valid
            out = self.driver.allocate(n, meta=inp.meta, valid_from=valid)
            job.outputs.append(out)

            if w <= 0:
                continue

            for i in range(input_valid + w - 1, n):
                window = []
                for j in range(i - w + 1, i + 1):
                    window.append(self.driver.read(inp, j))
                window.sort()
                median = window[w // 2]
                self.driver.write(out, i, median)

    # ── OBV (On-Balance Volume) ────────────────────────────

    def _run_obv(self, job: Job) -> None:
        """On-Balance Volume.

        OBV[i] = OBV[i-1] + (close[i] > close[i-1] ? vol : close[i] < close[i-1] ? -vol : 0)

        Args:
            job:
                inputs[0]: Close.
                inputs[1]: Volume.
                outputs: [NumericSeries].
        """
        close = job.inputs[0]
        volume = job.inputs[1]
        n = close.n
        input_valid = max(close.valid_from, volume.valid_from)

        out = self.driver.allocate(n, meta=close.meta, valid_from=input_valid)
        job.outputs.append(out)

        if n == 0:
            return

        obv = 0
        self.driver.write(out, 0, obv)
        for i in range(1, n):
            vol = max(volume.read(i), 1)  # Storage Invariant
            if close.read(i) > close.read(i - 1):
                obv += vol
            elif close.read(i) < close.read(i - 1):
                obv -= vol
            self.driver.write(out, i, obv)

    # ── VWAP (Volume Weighted Average Price) ──────────────

    def _run_vwap(self, job: Job) -> None:
        """Volume Weighted Average Price (cumulative).

        VWAP[i] = Σ(close[j]*vol[j]) / Σ(vol[j])

        Args:
            job:
                inputs[0]: Close.
                inputs[1]: Volume.
                outputs: [NumericSeries].
        """
        close = job.inputs[0]
        volume = job.inputs[1]
        n = close.n
        input_valid = max(close.valid_from, volume.valid_from)

        out = self.driver.allocate(n, meta=close.meta, valid_from=input_valid)
        job.outputs.append(out)

        cum_pv = 0
        cum_vol = 0
        for i in range(n):
            pv = close.read(i) * max(volume.read(i), 1)
            cum_pv += pv
            cum_vol += max(volume.read(i), 1)
            vwap = cum_pv // cum_vol if cum_vol > 0 else 1
            self.driver.write(out, i, vwap)

    # ── CCI (Commodity Channel Index) ──────────────────────

    CCI_SCALE = 1000  # CCI values: -100000..100000

    def _dispatch_cci_gpu(self, job, w: int, scale: int) -> bool:
        """Dispatch CCI to GPU. Returns True if successful."""
        if not self._gpu_available():
            return False

        high = job.inputs[0]
        low = job.inputs[1]
        close = job.inputs[2]
        out = job.outputs[0]
        n = high.n

        shader_key = f"cci_{w}_{scale}"
        if shader_key not in self._wgsl_cache:
            from numfast.core.gpu.shaders import cci_wgsl
            self._wgsl_cache[shader_key] = cci_wgsl(w, scale)
        wgsl = self._wgsl_cache[shader_key]

        from numfast.core.gpu.device import get_device
        import numpy as np

        device = get_device()
        h_key = getattr(high, '_gpu_buf_key', None)
        l_key = getattr(low, '_gpu_buf_key', None)
        c_key = getattr(close, '_gpu_buf_key', None)
        o_key = getattr(out, '_gpu_buf_key', None)
        if o_key is None:
            return False
        if h_key is None:
            h_key = self.driver.attach_buffer(high)
            self._gpu_upload(high, h_key)
        if l_key is None:
            l_key = self.driver.attach_buffer(low)
            self._gpu_upload(low, l_key)
        if c_key is None:
            c_key = self.driver.attach_buffer(close)
            self._gpu_upload(close, c_key)

        h_buf = self.driver._buffers[h_key]['gpu']
        l_buf = self.driver._buffers[l_key]['gpu']
        c_buf = self.driver._buffers[c_key]['gpu']
        o_buf = self.driver._buffers[o_key]['gpu']

        self._gpu_upload(high, h_key)
        self._gpu_upload(low, l_key)
        self._gpu_upload(close, c_key)

        workgroups = ((n + 255) // 256, 1, 1)
        self.driver.dispatch(wgsl, {0: h_buf, 1: l_buf, 2: c_buf, 3: o_buf}, workgroups)

        self._gpu_download(out, o_key)

        return True

    def _run_cci(self, job: Job) -> None:
        """Commodity Channel Index.

        TP = (H + L + C) / 3
        SMA_TP = SMA(TP, w)
        MD = mean(|TP[j] - SMA_TP[i]| for j in i-w+1..i)
        CCI = (TP[i] - SMA_TP[i]) / (0.015 * MD)

        Args:
            job:
                inputs[0]: High.
                inputs[1]: Low.
                inputs[2]: Close.
                params['w']: int or list[int] — windows.
                outputs: [NumericSeries...].
        """
        high = job.inputs[0]
        low = job.inputs[1]
        close = job.inputs[2]
        windows = _ensure_list(job.params.get("w", [14]))
        n = high.n
        input_valid = max(high.valid_from, low.valid_from, close.valid_from)
        SCALE = self.CCI_SCALE

        for w in windows:
            valid = input_valid + w - 1 if w > 0 else input_valid
            out = self.driver.allocate(n, meta=close.meta, valid_from=valid)
            job.outputs.append(out)

            if w > n or w <= 0:
                continue

            # Try GPU dispatch for single window
            if len(windows) == 1 and self._dispatch_cci_gpu(job, w, SCALE):
                continue

            for i in range(input_valid + w - 1, n):
                # Typical Price
                tp = (high.read(i) + low.read(i) + close.read(i)) // 3

                # SMA of TP
                sum_tp = 0
                for j in range(i - w + 1, i + 1):
                    tp_j = (high.read(j) + low.read(j) + close.read(j)) // 3
                    sum_tp += tp_j
                sma_tp = sum_tp // w

                # Mean Deviation
                md_sum = 0
                for j in range(i - w + 1, i + 1):
                    tp_j = (high.read(j) + low.read(j) + close.read(j)) // 3
                    md_sum += abs(tp_j - sma_tp)
                md = md_sum // w

                if md > 0:
                    # CCI = (TP - SMA_TP) / (0.015 * MD)
                    # 0.015 = 15/1000 = 3/200
                    # CCI = (TP - SMA_TP) * SCALE * 200 / (3 * MD)
                    diff = tp - sma_tp
                    cci = diff * SCALE * 200 // (3 * md)
                else:
                    cci = 0

                self.driver.write(out, i, cci)

    # ── Registry ───────────────────────────────────────────

    def register_op(self, op_id: str, fn) -> None:
        """Зарегистрировать новую операцию.

        Args:
            op_id: идентификатор.
            fn: функция(job).
        """
        self._ops[op_id] = fn


# ── API (упрощённый) ────────────────────────────────────────────────

class CoreAPI:
    """Точка входа для пользователя.

    Пример:
        api = CoreAPI()
        result = api.sma(series, windows=[14, 28])
    """

    def __init__(self, planner: Optional[Planner] = None, kernel: Optional[CpuKernel] = None):
        self.planner = planner or Planner()
        self.kernel = kernel or CpuKernel()
        # Register merge functions for known operations
        self.planner.register_merge("sma", merge_sma)
        self.planner.register_merge("rolling_min", merge_sma)
        self.planner.register_merge("rolling_max", merge_sma)
        self.planner.register_merge("rolling_stddev", merge_window)
        self.planner.register_merge("roc", merge_window)
        self.planner.register_merge("atr", merge_window)
        self.planner.register_merge("ema", merge_window)
        self.planner.register_merge("wma", merge_sma)
        self.planner.register_merge("rsi", merge_window)
        self.planner.register_merge("stoch", merge_window)
        self.planner.register_merge("median", merge_sma)
        self.planner.register_merge("cci", merge_window)

    def sma(
        self,
        series: NumericSeries,
        windows: List[int],
    ) -> List[NumericSeries]:
        """SMA: одно- или многоверсионный.

        Args:
            series: входной ряд.
            windows: список окон.

        Returns:
            list[NumericSeries]: по одному на окно.
        """
        # Register → Build (clears planner state) → Run → Return
        self.planner.register("sma", [series], {"w": windows})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        # Graph has exactly one merged job
        return graph.jobs[0].outputs

    def rolling_min(
        self,
        series: NumericSeries,
        windows: int | List[int],
    ) -> NumericSeries | List[NumericSeries]:
        """Rolling Minimum."""
        single = isinstance(windows, int)
        ws = [windows] if single else list(windows)
        self.planner.register("rolling_min", [series], {"w": ws})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        out = graph.jobs[0].outputs
        return out[0] if single else out

    def rolling_max(
        self,
        series: NumericSeries,
        windows: int | List[int],
    ) -> NumericSeries | List[NumericSeries]:
        """Rolling Maximum."""
        single = isinstance(windows, int)
        ws = [windows] if single else list(windows)
        self.planner.register("rolling_max", [series], {"w": ws})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        out = graph.jobs[0].outputs
        return out[0] if single else out

    def rolling_stddev(
        self,
        series: NumericSeries,
        windows: int | List[int],
    ) -> NumericSeries | List[NumericSeries]:
        """Rolling Standard Deviation."""
        single = isinstance(windows, int)
        ws = [windows] if single else list(windows)
        self.planner.register("rolling_stddev", [series], {"w": ws})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        out = graph.jobs[0].outputs
        return out[0] if single else out

    def roc(
        self,
        series: NumericSeries,
        periods: int | List[int],
    ) -> NumericSeries | List[NumericSeries]:
        """Rate of Change.

        Args:
            series: входной ряд.
            periods: период или список периодов.

        Returns:
            NumericSeries с roc в scale=ROC_SCALE (100000 = 100%).
        """
        single = isinstance(periods, int)
        ps = [periods] if single else list(periods)
        self.planner.register("roc", [series], {"period": ps})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        out = graph.jobs[0].outputs
        return out[0] if single else out

    def atr(
        self,
        high: NumericSeries,
        low: NumericSeries,
        close: NumericSeries,
        windows: int | List[int],
    ) -> NumericSeries | List[NumericSeries]:
        """Average True Range.

        Args:
            high: максимальные цены.
            low: минимальные цены.
            close: цены закрытия.
            windows: окно или список окон.

        Returns:
            NumericSeries (ATR) или список.
        """
        single = isinstance(windows, int)
        ws = [windows] if single else list(windows)
        self.planner.register("atr", [high, low, close], {"w": ws})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        out = graph.jobs[0].outputs
        return out[0] if single else out

    def ema(
        self,
        series: NumericSeries,
        windows: int | List[int],
    ) -> NumericSeries | List[NumericSeries]:
        """Exponential Moving Average.

        Args:
            series: входной ряд.
            windows: окно или список окон (период EMA).

        Returns:
            NumericSeries (EMA) или список.
        """
        single = isinstance(windows, int)
        ws = [windows] if single else list(windows)
        self.planner.register("ema", [series], {"w": ws})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        out = graph.jobs[0].outputs
        return out[0] if single else out

    def wma(self, series, windows):
        """Weighted Moving Average."""
        single = isinstance(windows, int)
        ws = [windows] if single else list(windows)
        self.planner.register("wma", [series], {"w": ws})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        out = graph.jobs[0].outputs
        return out[0] if single else out

    def trange(self, high, low, close):
        """True Range (без окна, отдельная операция)."""
        self.planner.register("trange", [high, low, close], {})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        return graph.jobs[0].outputs[0]

    def rsi(self, series, windows):
        """Relative Strength Index."""
        single = isinstance(windows, int)
        ws = [windows] if single else list(windows)
        self.planner.register("rsi", [series], {"w": ws})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        out = graph.jobs[0].outputs
        return out[0] if single else out

    def stoch(self, high, low, close, windows):
        """Stochastic Oscillator %K."""
        single = isinstance(windows, int)
        ws = [windows] if single else list(windows)
        self.planner.register("stoch", [high, low, close], {"w": ws})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        out = graph.jobs[0].outputs
        return out[0] if single else out

    def rolling_median(self, series, windows):
        """Rolling Median."""
        single = isinstance(windows, int)
        ws = [windows] if single else list(windows)
        self.planner.register("median", [series], {"w": ws})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        out = graph.jobs[0].outputs
        return out[0] if single else out

    def obv(self, close, volume):
        """On-Balance Volume (без окна)."""
        self.planner.register("obv", [close, volume], {})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        return graph.jobs[0].outputs[0]

    def vwap(self, close, volume):
        """Volume Weighted Average Price (без окна)."""
        self.planner.register("vwap", [close, volume], {})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        return graph.jobs[0].outputs[0]

    def cci(self, high, low, close, windows):
        """Commodity Channel Index."""
        single = isinstance(windows, int)
        ws = [windows] if single else list(windows)
        self.planner.register("cci", [high, low, close], {"w": ws})
        graph = self.planner.build()
        self.kernel.run_graph(graph)
        out = graph.jobs[0].outputs
        return out[0] if single else out
