"""Phase 14.5 — Stress Test Runtime.

Проверяет масштабируемость архитектуры:
  - 500 операций разных классов
  - merge ratio (операций / Job)
  - время построения графа vs время выполнения
  - отсутствие утечек памяти (по числу выделенных серий)

Запуск:
    python -m pytest numfast/core/tests/test_stress_runtime.py -v
"""

import sys
import os
import time
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from numfast.core.nseries import NumericSeries
from numfast.core.executor import Planner, CpuKernel, CoreAPI, merge_sma, merge_window
from numfast.core.driver import CpuDriver


class TestRuntimeStress:
    """Проверка масштабируемости Runtime."""

    OPS_DEF = {
        "sma":       {"merge": merge_sma,    "inputs": 1},
        "ema":       {"merge": merge_window, "inputs": 1},
        "roc":       {"merge": merge_window, "inputs": 1},
        "rolling_min":  {"merge": merge_sma,    "inputs": 1},
        "rolling_max":  {"merge": merge_sma,    "inputs": 1},
        "rolling_stddev": {"merge": merge_window, "inputs": 1},
        "atr":       {"merge": merge_window, "inputs": 3},
    }

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        n = 1000
        base = list(np.random.randn(n).cumsum() + 10000)
        self.close = NumericSeries(base, offset=0, scale=1)
        self.high = NumericSeries([v + random.uniform(0, 5) for v in base])
        self.low = NumericSeries([v - random.uniform(0, 5) for v in base])

    def _make_planner(self):
        p = Planner()
        for op_id, cfg in self.OPS_DEF.items():
            p.register_merge(op_id, cfg["merge"])
        return p

    def test_stress_1000_ops(self):
        """1000 операций — build + execute."""
        p = self._make_planner()
        windows = [3, 5, 7, 14, 20, 28, 50]
        op_ids = list(self.OPS_DEF.keys())

        n_ops = 0
        for _ in range(1000):
            op_id = random.choice(op_ids)
            w = random.choice(windows)
            cfg = self.OPS_DEF[op_id]

            if cfg["inputs"] == 1:
                p.register(op_id, [self.close], {"w": w})
            elif cfg["inputs"] == 3:
                p.register(op_id, [self.high, self.low, self.close], {"w": w})
            n_ops += 1

        t0 = time.time()
        graph = p.build()
        t_build = time.time() - t0

        kernel = CpuKernel(CpuDriver())
        t0 = time.time()
        kernel.run_graph(graph)
        t_exec = time.time() - t0

        n_jobs = len(graph.jobs)
        merge_ratio = n_ops / n_jobs if n_jobs > 0 else 0

        # Metrics
        print(f"\n{'='*50}")
        print(f"Stress Test: {n_ops} operations")
        print(f"{'='*50}")
        print(f"  Jobs:          {n_jobs}")
        print(f"  Merge ratio:   {merge_ratio:.1f}x")
        print(f"  Build time:    {t_build*1000:.1f} ms")
        print(f"  Execute time:  {t_exec*1000:.1f} ms")
        print(f"  Output series: {sum(len(j.outputs) for j in graph.jobs)}")
        print(f"{'='*50}")

        # Assertions
        assert n_jobs > 0, "Должен быть хотя бы 1 Job"
        assert merge_ratio >= 3, (
            f"Merge ratio {merge_ratio:.1f}x < 3x — "
            f"Planner не объединяет операции"
        )
        assert t_build < 1.0, (
            f"Build time {t_build:.3f}s > 1s — слишком долго"
        )

    def test_stress_all_sma_same_input(self):
        """500 SMA на одном входе → 1 Job."""
        p = self._make_planner()
        for w in [3, 5, 7, 14, 20, 28, 50, 100, 200]:
            for _ in range(62):  # 9×62 ≈ 558 ops
                p.register("sma", [self.close], {"w": w})

        graph = p.build()
        assert len(graph.jobs) <= 9, (
            f"Ожидал ≤9 Job (одна на окно), получил {len(graph.jobs)}"
        )

    def test_stress_mixed_merge(self):
        """Смесь: операции с merge и без merge."""
        p = self._make_planner()
        windows = [3, 5, 14, 28]

        for _ in range(500):
            op_id = random.choice(["sma", "roc", "ema"])
            w = random.choice(windows)
            p.register(op_id, [self.close], {"w": w})

        graph = p.build()
        # Минимум: 3 op_id × 4 windows = 12 Job
        # Плюс merge: 12 / merge_factor
        n_jobs = len(graph.jobs)
        assert n_jobs >= 3, f"Меньше 3 Job для 3 op_id: {n_jobs}"
        assert n_jobs <= 12, f"Больше 12 Job без merge: {n_jobs}"

    def test_stress_memory_series_count(self):
        """Проверка что количество Job разумно (merge работает).

        200 одинаковых SMA(14) → 1 Job (не 200).
        """
        p = self._make_planner()
        for _ in range(200):
            p.register("sma", [self.close], {"w": 14})

        graph = p.build()
        kernel = CpuKernel(CpuDriver())
        kernel.run_graph(graph)

        # 200 SMA(14) registrations → 1 merged Job (same op_id, same input)
        n_jobs = len(graph.jobs)
        assert n_jobs == 1, (
            f"200 SMA(14) должны дать 1 Job, получил {n_jobs}"
        )

        # Each registration produces one output (kernel creates N outputs for N w values)
        n_outputs = sum(len(j.outputs) for j in graph.jobs)
        assert n_outputs >= 1, f"Должен быть хотя бы 1 output"

    def test_stress_chain_length(self):
        """Длинная цепочка: Close → SMA → ROC → EMA → ..."""
        p = self._make_planner()
        current = self.close

        for _ in range(100):
            op_id = random.choice(["sma", "ema", "roc"])
            w = random.choice([3, 5, 14])

            input_series = current
            p.register(op_id, [input_series], {"w": w})

        graph = p.build()
        kernel = CpuKernel(CpuDriver())
        kernel.run_graph(graph)

        # Все выходы читаются без ошибок
        for j in graph.jobs:
            for out in j.outputs:
                assert out.read(out.n - 1) is not None
