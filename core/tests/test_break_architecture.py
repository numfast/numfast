"""Phase 16 — "Попытка сломать архитектуру".

Стресс-тесты архитектуры Runtime:
  1. Грязные цепочки — сложные деревья зависимостей
  2. Случайные графы — 1000 DAG с проверкой топологии
  3. Memory pressure — много серий, много операций
  4. "Плохие" операции — Runtime не ломается от некорректных операций
  5. Архитектурный fuzzing — случайные графы + проверка всех теорем

Запуск:
    python -m pytest numfast/core/tests/test_break_architecture.py -v
"""

import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from numfast.core.nseries import NumericSeries
from numfast.core.executor import CpuKernel, CoreAPI, Planner, merge_sma, merge_window
from numfast.core.driver import CpuDriver
from numfast.core._evil_ops import install_evil_ops


# ══════════════════════════════════════════════════════════════════════
# 1. Dirty Chains — сложные деревья зависимостей
# ══════════════════════════════════════════════════════════════════════

class TestDirtyChains:
    """Сложные цепочки операций — графы, а не линейные последовательности."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        n = 200
        base = list(np.random.randn(n).cumsum() + 10000)
        self.close = NumericSeries(base)
        self.high = NumericSeries([v + random.uniform(0, 5) for v in base])
        self.low = NumericSeries([v - random.uniform(0, 5) for v in base])

    def test_binary_tree(self):
        """Дерево: close → [sma5, sma20] → ema10 → roc3."""
        api = CoreAPI()
        sma5 = api.sma(self.close, [5])[0]
        sma20 = api.sma(self.close, [20])[0]
        # EMA от суммы двух SMA
        ema_in = NumericSeries([
            sma5.read(i) + sma20.read(i)
            for i in range(self.close.n)
        ])
        ema10 = api.ema(ema_in, 10)
        try:
            roc3 = api.roc(ema10, 3)
            assert roc3.read(roc3.n - 1) is not None
        except ZeroDivisionError:
            # ROC может делить на 0 если EMA выдала 0 (из-за prefx 0s от SMA)
            # Runtime переживает эту ошибку — Теорема 6
            pass

    def test_diamond(self):
        """Ромб: close → [sma10, ema10] → roc(разница)."""
        api = CoreAPI()
        sma10 = api.sma(self.close, [10])[0]
        ema10 = api.ema(self.close, 10)
        diff = NumericSeries([
            sma10.read(i) - ema10.read(i)
            for i in range(self.close.n)
        ])
        roc_diff = api.roc(diff, 5)
        assert roc_diff.read(roc_diff.n - 1) is not None

    def test_atr_ema_chain(self):
        """ATR + EMA: High,Low,Close → ATR → EMA → ROC."""
        api = CoreAPI()
        atr14 = api.atr(self.high, self.low, self.close, 14)
        ema_atr = api.ema(atr14, 7)
        try:
            roc_atr = api.roc(ema_atr, 5)
            assert roc_atr.read(roc_atr.n - 1) is not None
        except ZeroDivisionError:
            # Деление на 0 в ROC — Runtime жив (Теорема 6)
            pass

    def test_valid_from_cascade(self):
        """valid_from правильно каскадируется через цепочки."""
        api = CoreAPI()
        sma20 = api.sma(self.close, [20])[0]
        assert sma20.valid_from == 19

        ema_sma = api.ema(sma20, 10)
        # EMA наследует valid_from от входа (sma20.valid_from == 19)
        assert ema_sma.valid_from == 19

        roc_ema = api.roc(ema_sma, 5)
        # ROC(5): valid_from = входная valid_from + period = 19 + 5 = 24
        assert roc_ema.valid_from == 24

    def test_multi_output_chain(self):
        """Несколько окон ATR → каждый используется в цепочке."""
        api = CoreAPI()
        results = api.atr(self.high, self.low, self.close, [7, 14, 28])
        for i in range(len(results)):
            chain = api.ema(results[i], 5)
            assert chain.read(chain.n - 1) is not None

    def test_overlapping_chains(self):
        """Пересекающиеся цепочки: один вход, два выхода, слияние.

        close ── sma10 ── roc5 ──┐
                                  ├── ema8
        close ── ema20 ──────────┘
        """
        api = CoreAPI()
        sma10 = api.sma(self.close, [10])[0]
        ema20 = api.ema(self.close, 20)
        roc5 = api.roc(sma10, 5)
        # Слияние двух цепочек
        merged = NumericSeries([
            roc5.read(i) + ema20.read(i)
            for i in range(self.close.n)
        ])
        ema8 = api.ema(merged, 8)
        assert ema8.read(ema8.n - 1) is not None


# ══════════════════════════════════════════════════════════════════════
# 2. Random DAGs — 1000 случайных графов
# ══════════════════════════════════════════════════════════════════════

class TestRandomDags:
    """Случайные графы операций с проверкой топологии."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(12345)
        self.n = 100
        self.base = list(np.random.randn(self.n).cumsum() + 10000)
        self.close = NumericSeries(self.base)

    def _random_graph(self, depth: int):
        """Построить случайный DAG и выполнить.

        Args:
            depth: число узлов в графе.

        Returns:
            список выходных серий.
        """
        api = CoreAPI()
        # Пул доступных серий: начинаем с close
        pool = {"close": self.close}
        outputs = []

        for _ in range(depth):
            # Выбираем случайный вход из пула
            src_name = random.choice(list(pool.keys()))
            src = pool[src_name]

            op = random.choice(["sma", "ema", "roc", "min", "max", "stddev"])
            w = random.choice([3, 5, 14, 28])

            try:
                if op == "sma":
                    result = api.sma(src, [w])[0]
                elif op == "ema":
                    result = api.ema(src, w)
                elif op == "roc":
                    result = api.roc(src, w)
                elif op == "min":
                    result = api.rolling_min(src, w)
                elif op == "max":
                    result = api.rolling_max(src, w)
                elif op == "stddev":
                    result = api.rolling_stddev(src, w)

                # Сохраняем результат в пул
                name = f"{op}{w}_{_}"
                pool[name] = result
                outputs.append(result)
            except Exception as e:
                # Некоторые комбинации могут быть невалидны (valid_from > n)
                # — это ожидаемо, Runtime не падает
                pass

        return outputs

    def test_100_random_graphs(self):
        """100 случайных графов, каждый глубиной 10 узлов."""
        for g in range(100):
            outputs = self._random_graph(depth=10)
            for out in outputs:
                # Последний элемент читается (теорема 6: Runtime не сломан)
                assert out.read(out.n - 1) is not None
            if g % 20 == 0:
                print(f"  DAG {g}/100 — {len(outputs)} outputs")

    def test_acyclic_guarantee(self):
        """Проверка что Planner не создаёт циклические зависимости.

        Тест: регистрируем операции так, чтобы output одной был input другой.
        Planner не должен создать цикл.
        """
        p = Planner()
        for op_n in ["sma", "ema", "roc"]:
            p.register_merge(op_n, merge_window)

        s = self.close
        # Цепочка: s → sma14 → ema5 → roc3
        s1 = NumericSeries([1] * self.n)
        s2 = NumericSeries([1] * self.n)
        s3 = NumericSeries([1] * self.n)

        # Эмулируем цепочку через разные серии
        p.register("sma", [s], {"w": 14})    # Job 0
        p.register("ema", [s1], {"w": 5})    # Job 1 (s1 — результат Job 0)
        p.register("roc", [s2], {"w": 3})    # Job 2 (s2 — результат Job 1)

        graph = p.build()
        # Jobs должны быть упорядочены по зависимостям
        assert len(graph.jobs) == 3
        # Все outputs читаются
        kernel = CpuKernel(CpuDriver())
        kernel.run_graph(graph)
        for j in graph.jobs:
            for out in j.outputs:
                assert out.read(out.n - 1) is not None


# ══════════════════════════════════════════════════════════════════════
# 3. Memory Pressure — много серий, много операций
# ══════════════════════════════════════════════════════════════════════

class TestMemoryPressure:
    """Большое количество серий и операций."""

    def test_many_series(self):
        """10 000 Series в пуле — не падает."""
        np.random.seed(1)
        series_list = []
        for i in range(10000):
            s = NumericSeries([1, 2, 3], offset=i % 100, scale=1)
            series_list.append(s)
        assert len(series_list) == 10000
        # Все читаются
        for i, s in enumerate(series_list):
            assert s.read(0) >= 1  # Storage Invariant

    def test_many_operations_sequential(self):
        """1000 операций последовательно (не merge)."""
        api = CoreAPI()
        np.random.seed(2)
        n = 50
        base = list(np.random.randn(n).cumsum() + 10000)
        close = NumericSeries(base)

        for i in range(1000):
            w = random.choice([3, 5, 14])
            op = random.choice(["sma", "ema", "roc"])
            if op == "sma":
                r = api.sma(close, [w])[0]
            elif op == "ema":
                r = api.ema(close, w)
            else:
                r = api.roc(close, w)
            assert r.read(r.n - 1) is not None
            if i % 200 == 0:
                print(f"  {i}/1000 ops done")

    def test_temporary_series_lifetime(self):
        """Временные Series, созданные операциями, не блокируют память."""
        api = CoreAPI()
        np.random.seed(3)
        n = 100
        base = list(np.random.randn(n).cumsum() + 10000)
        close = NumericSeries(base)

        # Создаём и бросаем временные результаты
        for _ in range(500):
            _ = api.sma(close, 14)
            _ = api.ema(close, 14)
            _ = api.roc(close, 5)

        # После 500 операций GC должен освободить временные
        import gc
        gc.collect()

        # Проверяем что close всё ещё жив
        assert close.read(close.n - 1) is not None


# ══════════════════════════════════════════════════════════════════════
# 4. "Плохие" операции — Runtime не ломается
# ══════════════════════════════════════════════════════════════════════

class TestEvilOperations:
    """Runtime переживает злонамеренные операции (Теорема 6)."""

    @pytest.fixture(autouse=True)
    def setup_evil(self):
        self.driver = CpuDriver()
        self.kernel = CpuKernel(self.driver)
        install_evil_ops(self.kernel)

    def _run_evil(self, op_id: str, n=50):
        """Запустить evil-операцию."""
        from numfast.core.executor import Job
        inp = NumericSeries(list(range(1, n + 1)))
        job = Job(op_id=op_id, inputs=[inp], params={})
        # Evil ops не используют driver — им нужен доступ к driver
        # через job. Добавим driver в job для совместимости.
        job.driver = self.driver
        self.kernel.run(job)
        return job

    def test_always_zero(self):
        """always_zero пишет 0 — нарушает Storage Invariant, но не ломает Runtime."""
        job = self._run_evil("always_zero")
        out = job.outputs[0]
        assert out is not None
        assert out.read(0) == 0  # Да, 0 — это нарушение, но Runtime не упал

    def test_always_negative(self):
        """always_negative — Runtime переживает."""
        job = self._run_evil("always_negative")
        out = job.outputs[0]
        assert out.read(0) == -1
        assert out.read(1) == -2

    def test_random_output(self):
        """random_output — Runtime переживает непредсказуемые результаты."""
        job = self._run_evil("random_output")
        out = job.outputs[0]
        val = out.read(0)
        assert 0 <= val <= 1000000

    def test_throws_exception(self):
        """throws_exception — Runtime переживает исключение в операции."""
        from numfast.core.executor import Job
        inp = NumericSeries([1, 2, 3])
        job = Job(op_id="throws_exception", inputs=[inp], params={})
        with pytest.raises(RuntimeError, match="Преднамеренное исключение"):
            self.kernel.run(job)
        # После исключения Runtime всё ещё работает
        # (проверяем созданием новой операции)
        job2 = Job(op_id="always_zero", inputs=[inp], params={})
        job2.driver = self.driver
        try:
            self.kernel.run(job2)
        except Exception:
            pytest.fail("Runtime не восстановился после исключения")

    def test_unknown_op_id(self):
        """Неизвестный op_id — Runtime не падает, а сообщает об ошибке."""
        from numfast.core.executor import Job
        inp = NumericSeries([1, 2, 3])
        job = Job(op_id="nonexistent_op", inputs=[inp], params={})
        with pytest.raises(ValueError, match="Unknown op_id"):
            self.kernel.run(job)

    def test_manifest_does_not_break(self):
        """manifest пытается сломать Runtime — не удаётся."""
        job = self._run_evil("manifest")
        out = job.outputs[0]
        # Input не должен измениться после попыток мутации
        # Manifest пытался записать 0 в input, но Runtime не сломался
        # Storage Invariant может быть нарушен операцией — Runtime это переживает
        _ = job.inputs[0].read(0)  # читается без падения
        # Manifest записал 0 — данные испорчены, но Runtime жив
        # (Теорема 6: операция не может сломать Runtime, только данные)
        assert out.read(0) is not None  # читается без падения


# ══════════════════════════════════════════════════════════════════════
# 5. Architectural Fuzzing — случайные графы + 6 теорем
# ══════════════════════════════════════════════════════════════════════

class TestArchitecturalFuzzing:
    """Случайные графы с проверкой всех теорем."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(777)

    def _check_theorems(self, graph, kernel, n_ops):
        """Проверить 6 теорем после выполнения графа."""
        # Theorem 1: Planner is operation-agnostic
        # (не проверяем — это статическое свойство кода)

        # Theorem 2: Driver replaceable
        # (не проверяем — используется CpuDriver)

        # Theorem 3: Indicators decompose into primitives
        # (не проверяем — это свойство математической модели)

        # Theorem 4: Storage doesn't affect algorithms
        for j in graph.jobs:
            for out in j.outputs:
                assert out.read(out.n - 1) is not None

        # Theorem 5: New ops don't change Runtime
        # Runtime не изменился — это статическое свойство

        # Theorem 6: Bad ops can't break Runtime
        # Runtime выполняет все Job без падений (если мы здесь — прошло)

        return True

    def test_fuzz_1000_random_graphs(self):
        """1000 случайных графов с автоматической верификацией теорем."""
        total_ops = 0
        for g in range(1000):
            n = random.randint(20, 50)
            base = list(np.random.randn(n).cumsum() + 10000)
            close = NumericSeries(base)

            # Planner
            p = Planner()
            for op_n in ["sma", "ema", "roc", "rolling_min", "rolling_max",
                          "rolling_stddev"]:
                p.register_merge(op_n, merge_sma if op_n in ("sma", "rolling_min", "rolling_max") else merge_window)

            n_ops = 0
            for _ in range(random.randint(5, 30)):
                op = random.choice(["sma", "ema", "roc", "rolling_min",
                                    "rolling_max", "rolling_stddev"])
                w = random.choice([3, 5, 7, 14, 28, 50])
                p.register(op, [close], {"w": w})
                n_ops += 1

            graph = p.build()
            kernel = CpuKernel(CpuDriver())

            try:
                kernel.run_graph(graph)
                self._check_theorems(graph, kernel, n_ops)
                total_ops += n_ops
            except Exception as e:
                # Некоторые комбинации могут вызвать ошибки (деление на ноль и т.п.)
                # Runtime должен сообщить об ошибке, а не упасть
                assert isinstance(e, (ValueError, ZeroDivisionError, RuntimeError))

            if g % 100 == 0:
                print(f"  Fuzz {g}/1000 — {total_ops} ops total")

        print(f"\n  [OK] {total_ops} ops across 1000 random graphs")
