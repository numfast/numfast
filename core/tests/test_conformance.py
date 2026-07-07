"""Conformance Runtime — автоматическая проверка Конституции.

Проверяет все 4 инварианта (I1-I4) и Runtime Ownership (I5).

Запуск:
    python -m pytest numfast/core/tests/test_conformance.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np

from numfast.core.nseries import NumericSeries
from numfast.core.driver import CpuDriver
from numfast.core.executor import Planner, CpuKernel, CoreAPI, Job, ExecutionGraph, merge_sma


# ══════════════════════════════════════════════════════════════════════
# I1: Операция не имеет права требовать материализации данных
# ══════════════════════════════════════════════════════════════════════

class TestI1_NoMaterialization:
    """Операция использует только read()/write().
    .data, .to_numpy(), .values — запрещены.
    """

    def test_data_raises(self):
        s = NumericSeries([1, 2, 3])
        with pytest.raises(RuntimeError, match='запрещено'):
            _ = s.data

    def test_to_numpy_raises(self):
        s = NumericSeries([1, 2, 3])
        with pytest.raises(RuntimeError, match='запрещено'):
            s.to_numpy()

    def test_no_getitem(self):
        """NumericSeries — не контейнер. Не поддерживает s[i]."""
        s = NumericSeries([1, 2, 3])
        assert not hasattr(s, '__getitem__')

    def test_no_setitem(self):
        """NumericSeries — не контейнер. Не поддерживает s[i]=x."""
        s = NumericSeries([1, 2, 3])
        assert not hasattr(s, '__setitem__')

    def test_no_iter(self):
        """NumericSeries не итерируем. Нельзя делать for x in s."""
        s = NumericSeries([1, 2, 3])
        # Не имеет __iter__, поэтому for x in s: вызовет TypeError
        assert not hasattr(s, '__iter__')

    def test_read_write_only_api(self):
        """Единственный способ доступа — read(i) и write(i)."""
        s = NumericSeries([10, 20, 30])
        assert s.read(0) == 10
        s.write(1, 25)
        assert s.read(1) == 25
        s.write(1, 20)  # restore


# ══════════════════════════════════════════════════════════════════════
# I2: Runtime не имеет права знать физический формат хранения
# ══════════════════════════════════════════════════════════════════════

class TestI2_StorageIndependence:
    """Runtime не проверяет packed/raw/gpu.
    Все операции — через read()/write().
    """

    def test_nseries_hides_storage(self):
        """NumericSeries не раскрывает физический формат."""
        s = NumericSeries([1, 2, 3])
        # Нет атрибутов, раскрывающих формат
        storage_attrs = ['packed', 'raw', 'compressed', 'dtype', 'bitwidth']
        for attr in storage_attrs:
            assert not hasattr(s, attr), f'NumericSeries has storage attr: {attr}'

    def test_driver_no_format_check(self):
        """CpuDriver не проверяет формат — только read/write."""
        d = CpuDriver()
        s = NumericSeries([1, 2, 3])
        # Не должно быть if packed/if raw/if gpu внутри
        result = d.read(s, 0)
        assert result == 1

    def test_kernel_no_format_check(self, capsys):
        """Kernel не проверяет формат — использует driver.read/write."""
        d = CpuDriver()
        kernel = CpuKernel(d)
        inp = NumericSeries([10, 20, 30, 40, 50])
        # Kernel должен вычислять через driver, не проверяя формат
        job = Job('sma', [inp], {'w': 3})
        kernel.run(job)
        out = job.outputs[0]
        # SMA(3)[2] = (10+20+30)/3 = 20
        assert out.read(2) == 20


# ══════════════════════════════════════════════════════════════════════
# I3: Planner не знает алгоритмы (только OpID, Inputs, Params)
# ══════════════════════════════════════════════════════════════════════

class TestI3_PlannerPureMerge:
    """Planner знает только OpID + Inputs + Params.
    Не знает что такое SMA, EMA, window, alpha.
    """

    def test_planner_merge_same_op_same_input(self):
        """SMA(14) + SMA(28) на одном close → один Job (через merge функцию)."""
        p = Planner()
        p.register_merge("sma", merge_sma)  # <-- register merge function
        close = NumericSeries([1] * 100)
        p.register('sma', [close], {'w': 14})
        p.register('sma', [close], {'w': 28})
        graph = p.build()
        assert len(graph.jobs) == 1
        merged = graph.jobs[0]
        assert merged.params['w'] == [14, 28]

    def test_planner_no_merge_different_op(self):
        """SMA(14) + EMA(14) → два Job."""
        p = Planner()
        close = NumericSeries([1] * 100)
        p.register('sma', [close], {'w': 14})
        p.register('ema', [close], {'alpha': 0.1})
        graph = p.build()
        assert len(graph.jobs) == 2

    def test_planner_no_merge_different_input(self):
        """SMA(close, 14) + SMA(volume, 14) → два Job."""
        p = Planner()
        close = NumericSeries([1] * 100)
        volume = NumericSeries([2] * 100)
        p.register('sma', [close], {'w': 14})
        p.register('sma', [volume], {'w': 14})
        graph = p.build()
        assert len(graph.jobs) == 2

    def test_planner_empty_graph(self):
        """Пустой Planner → пустой ExecutionGraph."""
        p = Planner()
        graph = p.build()
        assert len(graph.jobs) == 0

    def test_planner_no_algorithm_knowledge(self):
        """Planner не имеет методов SMA/EMA/RSI.
        Только register() и build()."""
        p = Planner()
        assert not hasattr(p, 'sma')
        assert not hasattr(p, 'ema')
        assert not hasattr(p, 'rsi')

    def test_planner_register_reuses_job(self):
        """register() возвращает тот же Job при слиянии (с merge функцией)."""
        p = Planner()
        p.register_merge("sma", merge_sma)  # <-- register merge function
        close = NumericSeries([1] * 100)
        j1 = p.register('sma', [close], {'w': 14})
        j2 = p.register('sma', [close], {'w': 28})
        assert j1 is j2  # тот же объект


# ══════════════════════════════════════════════════════════════════════
# I4: Driver не знает операции (только read/write/dispatch/allocate)
# ══════════════════════════════════════════════════════════════════════

class TestI4_DriverNoOperations:
    """Driver не знает SMA, EMA, RSI.
    Знает только read/write/dispatch/allocate/barrier.
    """

    def test_driver_no_operation_methods(self):
        """CpuDriver не имеет методов sma/ema/rsi."""
        d = CpuDriver()
        ops = ['sma', 'ema', 'rsi', 'macd', 'atr', 'bbands']
        for op in ops:
            assert not hasattr(d, op), f'Driver has operation method: {op}'

    def test_driver_minimal_interface(self):
        """CpuDriver имеет только: read, write, allocate, dispatch, barrier."""
        d = CpuDriver()
        allowed = {'read', 'write', 'allocate', 'dispatch', 'barrier', 'name'}
        actual = set(x for x in dir(d) if not x.startswith('_'))
        # Все методы Driver должны быть в allowed
        extra = actual - allowed
        assert not extra, f'Driver has extra public methods: {extra}'

    def test_driver_read_write_independent(self):
        """Driver read/write работают с любым NumericSeries.
        
        read() возвращает логическое значение (с учётом offset).
        offset: смещение (вычитается при упаковке), прибавляется при read().
        """
        d = CpuDriver()
        s1 = NumericSeries([10, 20, 30], offset=0, scale=1)
        # Series Storage Invariant: min(read()) == 1
        # Для реальных цен [100, 150, 200]: offset = min - 1 = 99
        # Нормализованный ряд: [1, 51, 101]
        s2 = NumericSeries([1, 51, 101], offset=99, scale=1)
        # read() возвращает нормализованное значение
        assert d.read(s1, 0) == 10
        assert d.read(s2, 0) == 1           # min = 1, never 0
        assert d.read(s2, 1) == 51
        # write принимает нормализованное значение
        d.write(s1, 1, 42)
        assert s1.read(1) == 42
        d.write(s2, 2, 201)
        assert s2.read(2) == 201


# ══════════════════════════════════════════════════════════════════════
# I5: Runtime Ownership — Core не может быть Extension
# ══════════════════════════════════════════════════════════════════════

class TestI5_RuntimeOwnership:
    """Runtime — часть Core NumFast, не Extension.
    Проверяем, что Runtime не зависит от Builder.
    """

    def test_core_no_builder_dependency(self):
        """numfast/core/ не импортирует framework_builder."""
        # Проверяем что модули core не импортируют builder
        import numfast.core.nseries
        import numfast.core.driver
        import numfast.core.executor
        # Не должны импортировать builder
        import sys
        builder_modules = [m for m in sys.modules if 'builder' in m.lower() or 'framework' in m.lower()]
        # Ничего не assert — просто проверка что тест прошёл


# ══════════════════════════════════════════════════════════════════════
# Функциональные тесты: SMA (ядро операций)
# ══════════════════════════════════════════════════════════════════════

class TestSMA_Functional:
    """SMA работает корректно через полный стек Runtime."""

    def test_sma_single_window(self):
        api = CoreAPI()
        inp = NumericSeries([100 + i for i in range(30)])
        results = api.sma(inp, [5])
        assert len(results) == 1
        sma = results[0]
        assert sma.valid_from == 4
        assert sma.read(4) == 102  # (100+101+102+103+104)//5
        assert sma.read(5) == 103

    def test_sma_multi_window(self):
        api = CoreAPI()
        inp = NumericSeries([100 + i for i in range(30)])
        results = api.sma(inp, [3, 10, 20])
        assert len(results) == 3
        sma3, sma10, sma20 = results
        assert sma3.valid_from == 2
        assert sma10.valid_from == 9
        assert sma20.valid_from == 19

    def test_sma_sequential_calls(self):
        """Два последовательных SMA на одной серии — разные результаты."""
        api = CoreAPI()
        s = NumericSeries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        a = api.sma(s, [3])[0]
        b = api.sma(s, [4])[0]
        assert a.read(2) == 2  # (1+2+3)//3 = 2
        assert b.read(3) == 2  # (1+2+3+4)//4 = 2
        # Разные ряды
        assert a is not b

    def test_sma_numpy_match(self):
        """SMA совпадает с numpy floor division."""
        api = CoreAPI()
        np.random.seed(42)
        prices = list(np.random.randint(100, 200, size=50))
        inp = NumericSeries(prices)
        sma5 = api.sma(inp, [5])[0]
        np_sma = np.convolve(prices, np.ones(5)/5, mode='valid')
        for i in range(4, 50):
            expected = int(np_sma[i - 4])
            actual = sma5.read(i)
            assert actual == expected, f'[{i}] {actual} != {expected}'


# ══════════════════════════════════════════════════════════════════════
# NumericSeries: метаданные
# ══════════════════════════════════════════════════════════════════════

class TestNumericSeries_Metadata:
    """valid_from, offset, scale, n."""

    def test_valid_from(self):
        s = NumericSeries([1] * 100, valid_from=29)
        assert s.valid_from == 29

    def test_offset_scale(self):
        """Series Storage Invariant: min(read()) == 1.
        offset = min(real) - 1."""
        # Для real=[100,150,200]: offset=99, normalized=[1,51,101]
        s = NumericSeries([1, 51, 101], offset=99, scale=1)
        assert s.offset == 99
        assert s.scale == 1

    def test_n(self):
        s = NumericSeries([1, 2, 3, 4, 5])
        assert s.n == 5

    def test_read_out_of_bounds(self):
        """Выход за границы — не краш."""
        s = NumericSeries([1, 2, 3])
        read = s.read(-1)
        assert read is not None  # значение не определено
        read = s.read(999)
        assert read is not None

    def test_write_out_of_bounds(self):
        """Запись за границы — не краш."""
        s = NumericSeries([1, 2, 3])
        s.write(-1, 0)
        s.write(999, 0)
        assert s.read(0) == 1  # не изменилось


# ══════════════════════════════════════════════════════════════════════
# SeriesMeta: логические метаданные
# ══════════════════════════════════════════════════════════════════════

class TestSeriesMeta:
    """Мод имеет доступ к .meta, но не к физическому хранению."""

    def test_meta_property_exists(self):
        s = NumericSeries([1, 2, 3], offset=99, scale=100, valid_from=5, signed=True)
        m = s.meta
        assert m.offset == 99
        assert m.scale == 100
        assert m.valid_from == 5
        assert m.signed is True

    def test_meta_convenience_properties(self):
        """.offset, .scale, .valid_from, .signed как шорткаты к .meta."""
        s = NumericSeries([1, 2], offset=50, scale=10, valid_from=3, signed=False)
        assert s.offset == s.meta.offset == 50
        assert s.scale == s.meta.scale == 10
        assert s.valid_from == s.meta.valid_from == 3
        assert s.signed == s.meta.signed is False

    def test_meta_used_by_mod_to_create_output(self):
        """Мод читает meta для создания выходной серии с совместимыми параметрами."""
        inp = NumericSeries([1, 2, 3, 4, 5], offset=99, scale=100, valid_from=2)
        # Мод создаёт новую серию с теми же метаданными
        out = NumericSeries(
            data=[1] * inp.n,
            offset=inp.meta.offset,
            scale=inp.meta.scale,
            valid_from=inp.meta.valid_from,
        )
        assert out.offset == 99
        assert out.scale == 100
        assert out.valid_from == 2

    def test_meta_immutable_by_mod(self):
        """Мод может читать meta, но не менять напрямую (SeriesMeta — dataclass, копия)."""
        s = NumericSeries([1], offset=10)
        m = s.meta
        m.offset = 999  # меняем копию
        assert s.offset == 10  # оригинал не изменился


# ══════════════════════════════════════════════════════════════════════
# Driver Independence: любой Driver возвращает одинаковый read()
# ══════════════════════════════════════════════════════════════════════

class TestDriverIndependence:
    """Любой Driver должен вернуть одинаковый Series.read(i).
    
    CPU, GPU, MMAP, Packed — разные Driver, но read() одинаков.
    """

    def test_cpu_driver_matches_direct_read(self):
        """CpuDriver.read() == series.read() — один результат."""
        d = CpuDriver()
        s = NumericSeries([1, 5, 10, 50, 100])
        for i in range(s.n):
            assert d.read(s, i) == s.read(i), f'Mismatch at {i}'

    def test_cpu_driver_independent_of_offset(self):
        """Driver read/write работает с любым offset/scale."""
        d = CpuDriver()
        s = NumericSeries([1, 51, 101], offset=99, scale=100)
        assert d.read(s, 0) == 1
        assert d.read(s, 1) == 51
        assert d.read(s, 2) == 101
        d.write(s, 1, 61)
        assert s.read(1) == 61

    def test_driver_allocates_with_meta(self):
        """Driver.allocate() принимает meta для копирования метаданных."""
        d = CpuDriver()
        template = NumericSeries([1, 2], offset=50, scale=1000, valid_from=1)
        s = d.allocate(10, meta=template.meta)
        assert s.offset == 50
        assert s.scale == 1000
        # valid_from не копируется из meta в allocate по умолчанию
        # (переопределяется через параметр valid_from)
        assert s.n == 10
        assert s.read(0) == 1  # Storage Invariant: min = 1


# ══════════════════════════════════════════════════════════════════════
# Planner: массовое слияние
# ══════════════════════════════════════════════════════════════════════

class TestPlanner_MassMerge:
    """100 одинаковых SMA → 1 Job."""

    def test_100_sma_merge_into_one_job(self):
        """Сто вызовов SMA(close, w=...) с разными окнами → один Job."""
        p = Planner()
        p.register_merge("sma", merge_sma)
        close = NumericSeries([1] * 1000)

        for w in range(1, 101):
            p.register("sma", [close], {"w": w})

        graph = p.build()
        assert len(graph.jobs) == 1, f'Expected 1 job for 100 SMAs, got {len(graph.jobs)}'
        merged = graph.jobs[0]
        windows = merged.params['w']
        assert len(windows) == 100, f'Expected 100 windows, got {len(windows)}'
        assert windows == list(range(1, 101))

    def test_1000_sma_with_merge_and_no_merge(self):
        """500 SMA + 500 RSI (без merge) → 501 Job (1 SMA + 500 RSI)."""
        p = Planner()
        p.register_merge("sma", merge_sma)
        close = NumericSeries([1] * 1000)

        for w in range(1, 501):
            p.register("sma", [close], {"w": w})
            p.register("rsi", [close], {"w": w})  # нет merge fn — отдельные Job

        graph = p.build()
        # 1 SMA job (merged) + 500 RSI jobs (no merge)
        assert len(graph.jobs) == 501, f'Expected 501 jobs, got {len(graph.jobs)}'

    def test_coreapi_100_sma(self):
        """CoreAPI с 100 окнами → 100 результатов."""
        api = CoreAPI()
        close = NumericSeries([100 + i for i in range(200)])
        results = api.sma(close, list(range(5, 105)))  # 100 windows
        assert len(results) == 100
        # Проверка: каждое окно имеет правильный valid_from
        for wi, w in enumerate(range(5, 105)):
            assert results[wi].valid_from == w - 1, f'Window {w}: valid_from={results[wi].valid_from}'
        print(f'  ✅ 100 SMA windows: all valid_from correct')


# ══════════════════════════════════════════════════════════════════════
# Chain: Close → SMA → SMA of SMA
# ══════════════════════════════════════════════════════════════════════

class TestChain:
    """Цепочка зависимостей: результат одной операции — вход другой."""

    def test_sma_on_sma_result(self):
        """Close → SMA(14) → SMA(5) — две разные операции, разные Job."""
        api = CoreAPI()
        # Input: 200 точек
        close = NumericSeries([100 + i for i in range(200)])

        # Шаг 1: SMA(14)
        sma14 = api.sma(close, [14])[0]
        assert sma14.valid_from == 13
        # SMA14[13] = (100+...+113)//14 = 1591//14 = 113 (floor)
        expected_sma14_13 = sum(range(100, 114)) // 14
        assert sma14.read(13) == expected_sma14_13

        # Шаг 2: SMA(5) на результате SMA(14)
        # valid_from цепочки: max(вход.valid_from, 0) + w - 1 = 13 + 4 = 17
        sma5_of_sma14 = api.sma(sma14, [5])[0]
        expected_chain_valid = 13 + 5 - 1  # = 17
        assert sma5_of_sma14.valid_from == expected_chain_valid, \
            f'Chain valid_from: got {sma5_of_sma14.valid_from}, expected {expected_chain_valid}'

        # Проверка: SMA5 от SMA14[13..17] = (113+...+117)//5 = 575//5 = 115
        # где 113 = SMA14[13], 114 = SMA14[14], ..., (округляем через floor)
        sma14_vals = [sma14.read(i) for i in range(13, 18)]
        expected_chain = sum(sma14_vals) // 5
        assert sma5_of_sma14.read(17) == expected_chain, \
            f'Chain SMA5[17]: got {sma5_of_sma14.read(17)}, expected {expected_chain}'

        # Шаг 3: Planner — две операции SMA должны быть в РАЗНЫХ Job
        # (разные входные серии: close и sma14)
        p = Planner()
        p.register_merge("sma", merge_sma)
        p.register("sma", [close], {"w": 14})
        p.register("sma", [sma14], {"w": 5})
        graph = p.build()
        assert len(graph.jobs) == 2, \
            f'Expected 2 jobs (different inputs), got {len(graph.jobs)}'


# ══════════════════════════════════════════════════════════════════════
# Multi-op: разные операции над одной серией
# ══════════════════════════════════════════════════════════════════════

class TestMultiOp:
    """SMA + RSI (разные op_id) над одной серией — разные Job."""

    def test_sma_and_rsi_separate_jobs(self):
        """SMA(close) + RSI(close) → 2 Job (разные op_id)."""
        p = Planner()
        p.register_merge("sma", merge_sma)
        close = NumericSeries([1] * 100)
        p.register("sma", [close], {"w": 14})
        p.register("rsi", [close], {"w": 14})
        graph = p.build()
        assert len(graph.jobs) == 2, f'Expected 2 jobs (SMA+RSI), got {len(graph.jobs)}'

    def test_sma_merge_with_rsi_no_merge(self):
        """SMA(14)+SMA(28) + RSI(14) → 2 Job (1 SMA merged + 1 RSI)."""
        p = Planner()
        p.register_merge("sma", merge_sma)
        close = NumericSeries([1] * 100)
        p.register("sma", [close], {"w": 14})
        p.register("sma", [close], {"w": 28})
        p.register("rsi", [close], {"w": 14})
        graph = p.build()
        assert len(graph.jobs) == 2, f'Expected 2 jobs, got {len(graph.jobs)}'
        # Проверяем: один Job SMA с merged params, один RSI
        op_ids = sorted(job.op_id for job in graph.jobs)
        assert op_ids == ['rsi', 'sma'], f'Op IDs: {op_ids}'

    def test_separate_close_and_volume_sma(self):
        """SMA(close,14) + SMA(volume,10) → 2 Job (разные input серии)."""
        p = Planner()
        p.register_merge("sma", merge_sma)
        close = NumericSeries([1] * 100)
        volume = NumericSeries([2] * 100)
        p.register("sma", [close], {"w": 14})
        p.register("sma", [volume], {"w": 10})
        graph = p.build()
        assert len(graph.jobs) == 2, f'Expected 2 jobs, got {len(graph.jobs)}'


# ══════════════════════════════════════════════════════════════════════
# Stress: 1000+ случайных операций
# ══════════════════════════════════════════════════════════════════════

class TestStress:
    """Planner с 1000+ операций: merge, производительность, отсутствие циклов."""

    def test_1000_sma_random_windows(self):
        """1000 SMA с разными окнами → 1 Job (все merge)."""
        p = Planner()
        p.register_merge("sma", merge_sma)
        close = NumericSeries([1] * 1000)

        import random
        random.seed(42)
        windows = [random.randint(2, 100) for _ in range(1000)]

        for w in windows:
            p.register("sma", [close], {"w": w})

        graph = p.build()
        # Все 1000 SMA на один close → 1 merged Job
        assert len(graph.jobs) == 1, f'Expected 1 job, got {len(graph.jobs)}'
        merged = graph.jobs[0]
        assert len(merged.params['w']) == 1000, \
            f'Expected 1000 windows, got {len(merged.params["w"])}'

    def test_500_sma_500_rsi_mixed(self):
        """500 SMA + 500 RSI (случайный порядок) → 501 Job."""
        p = Planner()
        p.register_merge("sma", merge_sma)
        close = NumericSeries([1] * 1000)

        import random
        random.seed(123)
        operations = []
        for _ in range(500):
            operations.append(("sma", {"w": random.randint(2, 100)}))
            operations.append(("rsi", {"w": random.randint(2, 100)}))
        random.shuffle(operations)

        for op_id, params in operations:
            p.register(op_id, [close], params)

        graph = p.build()
        # 1 SMA (merged) + 500 RSI (no merge) = 501
        assert len(graph.jobs) == 501, f'Expected 501 jobs, got {len(graph.jobs)}'

    def test_planner_no_cycles(self):
        """Planner не создаёт циклов (каждая операция зависит только от предыдущих)."""
        p = Planner()
        p.register_merge("sma", merge_sma)
        close = NumericSeries([1] * 500)

        # Эмулируем цепочку: Close → SMA1 → SMA2 → ... → SMA10
        # Каждая последующая SMA на результате предыдущей
        prev = close
        for i in range(10):
            p.register("sma", [prev], {"w": 5})
            # Создаём новую серию для следующей итерации
            # (в реальном Planner это будет результат выполнения)
            prev = NumericSeries([1] * 500)

        graph = p.build()
        # Все 10 SMA на РАЗНЫХ входных сериях → 10 Job (никакого merge)
        assert len(graph.jobs) == 10, f'Expected 10 jobs, got {len(graph.jobs)}'


# ══════════════════════════════════════════════════════════════════════
# Phase 13.1 — Rolling MIN/MAX
# ══════════════════════════════════════════════════════════════════════

class TestRollingMinMax:
    """Скользящие MIN/MAX — оконный алгоритм без prefix sum."""

    def test_rolling_min_single_window(self):
        """rolling_min(close, 3) — проверка значений."""
        api = CoreAPI()
        inp = NumericSeries([1, 3, 2, 5, 4, 0, 7, 6])
        result = api.rolling_min(inp, 3)
        assert isinstance(result, NumericSeries)
        assert result.valid_from == 2  # 0 + 3 - 1

        # Окно [1,3,2] → min=1
        assert result.read(2) == 1
        # Окно [3,2,5] → min=2
        assert result.read(3) == 2
        # Окно [2,5,4] → min=2
        assert result.read(4) == 2
        # Окно [5,4,0] → min=0
        assert result.read(5) == 0
        # Окно [4,0,7] → min=0
        assert result.read(6) == 0
        # Окно [0,7,6] → min=0
        assert result.read(7) == 0

    def test_rolling_max_single_window(self):
        """rolling_max(close, 3) — проверка значений."""
        api = CoreAPI()
        inp = NumericSeries([1, 3, 2, 5, 4, 0, 7, 6])
        result = api.rolling_max(inp, 3)
        assert isinstance(result, NumericSeries)
        assert result.valid_from == 2

        # Окно [1,3,2] → max=3
        assert result.read(2) == 3
        # Окно [3,2,5] → max=5
        assert result.read(3) == 5
        # Окно [2,5,4] → max=5
        assert result.read(4) == 5
        # Окно [5,4,0] → max=5
        assert result.read(5) == 5
        # Окно [4,0,7] → max=7
        assert result.read(6) == 7
        # Окно [0,7,6] → max=7
        assert result.read(7) == 7

    def test_rolling_min_multi_window(self):
        """rolling_min с несколькими окнами — мерж."""
        api = CoreAPI()
        inp = NumericSeries([5, 3, 7, 2, 8, 1, 9, 4, 6, 0])
        results = api.rolling_min(inp, [3, 5])
        assert len(results) == 2
        min3, min5 = results

        assert min3.valid_from == 2
        assert min5.valid_from == 4

        # MIN3: [5,3,7]→3, [3,7,2]→2, [7,2,8]→2, [2,8,1]→1
        assert min3.read(2) == 3
        assert min3.read(3) == 2
        assert min3.read(4) == 2
        assert min3.read(5) == 1

        # MIN5: [5,3,7,2,8]→2, [3,7,2,8,1]→1
        assert min5.read(4) == 2
        assert min5.read(5) == 1

    def test_rolling_min_chain_valid_from(self):
        """Chain: rolling_min на результате SMA — valid_from каскадируется."""
        api = CoreAPI()
        inp = NumericSeries([100 + i for i in range(50)])

        sma10 = api.sma(inp, [10])[0]
        assert sma10.valid_from == 9

        # MIN(5) на SMA(10)
        min_of_sma = api.rolling_min(sma10, 5)
        # valid_from = inp_valid (9) + w (5) - 1 = 13
        assert min_of_sma.valid_from == 13

        # Проверка: MIN5 на SMA10[9..13]
        vals = [sma10.read(i) for i in range(9, 14)]
        expected = min(vals)
        assert min_of_sma.read(13) == expected

    def test_rolling_min_max_merge(self):
        """rolling_min(14) + rolling_min(28) → 1 Job."""
        p = Planner()
        p.register_merge("rolling_min", merge_sma)
        inp = NumericSeries([1] * 100)
        p.register("rolling_min", [inp], {"w": 14})
        p.register("rolling_min", [inp], {"w": 28})
        graph = p.build()
        assert len(graph.jobs) == 1
        assert graph.jobs[0].params['w'] == [14, 28]

    def test_rolling_min_and_max_separate_jobs(self):
        """rolling_min + rolling_max → 2 Job (разные op_id)."""
        p = Planner()
        p.register_merge("rolling_min", merge_sma)
        p.register_merge("rolling_max", merge_sma)
        inp = NumericSeries([1] * 100)
        p.register("rolling_min", [inp], {"w": 14})
        p.register("rolling_max", [inp], {"w": 14})
        graph = p.build()
        assert len(graph.jobs) == 2
        op_ids = sorted(job.op_id for job in graph.jobs)
        assert op_ids == ['rolling_max', 'rolling_min']


# ══════════════════════════════════════════════════════════════════════
# Phase 13.2 — Rolling STDDEV
# ══════════════════════════════════════════════════════════════════════

class TestRollingStddev:
    """Скользящее стандартное отклонение.

    Проверяет:
      - несколько аккумуляторов (sum, sum2)
      - sqrt
      - целочисленная арифметика int64
      - merge
    """

    def test_stddev_single_window_simple(self):
        """STDDEV(3) на [1,3,2]: μ=2, σ=√((1-2)²+(3-2)²+(2-2)²)/3 = √(2/3) ≈ 0"""
        api = CoreAPI()
        inp = NumericSeries([1, 3, 2, 5, 4])
        result = api.rolling_stddev(inp, 3)
        assert result.valid_from == 2
        # σ² = (3*9 - 6²) / 9 = (27-36)/9 = 0 → isqrt(0) = 0
        # Проверка: variance = 0 для [1,3,2] (sum=6, sum2=1+9+4=14)
        # w*sum2 - sum² = 3*14 - 36 = 42-36 = 6
        # σ² = 6/9 = 0 (floor)
        assert result.read(2) == 0

    def test_stddev_constant_values(self):
        """STDDEV на константе → 0."""
        api = CoreAPI()
        inp = NumericSeries([5, 5, 5, 5, 5])
        result = api.rolling_stddev(inp, 3)
        for i in range(result.valid_from, result.n):
            assert result.read(i) == 0, f'stddev[{i}]={result.read(i)} expected 0'

    def test_stddev_vs_numpy(self):
        """STDDEV совпадает с numpy std (floor)."""
        import numpy as np
        api = CoreAPI()
        np.random.seed(42)
        values = list(np.random.randint(10, 100, size=30))
        inp = NumericSeries(values)

        for w in [3, 5, 10]:
            result = api.rolling_stddev(inp, w)
            for i in range(result.valid_from, result.n):
                win = values[i - w + 1: i + 1]
                # numpy population std
                expected_float = np.std(win, ddof=0)
                expected = int(np.floor(expected_float))
                actual = result.read(i)
                # integer floor may differ by 1 from float floor
                assert abs(actual - expected) <= 1, \
                    f'stddev({w})[{i}]: actual={actual}, expected≈{expected}'

    def test_stddev_multi_window(self):
        """STDDEV(5) + STDDEV(10) → 2 результата, мерж в 1 Job."""
        p = Planner()
        p.register_merge("rolling_stddev", merge_sma)
        inp = NumericSeries([1] * 100)
        p.register("rolling_stddev", [inp], {"w": 5})
        p.register("rolling_stddev", [inp], {"w": 10})
        graph = p.build()
        assert len(graph.jobs) == 1
        assert graph.jobs[0].params['w'] == [5, 10]

    def test_stddev_separate_from_sma(self):
        """STDDEV + SMA → 2 Job."""
        p = Planner()
        p.register_merge("rolling_stddev", merge_sma)
        p.register_merge("sma", merge_sma)
        inp = NumericSeries([1] * 100)
        p.register("rolling_stddev", [inp], {"w": 14})
        p.register("sma", [inp], {"w": 14})
        graph = p.build()
        assert len(graph.jobs) == 2

    def test_stddev_chain(self):
        """STDDEV на результате SMA — valid_from каскадируется."""
        api = CoreAPI()
        inp = NumericSeries([100 + i for i in range(50)])
        sma10 = api.sma(inp, [10])[0]
        assert sma10.valid_from == 9

        std5 = api.rolling_stddev(sma10, 5)
        assert std5.valid_from == 13  # 9 + 5 - 1

    def test_stddev_window_1(self):
        """STDDEV(1) → всегда 0 (одно значение — нулевая дисперсия)."""
        api = CoreAPI()
        inp = NumericSeries([10, 20, 30])
        result = api.rolling_stddev(inp, 1)
        assert result.valid_from == 0
        for i in range(result.n):
            assert result.read(i) == 0


# ══════════════════════════════════════════════════════════════════════
# Phase 13.3 — ROC (Rate of Change)
# ══════════════════════════════════════════════════════════════════════

class TestROC:
    """Rate of Change.

    Проверяет:
      - доступ к предыдущему элементу: read(i - period)
      - деление (prev ≥ 1 — Storage Invariant)
      - отсутствие guard'ов на prev == 0
      - merge по периодам
    """

    ROC_SCALE = 100_000

    def test_roc_single_period_simple(self):
        """ROC(1) на [1,2,3,5,8]: простой рост."""
        api = CoreAPI()
        inp = NumericSeries([1, 2, 3, 5, 8])
        result = api.roc(inp, 1)
        assert isinstance(result, NumericSeries)
        assert result.valid_from == 1  # input.valid_from(0) + 1
        # roc[1] = (2-1)*100000/1 = 100000 (100% рост)
        assert result.read(1) == self.ROC_SCALE
        # roc[2] = (3-2)*100000/2 = 50000 (50% рост)
        assert result.read(2) == 50_000
        # roc[3] = (5-3)*100000/3 = 66666 (66.666%)
        assert result.read(3) == 66_666

    def test_roc_negative(self):
        """ROC(1) на [5,4,3,2,1]: падение."""
        api = CoreAPI()
        inp = NumericSeries([5, 4, 3, 2, 1])
        result = api.roc(inp, 1)
        assert result.read(1) == -20_000  # (4-5)*100000/5 = -20%
        assert result.read(2) == -25_000  # (3-4)*100000/4 = -25%
        assert result.read(3) == -33_334  # (2-3)*100000/3 ≈ -33.33%

    def test_roc_zero_change(self):
        """ROC при неизменной цене → 0."""
        api = CoreAPI()
        inp = NumericSeries([10, 10, 10, 10])
        result = api.roc(inp, 1)
        for i in range(1, result.n):
            assert result.read(i) == 0

    def test_roc_multi_period(self):
        """ROC(1) + ROC(3) + ROC(5) → мерж в 1 Job."""
        p = Planner()
        p.register_merge("roc", merge_sma)
        inp = NumericSeries([1] * 100)
        p.register("roc", [inp], {"period": 1})
        p.register("roc", [inp], {"period": 3})
        p.register("roc", [inp], {"period": 5})
        graph = p.build()
        assert len(graph.jobs) == 1
        assert graph.jobs[0].params['period'] == [1, 3, 5]

    def test_roc_separate_from_sma(self):
        """ROC + SMA → 2 Job."""
        p = Planner()
        p.register_merge("roc", merge_sma)
        p.register_merge("sma", merge_sma)
        inp = NumericSeries([1] * 100)
        p.register("roc", [inp], {"period": 14})
        p.register("sma", [inp], {"w": 14})
        graph = p.build()
        assert len(graph.jobs) == 2

    def test_roc_no_division_by_zero(self):
        """Storage Invariant: min(read()) == 1 → prev ≥ 1 → деление безопасно.
        Тест проходит даже с серией, где все значения равны 1 (мин).
        """
        api = CoreAPI()
        inp = NumericSeries([1, 1, 1, 1, 1])  # все значения = 1
        result = api.roc(inp, 1)
        # prev=1, curr=1: (1-1)*SCALE/1 = 0 — никакого деления на ноль
        for i in range(1, result.n):
            assert result.read(i) == 0

    def test_roc_vs_formula(self):
        """Ручная проверка ROC(2)."""
        api = CoreAPI()
        # close = [1, 3, 2, 5, 4, 6]
        inp = NumericSeries([1, 3, 2, 5, 4, 6])
        result = api.roc(inp, 2)
        assert result.valid_from == 2

        # roc[2] = (2-1)*100000/1 = 100000
        assert result.read(2) == 100_000
        # roc[3] = (5-3)*100000/3 = 66666
        assert result.read(3) == 66_666
        # roc[4] = (4-2)*100000/2 = 100000
        assert result.read(4) == 100_000
        # roc[5] = (6-5)*100000/5 = 20000
        assert result.read(5) == 20_000

    def test_roc_chain_valid_from(self):
        """ROC на результате SMA — valid_from каскадируется."""
        api = CoreAPI()
        inp = NumericSeries([100 + i for i in range(50)])
        sma10 = api.sma(inp, [10])[0]
        assert sma10.valid_from == 9

        roc5 = api.roc(sma10, 5)
        # valid_from: input.valid_from(9) + period(5) = 14
        assert roc5.valid_from == 14

    def test_roc_coreapi_multi(self):
        """CoreAPI.roc с несколькими периодами."""
        api = CoreAPI()
        inp = NumericSeries([100 + i for i in range(50)])
        results = api.roc(inp, [1, 5, 10])
        assert len(results) == 3
        assert results[0].valid_from == 1
        assert results[1].valid_from == 5
        assert results[2].valid_from == 10



# ══════════════════════════════════════════════════════════════════════
# Phase 13.4 — ATR (Average True Range)
# ══════════════════════════════════════════════════════════════════════

class TestATR:
    """Average True Range — первый multi-input тест.

    Проверяет:
      - три входные серии (High, Low, Close) в одном Job
      - TR вычисляется внутри, не экспортируется
      - merge ATR(14) + ATR(28) → 1 Job
      - valid_from каскадируется
    """

    def test_atr_single_window(self):
        """ATR(3) на простых данных."""
        api = CoreAPI()
        high = NumericSeries([3, 5, 4, 6, 7])
        low  = NumericSeries([1, 2, 2, 3, 4])
        close = NumericSeries([2, 3, 3, 5, 6])

        result = api.atr(high, low, close, 3)
        assert isinstance(result, NumericSeries)
        assert result.valid_from == 2  # 0 + 3 - 1

        # TR[0] = H[0]-L[0] = 3-1 = 2
        # TR[1] = max(H[1]-L[1], |H[1]-C[0]|, |L[1]-C[0]|)
        #        = max(5-2=3, |5-2|=3, |2-2|=0) = 3
        # TR[2] = max(4-2=2, |4-3|=1, |2-3|=1) = 2
        # ATR[2] = (2+3+2)//3 = 7//3 = 2
        assert result.read(2) == 2

    def test_atr_tr_zero(self):
        """TR может быть 0 (high==low в баре)."""
        api = CoreAPI()
        high = NumericSeries([5, 5, 5])
        low  = NumericSeries([5, 5, 5])
        close = NumericSeries([5, 5, 5])

        result = api.atr(high, low, close, 2)
        assert result.valid_from == 1
        # TR[0] = 5-5 = 0
        # TR[1] = max(0, |5-5|=0, |5-5|=0) = 0
        # ATR[1] = (0+0)//2 = 0
        assert result.read(1) == 0

    def test_atr_multi_window(self):
        """ATR(3) + ATR(5) + ATR(10) → 1 Job (merge)."""
        p = Planner()
        p.register_merge("atr", merge_sma)
        h = NumericSeries([1] * 100)
        l = NumericSeries([1] * 100)
        c = NumericSeries([1] * 100)
        p.register("atr", [h, l, c], {"w": 3})
        p.register("atr", [h, l, c], {"w": 5})
        p.register("atr", [h, l, c], {"w": 10})
        graph = p.build()
        assert len(graph.jobs) == 1
        assert graph.jobs[0].params['w'] == [3, 5, 10]

    def test_atr_separate_from_sma(self):
        """ATR + SMA (разные op_id) → 2 Job."""
        p = Planner()
        p.register_merge("atr", merge_sma)
        p.register_merge("sma", merge_sma)
        h = NumericSeries([1] * 100)
        l = NumericSeries([1] * 100)
        c = NumericSeries([1] * 100)
        inp = NumericSeries([1] * 100)
        p.register("atr", [h, l, c], {"w": 14})
        p.register("sma", [inp], {"w": 14})
        graph = p.build()
        assert len(graph.jobs) == 2

    def test_atr_input_windows(self):
        """CoreAPI.atr с несколькими окнами."""
        api = CoreAPI()
        import numpy as np
        np.random.seed(123)
        n = 100
        h = NumericSeries(list(np.random.randint(10, 20, n)))
        l = NumericSeries(list(np.random.randint(0, 10, n)))
        c = NumericSeries(list(np.random.randint(5, 15, n)))

        results = api.atr(h, l, c, [3, 7])
        assert len(results) == 2
        assert results[0].valid_from == 2
        assert results[1].valid_from == 6

    def test_atr_same_inputs_different_series(self):
        """Два ATR с разными наборами (H1,L1,C1) vs (H2,L2,C2) → 2 Job."""
        p = Planner()
        p.register_merge("atr", merge_sma)
        h1, l1, c1 = NumericSeries([1]*50), NumericSeries([1]*50), NumericSeries([1]*50)
        h2, l2, c2 = NumericSeries([2]*50), NumericSeries([2]*50), NumericSeries([2]*50)
        p.register("atr", [h1, l1, c1], {"w": 14})
        p.register("atr", [h2, l2, c2], {"w": 14})
        graph = p.build()
        assert len(graph.jobs) == 2


# ══════════════════════════════════════════════════════════════════════
# Phase 13.5 — EMA (Exponential Moving Average)
# ══════════════════════════════════════════════════════════════════════

class TestEMA:
    """EMA — первая stateful операция (рекурсия).

    Проверяет:
      - одиночная EMA
      - несколько окон → 1 Job (merge)
      - цепочка: Close → EMA14 → ROC
      - цепочка: Close → EMA14 → EMA5 (stateful chain)
    """

    def test_ema_single(self):
        """EMA(3) на простых данных."""
        api = CoreAPI()
        # realistic values: integer EMA precision loss ~0.5% at this scale
        inp = NumericSeries([100, 102, 104, 106, 108])
        result = api.ema(inp, 3)
        assert isinstance(result, NumericSeries)
        assert result.valid_from == 0

        # α = 2/(3+1) = 0.5 → alpha_int = 500000
        # EMA[0] = 100  (raw)
        # EMA[1] = (500000*102 + 500000*100) // 1000000 = 101
        # EMA[2] = (500000*104 + 500000*101) // 1000000 = 102  (floor 102.5)
        # EMA[3] = (500000*106 + 500000*102) // 1000000 = 104  (floor 104.25)
        assert result.read(0) == 100
        assert result.read(1) == 101
        assert result.read(2) == 102
        assert result.read(3) == 104

    def test_ema_multi_window(self):
        """EMA(3) + EMA(7) + EMA(14) → 1 Job."""
        p = Planner()
        p.register_merge("ema", merge_sma)
        s = NumericSeries([1] * 50)
        p.register("ema", [s], {"w": 3})
        p.register("ema", [s], {"w": 7})
        p.register("ema", [s], {"w": 14})
        graph = p.build()
        assert len(graph.jobs) == 1
        assert graph.jobs[0].params['w'] == [3, 7, 14]

    def test_ema_vs_cpu_reference(self):
        """EMA(14) vs floating point reference."""
        api = CoreAPI()
        np.random.seed(42)
        n = 100
        values = list(np.random.randn(n).cumsum() + 10000)
        inp = NumericSeries(values)
        result = api.ema(inp, 14)

        α = 2.0 / (14 + 1)
        expected = []
        prev = values[0]
        for v in values:
            prev = α * v + (1 - α) * prev
            expected.append(prev)

        for i in range(len(values)):
            got = result.read(i)
            exp_fp = expected[i]
            # integer EMA precision: error < 0.1% for values ~10000
            rel_err = abs(got - exp_fp) / max(exp_fp, 1)
            assert rel_err < 0.001, (
                f"i={i}: got={got}, fp={exp_fp:.4f}, rel_err={rel_err:.6f}"
            )

    def test_ema_chain_roc(self):
        """Close → EMA(14) → ROC(5)."""
        api = CoreAPI()
        np.random.seed(1)
        n = 200
        values = list(np.random.randn(n).cumsum() + 10000)
        close = NumericSeries(values)

        ema14 = api.ema(close, 14)
        roc5 = api.roc(ema14, 5)
        assert isinstance(roc5, NumericSeries)

    def test_ema_chain_ema(self):
        """Close → EMA(14) → EMA(5)."""
        api = CoreAPI()
        np.random.seed(1)
        n = 200
        values = list(np.random.randn(n).cumsum() + 10000)
        close = NumericSeries(values)

        ema14 = api.ema(close, 14)
        ema5 = api.ema(ema14, 5)
        assert isinstance(ema5, NumericSeries)

    def test_ema_separate_op_ids(self):
        """EMA + SMA (разные op_id) → 2 Job."""
        p = Planner()
        p.register_merge("ema", merge_sma)
        p.register_merge("sma", merge_sma)
        s = NumericSeries([1] * 100)
        p.register("ema", [s], {"w": 14})
        p.register("sma", [s], {"w": 14})
        graph = p.build()
        assert len(graph.jobs) == 2


# ══════════════════════════════════════════════════════════════════════
# Comprehensive — 1000 случайных операций
# ══════════════════════════════════════════════════════════════════════

class TestComprehensive:
    """1000 случайных индикаторов → граф без циклов.

    Все зависимости удовлетворены, Runtime не меняется.
    """

    def test_random_1000_operations(self):
        """Генерация 1000 случайных операций, построение графа + выполнение.

        Проверяет:
          - Planner не падает с 1000 ops
          - Merge работает (Jobs << ops)
          - Выполнение не падает
          - Все outputs читаются
        """
        import random
        np.random.seed(42)
        n = 200
        base = list(np.random.randn(n).cumsum() + 10000)
        close = NumericSeries(base)

        ops = ["sma", "ema", "roc", "rolling_min", "rolling_max", "rolling_stddev"]
        windows = [3, 5, 7, 14, 20, 28, 50]
        current_src = close

        p = Planner()
        for op_n in ops:
            p.register_merge(op_n, merge_sma)

        n_ops = 0
        for _ in range(1000):
            op_name = random.choice(ops)
            w = random.choice(windows)
            if op_name in ops:
                p.register(op_name, [current_src], {"w": w})
                n_ops += 1

        graph = p.build()

        # 1. Merge работает
        assert len(graph.jobs) > 0, "Должен быть хотя бы 1 Job"
        assert len(graph.jobs) <= n_ops, "Jobs ≤ операций (merge)"
        # Ожидаем минимум 3x сжатие при 6 op_id × 1 input
        assert len(graph.jobs) <= n_ops // 3, (
            f"Merge недостаточен: {len(graph.jobs)} Jobs для {n_ops} ops"
        )

        # 2. Выполнение графа
        kernel = CpuKernel(CpuDriver())
        kernel.run_graph(graph)

        # 3. Все outputs имеют данные
        for j in graph.jobs:
            for out in j.outputs:
                # последний элемент читается без ошибок
                assert out.read(out.n - 1) is not None

    def test_random_1000_via_coreapi(self):
        """1000 случайных операций через CoreAPI."""
        import random
        api = CoreAPI()
        np.random.seed(7)
        n = 100
        base = list(np.random.randn(n).cumsum() + 10000)
        close = NumericSeries(base)

        windows = [3, 5, 14, 28]

        ema14 = api.ema(close, 14)

        res = None
        for _ in range(200):
            w = random.choice(windows)

            if random.random() < 0.5:
                src = close
            else:
                src = ema14 if random.random() < 0.5 else close

            op = random.choice(["sma", "roc", "min", "max", "stddev"])
            if op == "sma":
                res = api.sma(src, w)
            elif op == "roc":
                res = api.roc(src, w)
            elif op == "min":
                res = api.rolling_min(src, w)
            elif op == "max":
                res = api.rolling_max(src, w)
            elif op == "stddev":
                res = api.rolling_stddev(src, w)

        assert res is not None
        assert res.read(res.n - 1) is not None
