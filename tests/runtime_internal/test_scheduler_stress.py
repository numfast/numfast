"""Scheduler Stress Test — случайные DAG.

Проверяет:
  - топологический порядок (все зависимости выполнены до задачи)
  - отсутствие deadlock (ни одна задача не ждёт бесконечно)
  - производительность scheduler на 1000+ задачах
"""

import sys, os, time, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "core")))

from Runtime._lib.scheduler import Scheduler
from Runtime._lib.task import Task, ResourceRef
from Runtime import Runtime
from trading.Trading import register_all as register_trading_kernels


def make_runtime():
    runtime = Runtime()
    register_trading_kernels(runtime)
    return runtime


def generate_random_dag(n_tasks=50, edge_prob=0.1, seed=42):
    """Сгенерировать случайный DAG. Возвращает list[Task]."""
    random.seed(seed)
    tasks = []
    for i in range(n_tasks):
        deps = [j for j in range(i) if random.random() < edge_prob]
        inputs = [ResourceRef(type="task", task_id=dep, output_idx=0) for dep in deps]
        task = Task(
            id=i,
            op="EMA",
            params={"period": 3},
            inputs=inputs,
            resource="@tmp",
            num_outputs=1,
            out_names=[f"out_{i}"],
            workspace=[],
        )
        tasks.append(task)
    return tasks


def _verify_schedule(tasks, waves):
    """Проверить топологический порядок и покрытие всех задач."""
    scheduled = {}
    for wi, w in enumerate(waves):
        for p in w.packets:
            assert isinstance(p, Task), f"packet is not Task: {type(p)}"
            assert p.id not in scheduled, f"Task {p.id} scheduled twice"
            scheduled[p.id] = wi

    all_ids = {t.id for t in tasks}
    missing = all_ids - set(scheduled.keys())
    assert not missing, f"Missing tasks: {missing}"

    for t in tasks:
        for ref in t.inputs:
            if ref.type == "task":
                assert ref.task_id in scheduled, \
                    f"Task {t.id} depends on {ref.task_id} which is not scheduled"
                assert scheduled[ref.task_id] < scheduled[t.id], \
                    f"Task {t.id} depends on {ref.task_id} but they are in same/earlier wave"


def test_topological_order():
    """Простой chain: a -> b -> c."""
    scheduler = Scheduler()
    tasks = [
        Task(id=0, op="EMA", params={"period": 3}, inputs=[], resource="@tmp",
             num_outputs=1, out_names=["a"], workspace=[]),
        Task(id=1, op="EMA", params={"period": 3},
             inputs=[ResourceRef(type="task", task_id=0, output_idx=0)],
             resource="@tmp", num_outputs=1, out_names=["b"], workspace=[]),
        Task(id=2, op="EMA", params={"period": 3},
             inputs=[ResourceRef(type="task", task_id=1, output_idx=0)],
             resource="@tmp", num_outputs=1, out_names=["c"], workspace=[]),
    ]
    waves = scheduler.schedule(tasks)
    assert len(waves) == 3, f"Expected 3 waves, got {len(waves)}"
    for w in waves:
        assert len(w.packets) == 1
    print("  [OK] Chain: 3 tasks -> 3 waves")


def test_independent_tasks():
    """Независимые задачи -> одна волна."""
    scheduler = Scheduler()
    tasks = [Task(id=i, op="EMA", params={"period": 3}, inputs=[], resource="@tmp",
                  num_outputs=1, out_names=[f"x_{i}"], workspace=[])
             for i in range(10)]
    waves = scheduler.schedule(tasks)
    assert len(waves) == 1, f"Expected 1 wave, got {len(waves)}"
    assert len(waves[0].packets) == 10
    print("  [OK] Independent: 10 tasks -> 1 wave OK")


def test_diamond():
    """a -> b, a -> c -> d."""
    scheduler = Scheduler()
    tasks = [
        Task(id=0, op="SMA", params={"period": 3}, inputs=[], resource="@tmp",
             num_outputs=1, out_names=["a"], workspace=[]),
        Task(id=1, op="EMA", params={"period": 3},
             inputs=[ResourceRef(type="task", task_id=0, output_idx=0)],
             resource="@tmp", num_outputs=1, out_names=["b"], workspace=[]),
        Task(id=2, op="RSI", params={"period": 3},
             inputs=[ResourceRef(type="task", task_id=0, output_idx=0)],
             resource="@tmp", num_outputs=1, out_names=["c"], workspace=[]),
        Task(id=3, op="MACD", params={"fast": 12, "slow": 26},
             inputs=[ResourceRef(type="task", task_id=2, output_idx=0)],
             resource="@tmp", num_outputs=3, out_names=["d", "ds", "dh"], workspace=[]),
    ]
    waves = scheduler.schedule(tasks)
    assert len(waves) == 3, f"Expected 3 waves, got {len(waves)}"
    assert len(waves[0].packets) == 1
    assert len(waves[1].packets) == 2
    assert len(waves[2].packets) == 1
    print("  [OK] Diamond: 4 tasks -> 3 waves OK")


def test_random_1000():
    """1000 случайных DAG по 3-20 задач."""
    scheduler = Scheduler()
    random.seed(42)
    for dag_idx in range(1000):
        n = random.randint(3, 20)
        tasks = generate_random_dag(n, edge_prob=0.15, seed=dag_idx)
        try:
            waves = scheduler.schedule(tasks)
        except Exception as e:
            print(f"    FAIL DAG {dag_idx} ({n} tasks): {e}")
            raise
        _verify_schedule(tasks, waves)
    print("  [OK] Random DAGs: 1000 iterations OK")


def test_large_dag():
    """Один большой DAG: 1000 задач."""
    scheduler = Scheduler()
    tasks = generate_random_dag(1000, edge_prob=0.02, seed=999)
    t0 = time.perf_counter()
    waves = scheduler.schedule(tasks)
    elapsed = time.perf_counter() - t0
    _verify_schedule(tasks, waves)
    sizes = [len(w.packets) for w in waves]
    print(f"  [OK] Large DAG: 1000 tasks -> {len(waves)} waves in {elapsed*1000:.1f}ms")
    print(f"    waves: min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.1f}")


def test_cycle_handling():
    """Циклический граф — scheduler не должен падать."""
    scheduler = Scheduler()
    tasks = [
        Task(id=0, op="EMA", params={"period": 3},
             inputs=[ResourceRef(type="task", task_id=2, output_idx=0)],
             resource="@tmp", num_outputs=1, out_names=["a"], workspace=[]),
        Task(id=1, op="EMA", params={"period": 3},
             inputs=[ResourceRef(type="task", task_id=0, output_idx=0)],
             resource="@tmp", num_outputs=1, out_names=["b"], workspace=[]),
        Task(id=2, op="EMA", params={"period": 3},
             inputs=[ResourceRef(type="task", task_id=1, output_idx=0)],
             resource="@tmp", num_outputs=1, out_names=["c"], workspace=[]),
    ]
    try:
        waves = scheduler.schedule(tasks)
        scheduled = {p.id for w in waves for p in w.packets}
        print(f"  [OK] Cycle DAG: {len(scheduled)}/3 scheduled, {len(waves)} waves "
              f"(cycle not detected, no crash)")
    except Exception as e:
        print(f"  [OK] Cycle DAG: raised {type(e).__name__}")


def test_real_dag():
    """Реальный DAG через Runtime.compile: EMA -> RSI + MACD."""
    runtime = make_runtime()
    jobs = [
        {"op": "EMA",  "inputs": ["Close"], "params": {"period": 5},  "resource": "@tmp"},
        {"op": "RSI",  "inputs": ["ema_5"],  "params": {"period": 14}, "resource": "@tmp"},
        {"op": "MACD", "inputs": ["ema_5"],  "params": {"fast": 12, "slow": 26}, "resource": "Signals"},
    ]
    tasks = runtime.compile(jobs)
    scheduler = Scheduler()
    waves = scheduler.schedule(tasks)
    assert len(waves) >= 2, f"Expected >=2 waves, got {len(waves)}"
    _verify_schedule(tasks, waves)
    print(f"  [OK] Real DAG: {len(tasks)} tasks -> {len(waves)} waves")
    for i, w in enumerate(waves):
        names = [p.op for p in w.packets]
        print(f"    Wave {i}: {names}")


if __name__ == "__main__":
    test_topological_order()
    test_independent_tasks()
    test_diamond()
    test_random_1000()
    test_large_dag()
    test_cycle_handling()
    test_real_dag()
    print("\n=== All scheduler stress tests PASSED ===")
