"""Runtime Profiler — встроенная инструментация Runtime.

Замеряет время ключевых фаз:
  - Compile
  - Scheduler
  - Memory alloc
  - Kernel execute
  - Driver execute

Использование:

    profiler = RuntimeProfiler()
    with profiler.phase("compile"):
        tasks = runtime.compile(jobs)
    with profiler.phase("execute"):
        runtime.execute(tasks, data)
    profiler.print_report()
"""

import time
from collections import defaultdict


class PhaseTimer:
    """Замер времени одной фазы."""

    def __init__(self, name):
        self.name = name
        self.elapsed = 0.0
        self.calls = 0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._start is not None:
            self.elapsed += time.perf_counter() - self._start
            self.calls += 1
            self._start = None

    @property
    def avg(self):
        return self.elapsed / self.calls if self.calls > 0 else 0.0


class RuntimeProfiler:
    """Профайлер Runtime.

    Собирает тайминги по фазам и выдаёт табличный отчёт.
    """

    def __init__(self):
        self._phases = defaultdict(lambda: PhaseTimer(None))
        self._enabled = True

    def phase(self, name: str) -> PhaseTimer:
        """Контекстный менеджер для замера фазы.

        Usage:
            with profiler.phase("compile"):
                do_work()
        """
        timer = self._phases[name]
        timer.name = name
        timer._start = time.perf_counter()

        class _Ctx:
            def __enter__(ctx_self):
                return timer
            def __exit__(ctx_self, *args):
                if timer._start is not None:
                    timer.elapsed += time.perf_counter() - timer._start
                    timer.calls += 1
                    timer._start = None
        return _Ctx()

    def reset(self):
        """Сбросить все тайминги."""
        self._phases.clear()

    def to_dict(self) -> dict:
        """Вернуть словарь {фаза: {elapsed, calls, avg}}."""
        return {
            name: {
                "elapsed": t.elapsed,
                "calls": t.calls,
                "avg": t.avg,
            }
            for name, t in sorted(self._phases.items())
            if t.calls > 0
        }

    def print_report(self, sort_by="elapsed"):
        """Напечатать таблицу.

        Args:
            sort_by: "elapsed" | "calls" | "avg" | "name"
        """
        items = [(n, t) for n, t in self._phases.items() if t.calls > 0]
        if not items:
            print("(no profiling data)")
            return

        key_map = {
            "elapsed": lambda x: x[1].elapsed,
            "calls": lambda x: x[1].calls,
            "avg": lambda x: x[1].avg,
            "name": lambda x: x[0],
        }
        items.sort(key=key_map.get(sort_by, key_map["elapsed"]), reverse=(sort_by != "name"))

        total = sum(t.elapsed for _, t in items)

        sep = "-" * 70
        print(f"\n{sep}")
        print(f"  Runtime Profiler Report")
        print(f"{sep}")
        print(f"  {'Phase':<20} {'Total (s)':<12} {'Calls':<8} {'Avg (ms)':<10} {'%':<8}")
        print(f"{sep}")
        for name, t in items:
            pct = (t.elapsed / total * 100) if total > 0 else 0
            print(f"  {name:<20} {t.elapsed:<12.4f} {t.calls:<8} {t.avg*1000:<10.4f} {pct:<8.1f}")
        print(f"{sep}")
        print(f"  {'TOTAL':<20} {total:<12.4f} {'':<8} {'':<10} {100.0:<8.1f}")
        print(f"{sep}\n")


# Singleton for easy import
_profiler = RuntimeProfiler()


def get_profiler() -> RuntimeProfiler:
    """Вернуть глобальный профайлер."""
    return _profiler


def phase(name: str) -> PhaseTimer:
    """Контекстный менеджер для глобального профайлера."""
    return _profiler.phase(name)
