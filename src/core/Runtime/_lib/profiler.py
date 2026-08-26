"""ExecutionProfiler — профилирование на уровне ExecutionPacket.

Не знает про RuntimeContext, SMA, EMA, RSI.
Профилирует ЛЮБОЙ kernel: EMA, FFT, MatMul, RollingSum — одинаково.

Driver заполняет exec_time_ns в packet.profile.
Profiler агрегирует и выдаёт отчёт.

Usage:
    profiler = ExecutionProfiler()
    driver = CpuDriver(profiler=profiler)
    ...
    profiler.print_report()
"""

import time
from collections import defaultdict


class ExecutionProfiler:
    """Собирает профили ExecutionPackets от Driver."""

    def __init__(self):
        self.records: list[dict] = []
        self._enabled = True
        self._start_time = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def on_packet_start(self, packet):
        """Вызывается Driver перед execute()."""
        if not self._enabled:
            return
        packet.profile["exec_start_ns"] = time.perf_counter_ns()

    def on_packet_finish(self, packet):
        """Вызывается Driver после execute()."""
        if not self._enabled:
            return
        packet.profile["exec_end_ns"] = time.perf_counter_ns()
        packet.profile["exec_time_ns"] = (
            packet.profile["exec_end_ns"] - packet.profile["exec_start_ns"]
        )

        self.records.append({
            "kernel": packet.kernel,
            "input_bytes": packet.profile["input_bytes"],
            "output_bytes": packet.profile["output_bytes"],
            "workspace_bytes": packet.profile["workspace_bytes"],
            "total_bytes": packet.profile["total_bytes"],
            "dispatch_x": packet.profile["dispatch_x"],
            "dispatch_y": packet.profile["dispatch_y"],
            "dispatch_z": packet.profile["dispatch_z"],
            "build_ns": packet.profile["build_ns"],
            "exec_time_ns": packet.profile["exec_time_ns"],
            "exec_time_ms": packet.profile["exec_time_ns"] / 1_000_000,
        })

    @property
    def total_dispatches(self) -> int:
        return len(self.records)

    @property
    def total_time_ms(self) -> float:
        return sum(r["exec_time_ms"] for r in self.records)

    @property
    def total_bytes_moved(self) -> int:
        return sum(r["total_bytes"] for r in self.records)

    def summary(self) -> dict:
        """Агрегировать по kernel."""
        agg = defaultdict(lambda: {
            "count": 0, "time_ms": 0,
            "in_bytes": 0, "out_bytes": 0, "ws_bytes": 0,
        })
        for r in self.records:
            k = r["kernel"]
            agg[k]["count"] += 1
            agg[k]["time_ms"] += r["exec_time_ms"]
            agg[k]["in_bytes"] += r["input_bytes"]
            agg[k]["out_bytes"] += r["output_bytes"]
            agg[k]["ws_bytes"] += r["workspace_bytes"]
        return dict(agg)

    def fusion_candidates(self) -> list[tuple[str, str, int]]:
        """Найти пары последовательных ядер с одинаковым размером данных."""
        candidates = []
        for i in range(len(self.records) - 1):
            a = self.records[i]
            b = self.records[i + 1]
            if a["output_bytes"] == b["input_bytes"] and a["output_bytes"] > 0:
                candidates.append((a["kernel"], b["kernel"], a["output_bytes"]))
        return candidates

    def print_report(self):
        """Форматированный отчёт."""
        lines = []
        lines.append("=" * 130)
        lines.append("EXECUTION PROFILER REPORT")
        lines.append(f"  Total dispatches: {self.total_dispatches}")
        lines.append(f"  Total exec time: {self.total_time_ms:.2f} ms")
        lines.append(f"  Total bytes moved: {self.total_bytes_moved / (1024*1024):.2f} MB")
        lines.append("=" * 130)

        h = f"{'Kernel':<25} {'Count':>6} {'Time(ms)':>10} {'In(MB)':>10} {'Out(MB)':>10} {'WS(MB)':>10} {'Avg(ms)':>10}"
        lines.append(h)
        lines.append("-" * 130)

        summary = self.summary()
        for k, v in sorted(summary.items(), key=lambda x: -x[1]["time_ms"]):
            avg = v["time_ms"] / v["count"] if v["count"] > 0 else 0
            lines.append(
                f"{k:<25} {v['count']:>6} {v['time_ms']:>10.2f} "
                f"{v['in_bytes']/1024/1024:>10.4f} {v['out_bytes']/1024/1024:>10.4f} "
                f"{v['ws_bytes']/1024/1024:>10.4f} {avg:>10.4f}"
            )

        lines.append("=" * 130)

        fc = self.fusion_candidates()
        if fc:
            lines.append(f"\nFUSION CANDIDATES ({len(fc)} pairs):")
            lines.append("-" * 80)
            for a, b, sz in fc:
                lines.append(f"  {a:<20} -> {b:<20}  ({sz/1024:.1f} KB shared)")
            lines.append("-" * 80)

        print("\n".join(lines))

    def reset(self):
        self.records.clear()


# Old RuntimeProfiler kept for compile-phase timing only
class PhaseTimer:
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
    def __init__(self):
        self._phases = defaultdict(lambda: PhaseTimer(None))
        self._enabled = True

    def phase(self, name: str):
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
        self._phases.clear()

    def to_dict(self):
        return {
            name: {"elapsed": t.elapsed, "calls": t.calls, "avg": t.avg}
            for name, t in sorted(self._phases.items())
            if t.calls > 0
        }

    def print_report(self, sort_by="elapsed"):
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
