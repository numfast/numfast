"""Base utilities for benchmarks."""

import time
from typing import Dict, List

class BenchmarkTimer:
    """Simple timer for benchmark stages."""

    def __init__(self):
        self._start = time.perf_counter()
        self._laps = []
        self._labels = []

    def lap(self, label: str):
        """Record a lap. Returns elapsed since last lap."""
        now = time.perf_counter()
        elapsed = now - self._start
        self._laps.append(elapsed)
        self._labels.append(label)

    def reset(self):
        self._start = time.perf_counter()
        self._laps = []
        self._labels = []

    def results(self) -> Dict[str, float]:
        """Return dict of label -> elapsed_seconds."""
        prev = 0.0
        result = {}
        for i, label in enumerate(self._labels):
            result[label] = self._laps[i] - prev
            prev = self._laps[i]
        return result


def format_bars_per_sec(n_bars: float, elapsed_s: float) -> str:
    """Format bars/sec with appropriate unit."""
    if elapsed_s <= 0:
        return "N/A"
    rate = n_bars / elapsed_s
    if rate >= 1_000_000:
        return f"{rate/1_000_000:.2f} M bars/s"
    elif rate >= 1_000:
        return f"{rate/1_000:.2f} K bars/s"
    else:
        return f"{rate:.2f} bars/s"


def format_exprs_per_sec(n_exprs: float, elapsed_s: float) -> str:
    """Format expressions/sec."""
    if elapsed_s <= 0:
        return "N/A"
    rate = n_exprs / elapsed_s
    if rate >= 1_000_000:
        return f"{rate/1_000_000:.2f} M expr/s"
    elif rate >= 1_000:
        return f"{rate/1_000:.2f} K expr/s"
    else:
        return f"{rate:.2f} expr/s"


class ResultTable:
    """Simple ASCII table for benchmark results."""

    def __init__(self, headers: List[str]):
        self.headers = headers
        self.rows = []

    def add_row(self, values: List[str]):
        self.rows.append(values)

    def render(self) -> str:
        # Calculate column widths
        col_widths = [len(h) for h in self.headers]
        for row in self.rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        # Build separator
        sep = "-+-".join("-" * w for w in col_widths)
        fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)

        lines = []
        lines.append(fmt.format(*self.headers))
        lines.append(sep)
        for row in self.rows:
            lines.append(fmt.format(*row))
        return "\n".join(lines)


def format_profile(profile: Dict[str, float]) -> str:
    """Format profile dict as aligned lines, sorted by time desc."""
    items = sorted(profile.items(), key=lambda x: -x[1])
    total = sum(v for _, v in items)
    lines = ["Profile breakdown:"]
    for label, sec in items:
        pct = (sec / total * 100) if total > 0 else 0
        lines.append(f"  {label:20s}: {sec*1000:8.3f} ms  ({pct:5.1f}%)")
    lines.append(f"  {'total':20s}: {total*1000:8.3f} ms")
    return "\n".join(lines)
