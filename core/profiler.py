"""Runtime Observability — Phase 21.

Инструменты анализа без изменения Runtime ABI.
Содержит:
  - RuntimeProfiler: замеры Planner + Kernel
  - ProfileReport: структурированный отчёт
  - GraphVisualizer: DAG tree + Graphviz
  - OptimizationReport: рекомендации
"""

import time
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
import numpy as np


@dataclass
class ProfileReport:
    """Структурированный отчёт о выполнении.
    
    Fields:
        n_registered: число зарегистрированных операций (до merge).
        n_merged: число Job после merge.
        merge_ratio: коэффициент слияния (n_registered / n_merged).
        n_gpu: число GPU dispatches.
        n_cpu: число CPU fallbacks.
        op_timing: {op_id: total_time_ms}.
        total_time_ms: общее время выполнения Kernel.
        graph: ExecutionGraph (для визуализации).
        gpu_available: bool.
        op_breakdown: список по операциям.
    """
    n_registered: int = 0
    n_merged: int = 0
    merge_ratio: float = 0.0
    n_gpu: int = 0
    n_cpu: int = 0
    op_timing: Dict[str, float] = field(default_factory=dict)
    total_time_ms: float = 0.0
    graph: object = None
    gpu_available: bool = False
    op_breakdown: List[dict] = field(default_factory=list)
    
    def summary(self) -> str:
        """Краткая сводка."""
        lines = [
            "Runtime Profile",
            "--------------",
            f"Operations registered:  {self.n_registered}",
            f"Jobs after merge:       {self.n_merged}",
            f"Merge ratio:            {self.merge_ratio:.1f}x",
            f"GPU dispatches:         {self.n_gpu}",
            f"CPU fallback:           {self.n_cpu}",
            f"GPU available:          {'yes' if self.gpu_available else 'no'}",
            f"Total execution:        {self.total_time_ms:.1f} ms",
            "",
            "Per operation:",
        ]
        for op_id, t_ms in sorted(self.op_timing.items(), key=lambda x: -x[1]):
            pct = t_ms / self.total_time_ms * 100 if self.total_time_ms > 0 else 0
            lines.append(f"  {op_id:20s}  {t_ms:6.1f} ms  ({pct:4.0f}%)")
        return "\n".join(lines)
    
    def text(self) -> str:
        """Полный отчёт."""
        L = '-' * 60
        lines = [
            L,
            "  NumFast Runtime Profile",
            L,
            "",
            f"  Planner",
            f"  {'-' * 40}",
            f"  Jobs created:         {self.n_registered:6d}",
            f"  Jobs merged:          {self.n_registered - self.n_merged:6d}",
            f"  Jobs executed:        {self.n_merged:6d}",
            f"  Merge ratio:          {self.merge_ratio:>6.1f}x",
            "",
            f"  Execution",
            f"  {'-' * 40}",
            f"  GPU dispatches:       {self.n_gpu:6d}",
            f"  CPU fallback:         {self.n_cpu:6d}",
            f"  Total time:           {self.total_time_ms:>6.1f} ms",
            "",
            f"  Per operation:",
            f"  {'-' * 40}",
        ]
        for op_id, t_ms in sorted(self.op_timing.items(), key=lambda x: -x[1]):
            pct = t_ms / self.total_time_ms * 100 if self.total_time_ms > 0 else 0
            lines.append(f"  {op_id:20s}  {t_ms:>6.1f} ms  ({pct:>4.0f}%)")
        
        if self.op_breakdown:
            lines.extend([
                "",
                f"  Job breakdown:",
                f"  {'-' * 40}",
            ])
            for j in self.op_breakdown:
                gpu_tag = " GPU" if j.get('gpu', False) else " CPU"
                lines.append(
                    f"  {j['op_id']:20s}  w={j.get('w', j.get('period', ''))}"
                    f"  {j['time_ms']:>6.1f} ms{gpu_tag}"
                )
        
        lines.extend([
            "",
            L,
        ])
        return "\n".join(lines)


class RuntimeProfiler:
    """Профилировщик Runtime.
    
    Оборачивает CoreAPI и собирает метрики выполнения.
    Не изменяет Runtime ABI — только читает состояние.
    
    Пример:
        api = CoreAPI()
        profiler = RuntimeProfiler(api)
        
        with profiler.session() as session:
            session.sma(close, 14)
            session.ema(close, 20)
        
        print(profiler.report().text())
    """
    
    def __init__(self, api):
        self._api = api
        self._report = None
        self._n_registered = 0
        self._op_timing = defaultdict(float)
        self._op_breakdown = []
        self._n_gpu = 0
        self._n_cpu = 0
        self._gpu_available = False
        self._graphs = []
    
    def session(self):
        """Создать профилировочную сессию."""
        return _ProfilingSession(self)
    
    def profile_call(self, op_id: str, fn: Callable, *args, **kwargs):
        """Профилировать один вызов операции.
        
        Args:
            op_id: идентификатор (sma, ema, ...).
            fn: функция CoreAPI (api.sma, api.ema, ...).
            args/kwargs: аргументы функции.
        
        Returns:
            результат fn.
        """
        self._n_registered += 1
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t = (time.perf_counter() - t0) * 1000
        self._op_timing[op_id] += t
        
        # Определяем GPU/CPU
        kernel = self._api.kernel
        is_gpu = (hasattr(kernel, '_gpu_available') and 
                  kernel._gpu_available())
        if is_gpu:
            self._n_gpu += 1
        else:
            self._n_cpu += 1
        self._gpu_available = is_gpu
        
        self._op_breakdown.append({
            'op_id': op_id,
            'time_ms': round(t, 2),
            'gpu': is_gpu,
        })
        
        return result
    
    def build_report(self, graphs=None) -> ProfileReport:
        """Построить отчёт на основе собранных метрик.
        
        Args:
            graphs: список ExecutionGraph (если None, берёт _graphs).
        """
        if graphs is None:
            graphs = getattr(self, '_graphs', [])
        
        # Calculate totals across all graphs
        total_registered = self._n_registered
        total_merged = sum(len(g.jobs) for g in graphs)
        total_graphs = len(graphs)
        
        merge_ratio = total_registered / total_merged if total_merged > 0 else 0.0
        
        total_ms = sum(self._op_timing.values())
        
        # Combined graph for visualization
        combined_graph = _combine_graphs(graphs) if graphs else None
        
        return ProfileReport(
            n_registered=total_registered,
            n_merged=total_merged,
            merge_ratio=merge_ratio,
            n_gpu=self._n_gpu,
            n_cpu=self._n_cpu,
            op_timing=dict(self._op_timing),
            total_time_ms=total_ms,
            graph=combined_graph,
            gpu_available=self._gpu_available,
            op_breakdown=self._op_breakdown,
        )
    
    def report(self) -> ProfileReport:
        """Вернуть отчёт (перестроить если нужно)."""
        if self._report is None:
            self._report = self.build_report()
        return self._report


class _ProfilingSession:
    """Контекстный менеджер для профилировочной сессии.
    
    Перехватывает вызовы CoreAPI и профилирует их.
    Патчит Planner.build() для захвата ExecutionGraph.
    """
    
    def __init__(self, profiler: 'RuntimeProfiler'):
        self._profiler = profiler
        self._api = profiler._api
        self._original_build = None
    
    def __enter__(self):
        self._profiler._graphs = []
        # Patch Planner.build() to capture ExecutionGraph
        planner = self._api.planner
        self._original_build = planner.build
        def patched_build():
            graph = self._original_build()
            self._profiler._graphs.append(graph)
            return graph
        planner.build = patched_build
        return self
    
    def __exit__(self, *args):
        if self._original_build is not None:
            self._api.planner.build = self._original_build
        graphs = getattr(self._profiler, '_graphs', [])
        self._profiler._report = self._profiler.build_report(graphs)
    
    def __getattr__(self, name):
        """Прокси для CoreAPI методов с профилированием."""
        api_method = getattr(self._api, name, None)
        if api_method is None:
            raise AttributeError(f"CoreAPI has no method '{name}'")
        
        def profiled(*args, **kwargs):
            from numfast.core.nseries import NumericSeries
            converted = tuple(
                NumericSeries(a.copy()) if isinstance(a, np.ndarray) and a.dtype.kind in ('i', 'u')
                else a
                for a in args
            )
            return self._profiler.profile_call(name, api_method, *converted, **kwargs)
        
        return profiled


class GraphVisualizer:
    """Визуализация DAG операций.
    
    Строит дерево зависимостей из ExecutionGraph.
    """
    
    @staticmethod
    def tree(graph) -> str:
        """Text tree of the execution graph.
        
        Args:
            graph: ExecutionGraph (или список Job).
        
        Returns:
            Многострочное дерево.
        """
        jobs = graph.jobs if hasattr(graph, 'jobs') else graph
        if not jobs:
            return "(empty)"
        
        lines = ["Execution Graph", "=============="]
        
        for idx, job in enumerate(jobs):
            op_id = job.op_id
            inputs = job.inputs if hasattr(job, 'inputs') else []
            params = job.params if hasattr(job, 'params') else {}
            
            # Параметры
            param_str = ""
            for k, v in params.items():
                if isinstance(v, list):
                    param_str += f" {k}={v}"
                else:
                    param_str += f" {k}={v}"
            
            lines.append(f"\n  Job #{idx}: {op_id}{param_str}")
            
            vert = "|" if idx < len(jobs) - 1 else " "
            # Inputs
            if inputs:
                lines.append(f"  {vert}  Inputs: {len(inputs)} series")
                for ii, inp in enumerate(inputs):
                    marker = "+--" if ii < len(inputs) - 1 else "+--"
                    lines.append(f"  {vert}  {marker} Series(id={id(inp) % 10000}, n={inp.n if hasattr(inp, 'n') else '?'}, vf={inp.valid_from if hasattr(inp, 'valid_from') else '?'})")
            
            # Outputs
            outputs = job.outputs if hasattr(job, 'outputs') else []
            if outputs:
                lines.append(f"  {vert if inputs else '  '}  Outputs: {len(outputs)} series")
                for oi, out in enumerate(outputs):
                    marker = "+--" if oi < len(outputs) - 1 else "+--"
                    lines.append(f"  {vert if inputs else '  '}  {marker} Series(id={id(out) % 10000}, n={out.n if hasattr(out, 'n') else '?'}, vf={out.valid_from if hasattr(out, 'valid_from') else '?'})")
        
        return "\n".join(lines)
    
    @staticmethod
    def graphviz(graph) -> str:
        """Generate Graphviz DOT source.
        
        Args:
            graph: ExecutionGraph.
        
        Returns:
            DOT format string.
        """
        jobs = graph.jobs if hasattr(graph, 'jobs') else graph
        if not jobs:
            return "digraph G { }"
        
        lines = [
            "digraph RuntimeGraph {",
            '  rankdir="LR";',
            '  node [shape=box, style=rounded];',
        ]
        
        node_ids = {}
        counter = 0
        
        for job in jobs:
            op_id = job.op_id
            params = job.params if hasattr(job, 'params') else {}
            param_str = "\\n".join(f"{k}={v}" for k, v in params.items())
            label = f"{op_id}\\n{param_str}" if param_str else op_id
            
            job_node = f"job_{id(job) % 10000}"
            node_ids[id(job)] = job_node
            lines.append(f'  {job_node} [label="{label}"];')
            
            # Input series → job
            inputs = job.inputs if hasattr(job, 'inputs') else []
            for inp in inputs:
                inp_node = f"in_{id(inp) % 10000}"
                lines.append(f'  {inp_node} [label="Series\\nn={inp.n}", shape=ellipse];')
                lines.append(f'  {inp_node} -> {job_node};')
            
            # Job → output series
            outputs = job.outputs if hasattr(job, 'outputs') else []
            for out in outputs:
                out_node = f"out_{id(out) % 10000}"
                lines.append(f'  {out_node} [label="Result\\nn={out.n}", shape=ellipse];')
                lines.append(f'  {job_node} -> {out_node};')
        
        lines.append("}")
        return "\n".join(lines)


class OptimizationReport:
    """Анализ и рекомендации по оптимизации."""
    
    @staticmethod
    def analyze(profile: ProfileReport) -> str:
        """Сгенерировать отчёт об оптимизациях.
        
        Args:
            profile: ProfileReport.
        
        Returns:
            Текстовый отчёт с рекомендациями.
        """
        lines = [
            "Optimization Report",
            "==================",
        ]
        
        # 1. Merge анализ
        if profile.n_registered > 0:
            saved = profile.n_registered - profile.n_merged
            lines.append(f"\n  + Merged {saved} duplicate operations "
                        f"({profile.merge_ratio:.1f}x reduction)")
        
        # 2. GPU анализ
        if profile.gpu_available:
            if profile.n_gpu > 0:
                lines.append(f"  + {profile.n_gpu} GPU dispatches successful")
            else:
                lines.append(f"  ! GPU available but not used")
        else:
            lines.append(f"  i CPU-only execution (GPU not available)")
        
        # 3. CPU fallbacks
        if profile.n_cpu > 0 and profile.gpu_available:
            lines.append(f"  ! {profile.n_cpu} operations fell back to CPU "
                        f"(no WGSL implementation yet)")
        
        # 4. Per-operation анализ
        if profile.op_timing:
            total = profile.total_time_ms
            lines.append(f"\n  Per-operation analysis:")
            for op_id, t_ms in sorted(profile.op_timing.items(), 
                                       key=lambda x: -x[1]):
                pct = t_ms / total * 100 if total > 0 else 0
                bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
                lines.append(f"  {op_id:20s} {bar} {t_ms:6.1f} ms ({pct:4.0f}%)")
        
        # 5. Рекомендации
        lines.extend([
            "",
            f"  Recommendations",
            f"  {'-' * 40}",
        ])
        
        suggestions = []
        
        # Проверка на сходные операции, которые можно совместить
        op_names = list(profile.op_timing.keys())
        if 'sma' in op_names and 'ema' in op_names:
            suggestions.append(
                "  - SMA and EMA can share close input -- already merged by Planner"
            )
        
        if 'roc' in op_names and 'sma' in op_names:
            suggestions.append(
                "  - ROC eligible for pointwise fusion with SMA prefix"
            )
        
        # Проверка на GPU-потенциал
        if not profile.gpu_available:
            suggestions.append(
                "  - Enable WebGPU driver for potential speedup on large datasets (>1M rows)"
            )
        elif profile.n_cpu > 0:
            suggestions.append(
                f"  - Port {profile.n_cpu} CPU operations to WGSL to eliminate fallbacks"
            )
        
        if not suggestions:
            suggestions.append("  - No optimization opportunities detected")
        
        lines.extend(suggestions)
        lines.extend(["", "=" * 50])
        
        return "\n".join(lines)


def _combine_graphs(graphs):
    """Combine multiple ExecutionGraphs into one for visualization."""
    if not graphs:
        return None
    if len(graphs) == 1:
        return graphs[0]
    
    # Create a combined graph with all jobs
    from numfast.core.executor import ExecutionGraph
    combined = ExecutionGraph()
    seen = set()
    for g in graphs:
        for job in g.jobs:
            # Deduplicate by (op_id, input_ids, params_str)
            input_ids = tuple(id(s) for s in (job.inputs or []))
            params_str = str(sorted(job.params.items())) if job.params else ""
            key = (job.op_id, input_ids, params_str)
            if key not in seen:
                seen.add(key)
                combined.add(job)
    return combined


def profile(api, fn, *args, **kwargs):
    """Удобная функция: запустить fn через Profiler и вернуть (result, report).
    
    Пример:
        api = CoreAPI()
        result, report = profile(api, api.sma, close, 14)
        print(report.text())
    """
    profiler = RuntimeProfiler(api)
    with profiler.session() as session:
        result = fn(*args, **kwargs)
    return result, profiler.report()
