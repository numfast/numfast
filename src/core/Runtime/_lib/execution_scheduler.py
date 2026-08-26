"""ExecutionScheduler — стратегии исполнения: chunking / fusion / обычный путь.

Проблема (chunking):
  WebGPU имеет лимит maxComputeWorkgroupsPerDimension. Builder считает
  dispatch = ceil(N / 64). При N больше лимита драйвера dispatch превышает
  допустимый — WebGPU падает с GPUValidationError. Фактический лимит
  берётся из driver.max_dispatch_elements() — планировщик не знает чисел.

Решение (chunking):
  Если все kernels графа помечены "chunkable": True в capabilities
  (kernel_table[kernel]["capabilities"]["chunkable"]), входные данные
  режутся на чанки <= лимита, каждый чанк исполняется отдельным
  dispatch, результаты пишутся прямо в предвыделенные полные массивы
  (подмена view output_buffers на BlockView(full[offset:end])) и
  сохраняются через driver.store_output().

Решение (fusion):
  Если драйвер имеет execute_fused() и граф даёт цепочку >1 пакета
  при размере <= лимита — цепочка исполняется одним проходом:
  промежуточные буферы остаются на GPU (без readback), финальные выходы
  (чей raw не потребляется как вход ни одним пакетом) readback'ятся и
  сохраняются через driver.store_output().

  Для остальных графов execute() возвращает None — Runtime.execute идёт
  обычным путём (builder + execute_wave), поведение не меняется.
"""

from typing import Optional

import numpy as np

from .builder import builder
from .mod_iface import BlockView
from .scheduler import ScheduleWave


class ExecutionScheduler:
    """Планировщик исполнения: chunking / fusion / обычный путь."""

    def execute(self, graph, source_data: dict, driver, kernel_table: dict,
                max_elements: Optional[int] = None) -> Optional[list]:
        """Выполнить граф, выбрав стратегию: chunking / fusion / обычный путь.

        Стратегии (порядок решения):
        1. Chunking — все узлы chunkable и размер > лимита: данные режутся
           на чанки <= лимита, каждый чанк исполняется отдельным dispatch,
           результаты пишутся в предвыделенные полные массивы и
           сохраняются через driver.store_output().
        2. Fusion — драйвер имеет execute_fused() и граф даёт >1 пакета
           при размере <= лимита (или без лимита): цепочка исполняется
           одним проходом, промежуточные буферы остаются на GPU, финальные
           выходы readback'ятся и сохраняются через store_output().
        3. Обычный путь — во всех остальных случаях: возвращает None,
           Runtime.execute идёт через builder + execute_wave.

        Args:
            graph: ExecutionGraph после planner/optimize_graph
            source_data: dict имя_колонки -> numpy массив (raw inputs)
            driver: Driver (WebGPU поддерживает chunking и fusion,
                остальные — нет)
            kernel_table: реестр ядер
            max_elements: принудительный лимит размера чанка
                (None = авто: driver.max_dispatch_elements(),
                иначе без чанкования)

        Returns:
            list[ExecutionPacket] последнего чанка (chunking) или всей
            цепочки (fusion), если граф исполнен планировщиком,
            иначе None — Runtime.execute использует обычный путь.
        """
        if not graph.nodes:
            return None

        size = self._data_size(graph, source_data)
        if size is None:
            return None

        limit = self._resolve_limit(driver, max_elements)

        # --- Стратегия 1: чанкование больших elementwise-графов ---
        if (limit is not None and size > limit
                and self._all_chunkable(graph, kernel_table)):
            return self._execute_chunked(
                graph, source_data, driver, kernel_table, size, limit
            )

        # --- Стратегия 2: fusion цепочки на драйвере с execute_fused ---
        if (hasattr(driver, "execute_fused")
                and (limit is None or size <= limit)):
            packets = builder(graph, source_data, driver, kernel_table)
            if len(packets) > 1:
                driver.execute_fused(packets)
                self._store_final_outputs(graph, packets, driver)
                return packets

        # --- Стратегия 3: обычный путь ---
        return None

    @staticmethod
    def _all_chunkable(graph, kernel_table: dict) -> bool:
        """Все ли узлы графа помечены chunkable в capabilities."""
        for node in graph.nodes:
            caps = kernel_table.get(node.kernel_id, {}).get("capabilities", {})
            if not caps.get("chunkable", False):
                return False
        return True

    def _execute_chunked(self, graph, source_data: dict, driver,
                         kernel_table: dict, size: int,
                         limit: int) -> Optional[list]:
        """Исполнить граф по чанкам, сохранить полные результаты.

        Возвращает ExecutionPackets последнего чанка (для совместимости
        сигнатуры), иначе None.
        """
        # out_name -> (node.id, output_idx) — для финального store_output
        out_map: dict[str, tuple[int, int]] = {}
        for node in graph.nodes:
            for i, out in enumerate(node.outputs):
                out_map[out.name] = (node.id, i)

        # Предвыделенные полные выходы: readback пишет прямо в full[срез]
        full_buffers: dict[str, np.ndarray] = {
            name: np.zeros(size, dtype=np.float64) for name in out_map
        }
        last_packets: Optional[list] = None

        for start in range(0, size, limit):
            end = min(start + limit, size)
            chunk_source = {
                name: arr[start:end] for name, arr in source_data.items()
            }
            chunk_packets = builder(graph, chunk_source, driver, kernel_table)
            # Подмена view: результат чанка пишется в полный массив
            for packet, node in zip(chunk_packets, graph.nodes):
                for j, bv in enumerate(packet.output_buffers):
                    out_name = node.outputs[j].name
                    bv.view = BlockView(full_buffers[out_name][start:end])
            for packet in chunk_packets:
                driver.execute_wave(ScheduleWave(packets=[packet]))
            last_packets = chunk_packets

        for name, (node_id, output_idx) in out_map.items():
            driver.store_output(name, node_id, output_idx, full_buffers[name])

        return last_packets

    @staticmethod
    def _store_final_outputs(graph, packets: list, driver) -> None:
        """Сохранить финальные выходы fused-цепочки через store_output().

        Финальные = выходы, чей raw не потребляется как вход ни одним
        пакетом (промежуточные остаются на GPU и в numpy-заглушках).
        """
        consumed: set[tuple[int, int]] = set()
        for node in graph.nodes:
            for pr in node.inputs:
                if isinstance(pr.source, int):
                    consumed.add((pr.source, pr.port if isinstance(pr.port, int) else 0))

        by_id = {node.id: packet for node, packet in zip(graph.nodes, packets)}
        for node in graph.nodes:
            for i, out in enumerate(node.outputs):
                if (node.id, i) in consumed:
                    continue
                raw = by_id[node.id].output_buffers[i].view._raw
                driver.store_output(out.name, node.id, i, raw)

    @staticmethod
    def _data_size(graph, source_data: dict) -> Optional[int]:
        """Размер данных N: длина первого @input первого узла графа.

        Returns:
            int N, если в графе есть внешний вход с данными, иначе None.
        """
        for node in graph.nodes:
            for pr in node.inputs:
                if pr.source == "@input":
                    arr = source_data.get(str(pr.port))
                    if arr is None:
                        return None
                    return len(arr)
        return None

    @staticmethod
    def _resolve_limit(driver, max_elements: Optional[int]) -> Optional[int]:
        """Лимит размера чанка.

        Приоритет:
        1. Явный max_elements (принудительное чанкование, для тестов).
        2. driver.max_dispatch_elements() — если драйвер объявляет лимит.
        3. None — без чанкования (CPU и другие драйверы).

        Магических чисел в планировщике нет: лимит принадлежит драйверу.
        """
        if max_elements is not None:
            return int(max_elements)
        max_fn = getattr(driver, "max_dispatch_elements", None)
        return max_fn() if callable(max_fn) else None
