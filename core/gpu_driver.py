"""GPU Driver — Mock для Phase 14.1 (Driver Conformance).

Реализует тот же интерфейс, что CpuDriver, но:
  - идентифицирует себя как "mock_gpu"
  - хранит ключ shadow-буфера на самом объекте серии
    (предотвращает коллизии id() после GC)

Конституция I4: Driver не знает операции.
"""

from numfast.core.nseries import NumericSeries

# Имя атрибута для ключа shadow-буфера
_KEY_ATTR = "_mock_gpu_shadow_key"


class MockGpuDriver:
    """Mock GPU Driver.

    Хранит теневые буферы только для серий, созданных через allocate().
    Ключ буфера сохраняется как атрибут на объекте NumericSeries,
    что решает проблему переиспользования id() после GC.
    """

    def __init__(self):
        self.name = "mock_gpu"
        self._shadows: dict[int, list[int]] = {}
        self._next_key: int = 0

    def _shadow_key(self, series: NumericSeries) -> int | None:
        """Вернуть ключ теневого буфера, если серия создана allocate()."""
        return getattr(series, _KEY_ATTR, None)

    def read(self, series: NumericSeries, i: int) -> int:
        """Прочитать логическое значение."""
        assert 0 <= i < series.n, (
            f"MockGpuDriver.read: index {i} out of range [0, {series.n})"
        )
        val = series.read(i)
        key = self._shadow_key(series)
        if key is not None:
            shadow = self._shadows[key]
            assert shadow[i] == val, (
                f"MockGpuDriver: shadow mismatch at series[{i}]: "
                f"shadow={shadow[i]}, series={val}"
            )
        return val

    def write(self, series: NumericSeries, i: int, value: int) -> None:
        """Записать логическое значение."""
        assert 0 <= i < series.n, (
            f"MockGpuDriver.write: index {i} out of range [0, {series.n})"
        )
        series.write(i, value)
        key = self._shadow_key(series)
        if key is not None:
            self._shadows[key][i] = value

    def allocate(
        self,
        n: int,
        meta=None,
        offset: int = 0,
        scale: int = 1,
        valid_from: int = 0,
    ) -> NumericSeries:
        """Выделить память под Series."""
        if meta is not None:
            if hasattr(meta, 'offset'):
                offset = meta.offset
                scale = meta.scale
            else:
                offset = meta.get('offset', 0)
                scale = meta.get('scale', 1)
        series = NumericSeries(
            data=[1] * n,  # Storage Invariant: min=1
            offset=offset,
            scale=scale,
            valid_from=valid_from,
        )
        key = self._next_key
        self._next_key += 1
        self._shadows[key] = [1] * n
        setattr(series, _KEY_ATTR, key)
        return series

    def dispatch(self, kernel: str, bindings: dict, workgroups: tuple) -> None:
        """Запустить kernel (mock — no-op)."""
        pass

    def barrier(self) -> None:
        """Синхронизация (mock — no-op)."""
        pass

    def __repr__(self) -> str:
        return f"MockGpuDriver(shadow_buffers={len(self._shadows)})"


# ── WebGpuDriver (Phase 14.2) ──────────────────────────────────

_GPU_KEY_ATTR = "_gpu_buf_key"


class WebGpuDriver:
    """Real WebGPU Driver — storage buffers + WGSL dispatch.

    Реализует тот же интерфейс, что CpuDriver, но использует
    wgpu storage buffers под капотом.

    read() → из CPU shadow (NumericSeries)
    write() → в CPU shadow + upload в GPU buffer
    dispatch() → компилирует WGSL, запускает compute shader

    Конституция I4: Driver не знает операции.
    """

    def __init__(self):
        self.name = "webgpu"
        self._device = None  # lazy init
        self._buffers: dict[int, dict] = {}
        self._next_key: int = 0
        self._pipeline_cache: dict[str, object] = {}   # wgsl_hash → (shader_module, pipeline)
        self._bind_group_cache: dict[str, object] = {}  # key → bind_group

    def _ensure_device(self):
        if self._device is None:
            from numfast.core.gpu.device import get_device
            self._device = get_device()

    def _gpu_key(self, series: NumericSeries) -> int | None:
        return getattr(series, _GPU_KEY_ATTR, None)

    def read(self, series: NumericSeries, i: int) -> int:
        """Прочитать логическое значение из CPU shadow."""
        return series.read(i)

    def write(self, series: NumericSeries, i: int, value: int) -> None:
        """Записать: в CPU shadow + upload в GPU буфер."""
        series.write(i, value)
        key = self._gpu_key(series)
        if key is not None:
            import numpy as np
            buf = self._buffers[key]
            data = np.int32(value)
            self._device.queue.write_buffer(buf['gpu'], i * 4, data.tobytes())

    def allocate(
        self,
        n: int,
        meta=None,
        offset: int = 0,
        scale: int = 1,
        valid_from: int = 0,
    ) -> NumericSeries:
        """Выделить память: CPU shadow + GPU storage buffer."""
        self._ensure_device()

        if meta is not None:
            if hasattr(meta, 'offset'):
                offset = meta.offset
                scale = meta.scale
            else:
                offset = meta.get('offset', 0)
                scale = meta.get('scale', 1)

        # CPU shadow
        series = NumericSeries(
            data=[1] * n,
            offset=offset,
            scale=scale,
            valid_from=valid_from,
        )

        key = self._next_key
        self._next_key += 1

        # GPU storage buffer
        import wgpu
        gpu_buf = self._device.create_buffer(
            size=n * 4,  # i32 = 4 bytes per element
            usage=wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_DST
                | wgpu.BufferUsage.COPY_SRC,
        )

        # Upload initial values (min=1 per Storage Invariant)
        import numpy as np
        init_data = np.array([1] * n, dtype=np.int32)
        self._device.queue.write_buffer(gpu_buf, 0, init_data.tobytes())

        self._buffers[key] = {'gpu': gpu_buf, 'n': n}
        setattr(series, _GPU_KEY_ATTR, key)
        return series

    def attach_buffer(self, series: NumericSeries) -> int:
        """Attach GPU storage buffer to existing series (no CPU shadow creation).

        Used when series was created externally (e.g. by user via NumericSeries())
        but needs GPU access for dispatch.

        Args:
            series: existing NumericSeries with data.

        Returns:
            GPU buffer key.
        """
        key = self._next_key
        self._next_key += 1

        import wgpu
        n = series.n
        gpu_buf = self._device.create_buffer(
            size=n * 4,
            usage=wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_DST
                | wgpu.BufferUsage.COPY_SRC,
        )

        self._buffers[key] = {'gpu': gpu_buf, 'n': n}
        setattr(series, _GPU_KEY_ATTR, key)
        return key

    def has_buffer(self, series: NumericSeries) -> bool:
        """Check if series has a GPU buffer attached."""
        return getattr(series, _GPU_KEY_ATTR, None) is not None

    def dispatch(
        self,
        shader_source: str,
        bindings: dict[int, object],
        workgroups: tuple[int, ...],
    ) -> None:
        """Запустить WGSL compute shader с кэшированием pipeline."""
        self._ensure_device()

        # Cache pipeline by WGSL content hash
        import hashlib
        shader_hash = hashlib.md5(shader_source.encode()).hexdigest()
        if shader_hash not in self._pipeline_cache:
            shader = self._device.create_shader_module(code=shader_source)
            pipeline = self._device.create_compute_pipeline(
                layout="auto",
                compute={"module": shader, "entry_point": "main"},
            )
            self._pipeline_cache[shader_hash] = (shader, pipeline)
        else:
            _, pipeline = self._pipeline_cache[shader_hash]

        # Cache bind group by (pipeline_ptr, tuple_of_buf_ptrs)
        buf_ids = tuple(str(id(b)) for b in bindings.values())
        bg_key = f"{id(pipeline)}_{'_'.join(buf_ids)}"
        if bg_key not in self._bind_group_cache:
            entries = [
                {"binding": b, "resource": buf}
                for b, buf in bindings.items()
            ]
            bind_group = self._device.create_bind_group(
                layout=pipeline.get_bind_group_layout(0),
                entries=entries,
            )
            self._bind_group_cache[bg_key] = bind_group
        else:
            bind_group = self._bind_group_cache[bg_key]

        commands = self._device.create_command_encoder()
        pass_enc = commands.begin_compute_pass()
        pass_enc.set_pipeline(pipeline)
        pass_enc.set_bind_group(0, bind_group)
        pass_enc.dispatch_workgroups(*workgroups)
        pass_enc.end()
        self._device.queue.submit([commands.finish()])

    def upload_batch(self, buf_key: int, data: bytes) -> None:
        """Batch upload: write entire buffer content from bytes."""
        buf = self._buffers[buf_key]['gpu']
        self._device.queue.write_buffer(buf, 0, data)

    def read_batch(self, buf_key: int) -> bytes:
        """Batch readback: read entire buffer content."""
        buf = self._buffers[buf_key]['gpu']
        return self._device.queue.read_buffer(buf)

    def barrier(self) -> None:
        """Flush."""
        self._ensure_device()
        self._device.queue.submit([])

    def __repr__(self) -> str:
        return f"WebGpuDriver(buffers={len(self._buffers)})"
