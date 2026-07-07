"""Phase 14.2.2 — WebGpuDriver + WGSL dispatch.

Проверяет:
  - allocate() → GPU storage buffer
  - write() → upload на GPU
  - dispatch() → WGSL compute shader
  - readback с GPU (через queue.read_buffer)

Запуск:
    python -m pytest numfast/core/tests/test_webgpu_driver.py -v --timeout=30
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import wgpu
import numpy as np
import pytest

from numfast.core.nseries import NumericSeries
from numfast.core.gpu_driver import WebGpuDriver
from numfast.core.gpu.shaders import SHADERS, COPY_WGSL


@pytest.fixture
def driver():
    return WebGpuDriver()


class TestWebGpuDriver:
    """WebGpuDriver: StorageBuffer + Dispatch."""

    def test_allocate_creates_gpu_buffer(self, driver):
        """allocate() создаёт GPU storage buffer."""
        s = driver.allocate(64)
        key = getattr(s, '_gpu_buf_key', None)
        assert key is not None
        assert key in driver._buffers
        assert driver._buffers[key]['n'] == 64

    def test_write_uploads_to_gpu(self, driver):
        """write() → upload значения в GPU буфер."""
        s = driver.allocate(10)
        driver.write(s, 0, 42)
        driver.write(s, 5, 99)

        device = driver._device
        key = driver._gpu_key(s)
        gpu_buf = driver._buffers[key]['gpu']

        # Read back from GPU
        result = device.queue.read_buffer(gpu_buf)
        data = np.frombuffer(result, dtype=np.int32)
        assert data[0] == 42, f"GPU[0]={data[0]}, expected 42"
        assert data[5] == 99, f"GPU[5]={data[5]}, expected 99"

    def test_dispatch_copy(self, driver):
        """dispatch(copy) копирует input → output на GPU."""
        n = 128
        inp = driver.allocate(n)
        out = driver.allocate(n)

        # Write input data
        for i in range(n):
            driver.write(inp, i, i + 10)

        device = driver._device
        key_in = driver._gpu_key(inp)
        key_out = driver._gpu_key(out)
        buf_in = driver._buffers[key_in]['gpu']
        buf_out = driver._buffers[key_out]['gpu']

        # Dispatch copy shader
        driver.dispatch(
            COPY_WGSL,
            {0: buf_in, 1: buf_out},
            ((n + 255) // 256,),
        )

        # Read back output from GPU
        result = device.queue.read_buffer(buf_out)
        data = np.frombuffer(result, dtype=np.int32)
        expected = np.array([i + 10 for i in range(n)], dtype=np.int32)
        np.testing.assert_array_equal(data, expected)

    def test_dispatch_add_one(self, driver):
        """dispatch(add_one) → y = x + 1."""
        n = 256
        inp = driver.allocate(n)
        out = driver.allocate(n)

        for i in range(n):
            driver.write(inp, i, i)

        device = driver._device
        buf_in = driver._buffers[driver._gpu_key(inp)]['gpu']
        buf_out = driver._buffers[driver._gpu_key(out)]['gpu']

        driver.dispatch(
            SHADERS["add_one"],
            {0: buf_in, 1: buf_out},
            ((n + 255) // 256,),
        )

        result = device.queue.read_buffer(buf_out)
        data = np.frombuffer(result, dtype=np.int32)
        expected = np.array([i + 1 for i in range(n)], dtype=np.int32)
        np.testing.assert_array_equal(data, expected)

    def test_write_and_read_consistency(self, driver):
        """read() из CPU shadow совпадает с тем, что на GPU."""
        s = driver.allocate(50)
        for i in range(50):
            driver.write(s, i, i * 3 + 7)

        # Read from CPU shadow
        for i in range(50):
            assert driver.read(s, i) == i * 3 + 7, f"CPU shadow mismatch at {i}"

        # Verify GPU buffer matches
        device = driver._device
        key = driver._gpu_key(s)
        gpu_buf = driver._buffers[key]['gpu']
        result = device.queue.read_buffer(gpu_buf)
        data = np.frombuffer(result, dtype=np.int32)
        expected = np.array([i * 3 + 7 for i in range(50)], dtype=np.int32)
        np.testing.assert_array_equal(data, expected)


# ══════════════════════════════════════════════════════════════════════
# Phase 14.2.5 — SMA WGSL (Algorithm proven on GPU)
# ══════════════════════════════════════════════════════════════════════

class TestSmaWgsl:
    """SMA через WGSL compute shader vs CpuDriver — bit-exact.

    Доказывает: один алгоритм, два исполнителя, одинаковый результат.
    """

    SMA_WGSL = """
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = i32(id.x);
    let n = i32(arrayLength(&input));
    let w: i32 = 3;

    if (i < w - 1) {
        output[i] = 0;
        return;
    }

    var sum: i32 = 0;
    for (var j: i32 = i - w + 1; j <= i; j = j + 1) {
        sum = sum + input[j];
    }
    output[i] = sum / w;
}
"""

    def test_sma3_wgsl_vs_cpu(self):
        """SMA(3) — WGSL vs CPU, bit-exact."""
        import numpy as np

        # ── Reference: CpuDriver SMA ──
        from numfast.core.executor import CoreAPI
        api_cpu = CoreAPI()
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        inp_cpu = NumericSeries(data)
        ref = api_cpu.sma(inp_cpu, [3])[0]

        # ── WebGpuDriver + WGSL SMA ──
        driver = WebGpuDriver()
        inp = driver.allocate(len(data))
        out = driver.allocate(len(data))
        device = driver._device

        # Write input data
        for i, v in enumerate(data):
            driver.write(inp, i, v)

        key_in = driver._gpu_key(inp)
        key_out = driver._gpu_key(out)
        buf_in = driver._buffers[key_in]['gpu']
        buf_out = driver._buffers[key_out]['gpu']

        # Dispatch SMA WGSL
        driver.dispatch(
            self.SMA_WGSL,
            {0: buf_in, 1: buf_out},
            ((len(data) + 255) // 256,),
        )

        # Read back from GPU
        result = device.queue.read_buffer(buf_out)
        gpu_data = np.frombuffer(result, dtype=np.int32)

        # Compare
        for i in range(len(data)):
            expected = ref.read(i)
            got = int(gpu_data[i])
            assert got == expected, (
                f"SMA mismatch at [{i}]: GPU={got}, CPU={expected}"
            )

    def test_sma3_large_random(self):
        """SMA(3) на случайных данных — bit-exact."""
        import numpy as np
        np.random.seed(42)
        n = 1000
        data = list(np.random.randint(1, 1000, n))

        # CPU reference
        from numfast.core.executor import CoreAPI
        api_cpu = CoreAPI()
        inp_cpu = NumericSeries(data)
        ref = api_cpu.sma(inp_cpu, [3])[0]

        # GPU SMA
        driver = WebGpuDriver()
        inp = driver.allocate(n)
        out = driver.allocate(n)
        device = driver._device
        for i, v in enumerate(data):
            driver.write(inp, i, v)

        key_in = driver._gpu_key(inp)
        key_out = driver._gpu_key(out)
        buf_in = driver._buffers[key_in]['gpu']
        buf_out = driver._buffers[key_out]['gpu']

        driver.dispatch(
            self.SMA_WGSL,
            {0: buf_in, 1: buf_out},
            ((n + 255) // 256,),
        )

        result = device.queue.read_buffer(buf_out)
        gpu_data = np.frombuffer(result, dtype=np.int32)

        max_diff = 0
        for i in range(n):
            diff = abs(int(gpu_data[i]) - ref.read(i))
            if diff > max_diff:
                max_diff = diff

        assert max_diff == 0, f"SMA(3) max_diff = {max_diff} (expected 0)"
