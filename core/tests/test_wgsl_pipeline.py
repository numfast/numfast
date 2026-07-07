"""Phase 14.2.1 — Minimal WGSL compute pipeline.

Проверяет:
  - инициализация wgpu device
  - создание storage буферов
  - компиляция WGSL шейдера
  - dispatch compute shader (copy)
  - чтение результатов

Запуск:
    python -m pytest numfast/core/tests/test_wgsl_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import wgpu
import numpy as np
import pytest

from numfast.core.gpu.device import get_device


class TestWgslPipeline:
    """Минимальный compute pipeline."""

    def test_device_created(self):
        """GPUDevice создаётся."""
        device = get_device()
        assert device is not None
        assert isinstance(device, wgpu.GPUDevice)

    def test_copy_shader_basic(self):
        """y[i] = x[i] — copy shader."""
        device = get_device()
        n = 64

        # Input data
        input_data = np.array(list(range(n)), dtype=np.int32)

        # Create storage buffers
        input_buf = device.create_buffer(
            size=input_data.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        output_buf = device.create_buffer(
            size=input_data.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

        # Upload input data
        tmp = device.create_buffer_with_data(
            data=input_data,
            usage=wgpu.BufferUsage.COPY_SRC,
        )
        commands = device.create_command_encoder()
        commands.copy_buffer_to_buffer(tmp, 0, input_buf, 0, input_data.nbytes)
        device.queue.submit([commands.finish()])

        # Compile shader
        shader = device.create_shader_module(code=COPY_WGSL)

        # Create compute pipeline
        pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": "main"},
        )

        # Create bind group
        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": input_buf},
                {"binding": 1, "resource": output_buf},
            ],
        )

        # Dispatch
        commands = device.create_command_encoder()
        pass_enc = commands.begin_compute_pass()
        pass_enc.set_pipeline(pipeline)
        pass_enc.set_bind_group(0, bind_group)
        workgroups = (n + 255) // 256
        pass_enc.dispatch_workgroups(workgroups)
        pass_enc.end()
        device.queue.submit([commands.finish()])

        # Read back via queue.read_buffer (создаёт стейджинг внутри)
        result = device.queue.read_buffer(output_buf)
        output_data = np.frombuffer(result, dtype=np.int32)

        # Verify
        np.testing.assert_array_equal(output_data, input_data)

    def test_add_one_shader(self):
        """y[i] = x[i] + 1."""
        device = get_device()
        n = 128

        input_data = np.array(list(range(n)), dtype=np.int32)

        input_buf = device.create_buffer(
            size=input_data.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        output_buf = device.create_buffer(
            size=input_data.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

        # Upload
        tmp = device.create_buffer_with_data(data=input_data, usage=wgpu.BufferUsage.COPY_SRC)
        commands = device.create_command_encoder()
        commands.copy_buffer_to_buffer(tmp, 0, input_buf, 0, input_data.nbytes)
        device.queue.submit([commands.finish()])

        shader = device.create_shader_module(code=ADD_ONE_WGSL)

        pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": "main"},
        )

        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": input_buf},
                {"binding": 1, "resource": output_buf},
            ],
        )

        commands = device.create_command_encoder()
        pass_enc = commands.begin_compute_pass()
        pass_enc.set_pipeline(pipeline)
        pass_enc.set_bind_group(0, bind_group)
        pass_enc.dispatch_workgroups((n + 255) // 256)
        pass_enc.end()
        device.queue.submit([commands.finish()])

        result = device.queue.read_buffer(output_buf)
        output_data = np.frombuffer(result, dtype=np.int32)

        expected = input_data + 1
        np.testing.assert_array_equal(output_data, expected)


# ── WGSL sources (inline для тестов) ──────────────────────────

COPY_WGSL = """
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = input[i];
}
"""

ADD_ONE_WGSL = """
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = input[i] + 1;
}
"""
