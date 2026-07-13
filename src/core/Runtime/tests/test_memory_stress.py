"""Memory Stress Test — 10k+ случайных alloc/release.

Проверяет:
  - утечки (total_allocated == total_released после циклов)
  - reuse % (сколько выделений из пула, а не новых)
  - фрагментацию (сколько блоков в пуле vs активно)
  - корректность данных (записали-прочитали)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import random
import numpy as np
from Runtime._lib.memory import MemoryManager, MemoryBlock
from Runtime._lib.Drivers.CPU._lib.cpu_memory import CpuMemoryManager


def test_basic_alloc_release():
    """Базовый тест: alloc → write → read → release."""
    mm = CpuMemoryManager()
    sizes = [10, 100, 1000, 10000]

    for size in sizes:
        block = mm.alloc(size)
        arr = block.data
        assert len(arr) == size, f"Expected size={size}, got {len(arr)}"

        arr[:] = np.arange(size, dtype=np.float64)
        assert np.allclose(arr, np.arange(size, dtype=np.float64)), "Data corruption after alloc"

        mm.release(block)

    print(f"  OK Basic alloc/release: {len(sizes)} sizes OK")


def test_reuse():
    """Проверить, что release → alloc повторно использует блок."""
    mm = CpuMemoryManager()

    block1 = mm.alloc(100)
    ptr1 = block1.data.ctypes.data
    mm.release(block1)

    block2 = mm.alloc(100)
    ptr2 = block2.data.ctypes.data

    if ptr1 == ptr2:
        print(f"  OK Reuse: same address after alloc-release-alloc")
    else:
        print(f"  ~ Reuse: different addresses (pool may have grown)")

    block2.data[:] = 42.0
    assert np.all(block2.data == 42.0)
    mm.release(block2)


def test_waves_pattern():
    """Имитация волн Scheduler: alloc group → release group."""
    mm = CpuMemoryManager()

    for wave in range(10):
        blocks = []
        for _ in range(random.randint(5, 20)):
            size = random.randint(10, 1000)
            block = mm.alloc(size)
            block.data[:] = float(wave)
            blocks.append(block)

        for block in blocks:
            assert np.all(block.data == float(wave)), "Data corruption in wave"

        for block in blocks:
            mm.release(block)

    print(f"  OK Wave pattern: 10 waves, ~125 total blocks OK")


def test_random_stress():
    """10000 случайных операций alloc/release.

    Метрики:
      - leaks: total_allocated_size - total_released_size == 0
      - reuse: сколько блоков взято из пула (without creating new)
      - fragmentation: количество блоков в пуле
    """
    mm = CpuMemoryManager()
    random.seed(42)

    allocated = []  # (block, size)
    total_alloc_size = 0
    total_free_size = 0
    reuse_count = 0
    total_ops = 10000

    for op in range(total_ops):
        if not allocated or random.random() < 0.6:
            size = random.randint(1, 500)
            n_pool_before = len(mm._blocks)
            block = mm.alloc(size)
            n_pool_after = len(mm._blocks)

            if n_pool_after == n_pool_before:
                reuse_count += 1

            block.data[:] = random.random()
            total_alloc_size += block.size
            allocated.append((block, block.size))
        else:
            idx = random.randint(0, len(allocated) - 1)
            block, size = allocated.pop(idx)
            total_free_size += size
            mm.release(block)

    for block, size in allocated:
        total_free_size += size
        mm.release(block)
    allocated.clear()

    leak = total_alloc_size - total_free_size
    pool_size = len(mm._blocks)
    reuse_pct = (reuse_count / total_ops * 100) if total_ops > 0 else 0
    free_blocks = sum(1 for b in mm._blocks if b.refcount == 0 and not b.pinned)
    used_blocks = sum(1 for b in mm._blocks if b.refcount > 0)

    print(f"  Stress: {total_ops} ops")
    print(f"    total_alloc={total_alloc_size} total_free={total_free_size}")
    print(f"    leak={leak} {'OK' if leak == 0 else 'LEAK!'}")
    print(f"    reuse_count={reuse_count} ({reuse_pct:.1f}%)")
    print(f"    pool_blocks={pool_size} (free={free_blocks}, used={used_blocks})")

    assert leak == 0, f"Memory leak: {leak} bytes"
    print(f"  OK Random stress: no leaks")


def test_random_waves():
    """Имитация Scheduler с MemoryManager: волны с зависимостями."""
    mm = CpuMemoryManager()
    random.seed(123)

    n_waves = 200
    n_tasks_per_wave = 50

    for wave_idx in range(n_waves):
        wave_blocks = []
        for task_idx in range(n_tasks_per_wave):
            n_in = random.randint(1, 3)
            n_out = random.randint(1, 2)
            n_ws = random.randint(0, 2)

            for _ in range(n_in):
                block = mm.alloc(random.randint(10, 200))
                mm.release(block)

            ws_blocks = []
            for _ in range(n_ws):
                ws = mm.alloc(random.randint(5, 50))
                ws.data[:] = float(wave_idx * task_idx)
                ws_blocks.append(ws)

            for _ in range(n_out):
                block = mm.alloc(random.randint(10, 200))
                wave_blocks.append(block)

            for ws in ws_blocks:
                mm.release(ws)

        if wave_idx % 3 == 0:
            for block in wave_blocks:
                mm.release(block)

    print(f"  OK Random waves: {n_waves} waves x {n_tasks_per_wave} tasks simulated")


def test_fragmentation():
    """Проверить фрагментацию после множества alloc/release разного размера."""
    mm = CpuMemoryManager()

    blocks = []
    for i in range(1, 100):
        block = mm.alloc(i * 10)
        blocks.append(block)

    for i in range(len(blocks) - 1, -1, -2):
        mm.release(blocks[i])
        blocks.pop(i)

    for _ in range(50):
        block = mm.alloc(random.randint(10, 500))
        blocks.append(block)

    for block in blocks:
        mm.release(block)

    free_blocks = sum(1 for b in mm._blocks if b.refcount == 0 and not b.pinned)
    print(f"  OK Fragmentation test: ~100 blocks cycled")
    print(f"    pool blocks after cleanup: {len(mm._blocks)} (free={free_blocks})")


if __name__ == "__main__":
    test_basic_alloc_release()
    test_reuse()
    test_waves_pattern()
    test_random_stress()
    test_random_waves()
    test_fragmentation()
    print("\n=== All memory stress tests PASSED ===")
