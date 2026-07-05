#!/usr/bin/env python3
"""Generate expected outputs for Lab 001.

Запуск:
    python -m Labs.001_gpu_vector_math.expected

Создаёт .npy файлы в expected/ для golden tests.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np


def generate_expected():
    out_dir = os.path.join(os.path.dirname(__file__), "expected")
    os.makedirs(out_dir, exist_ok=True)

    n = 64
    data = np.array([float(i) * 0.5 for i in range(n)], dtype=np.float64)
    a = data
    b = np.array([float(i) * 2.0 for i in range(n)], dtype=np.float64)

    # Copy
    np.save(os.path.join(out_dir, "copy_expected.npy"), data)

    # AddConstant (value=3.14)
    np.save(os.path.join(out_dir, "add_3.14_expected.npy"), data + 3.14)

    # Multiply
    np.save(os.path.join(out_dir, "mul_expected.npy"), a * b)

    print("Generated expected outputs in expected/")
    for f in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, f)
        arr = np.load(path)
        print(f"  {f}: shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    generate_expected()
