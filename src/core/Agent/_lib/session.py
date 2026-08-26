"""Session — эксклюзивный владелец Runtime.

Сессия:
  - Создаёт Runtime
  - Регистрирует вычислительные ядра
  - Предоставляет инструменты (run, test, benchmark, lab, compare, diff)
  - Освобождает Runtime при закрытии
"""

import time
from typing import Optional

from Runtime.Runtime import Runtime
from Compute.Compute import register_all as register_compute_kernels
from operations.Operations import scan, matmul, fft, sort, histogram


class Session:
    """Сессия выполнения. Эксклюзивно владеет Runtime.

    Usage:
        with sdk.session() as session:
            session.run("script.py")
    """

    def __init__(self, use_gpu: bool = False):
        self._use_gpu = use_gpu
        self._runtime: Optional[Runtime] = None
        self._driver = None

    # ── Context manager ────────────────────────────────────────────────

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
        return False

    def _open(self):
        """Создать Runtime и зарегистрировать ядра."""
        from Runtime._lib.Drivers.WebGPU import WebGpuDriver

        self._driver = WebGpuDriver() if self._use_gpu else None
        self._runtime = Runtime(driver=self._driver)
        # Регистрируем Compute ядра (Map, Reduce, Scan, Histogram, Sort, MatMul, FFT)
        register_compute_kernels(self._runtime)

    def _close(self):
        """Освободить ресурсы."""
        if self._driver is not None:
            self._driver.release()
            self._driver = None
        self._runtime = None

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def runtime(self) -> Optional[Runtime]:
        """Runtime этой сессии. None после закрытия."""
        return self._runtime

    @property
    def driver_name(self) -> str:
        """Имя активного драйвера: 'cpu' или 'webgpu'."""
        return "webgpu" if self._use_gpu else "cpu"

    # ── Tools ──────────────────────────────────────────────────────────

    def run(self, script: str, **kwargs):
        """Выполнить Python-скрипт.

        Пока — заглушка. Запускает скрипт через compile() + execute() из строки кода.

        Args:
            script: Python-код как строка
            **kwargs: переменные для exec()

        Returns:
            Result
        """
        from _lib import Result
        t0 = time.time()
        try:
            local_vars = dict(kwargs)
            exec(script, {"__builtins__": __builtins__}, local_vars)
            t1 = time.time()
            return Result(
                status="ok",
                data=local_vars,
                duration=t1 - t0,
                metadata={"driver": self.driver_name},
            )
        except Exception as e:
            t1 = time.time()
            return Result(
                status="error",
                error=str(e),
                duration=t1 - t0,
                metadata={"driver": self.driver_name},
            )

    def test(self, path: str = "tests"):
        """Запустить pytest для указанного пути.

        Args:
            path: путь к тестам

        Returns:
            Result
        """
        from _lib import Result
        t0 = time.time()
        try:
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "-m", "pytest", path, "-v", "--tb=short"],
                capture_output=True, text=True, timeout=120,
            )
            t1 = time.time()
            passed = result.returncode == 0
            return Result(
                status="ok" if passed else "error",
                data={
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "passed": result.stdout.count("PASSED") if not passed else None,
                    "failed": result.stderr.count("FAILED") if not passed else None,
                },
                error="" if passed else result.stderr[:500],
                duration=t1 - t0,
                metadata={"driver": self.driver_name, "path": path},
            )
        except Exception as e:
            t1 = time.time()
            return Result(
                status="error",
                error=str(e),
                duration=t1 - t0,
                metadata={"driver": self.driver_name, "path": path},
            )

    def benchmark(self, name: str, script: str = None, n_iter: int = 5):
        """Замерить производительность.

        Args:
            name: имя бенчмарка
            script: Python-код для замера
            n_iter: количество итераций

        Returns:
            Result
        """
        from _lib import Result
        t0 = time.time()
        try:
            import time as _time
            times = []
            for _ in range(n_iter):
                t_start = _time.perf_counter()
                if script:
                    local_vars = {}
                    exec(script, {"__builtins__": __builtins__}, local_vars)
                t_end = _time.perf_counter()
                times.append(t_end - t_start)

            t1 = time.time()
            avg_time = sum(times) / len(times)
            return Result(
                status="ok",
                data={
                    "name": name,
                    "n_iter": n_iter,
                    "times": times,
                    "avg": avg_time,
                    "min": min(times),
                    "max": max(times),
                },
                duration=t1 - t0,
                metadata={"driver": self.driver_name, "n_iter": n_iter},
            )
        except Exception as e:
            t1 = time.time()
            return Result(
                status="error",
                error=str(e),
                duration=t1 - t0,
                metadata={"driver": self.driver_name},
            )

    def lab(self, number: int):
        """Запустить Lab по номеру.

        Args:
            number: номер Lab (1, 2, 5, ...)

        Returns:
            Result
        """
        from _lib import Result
        t0 = time.time()
        try:
            lab_name = f"Labs._{number:03d}_{{}}"
            # Try to find the lab by number
            labs_dir = self._find_lab_dir(number)
            if labs_dir is None:
                return Result(
                    status="error",
                    error=f"Lab {number} not found",
                    duration=time.time() - t0,
                    metadata={"driver": self.driver_name},
                )
            t1 = time.time()
            return Result(
                status="ok",
                data={"lab": number, "path": str(labs_dir)},
                duration=t1 - t0,
                metadata={"driver": self.driver_name},
            )
        except Exception as e:
            t1 = time.time()
            return Result(
                status="error",
                error=str(e),
                duration=t1 - t0,
                metadata={"driver": self.driver_name},
            )

    def _find_lab_dir(self, number: int):
        """Найти директорию Lab по номеру."""
        import os
        # Search in common locations
        for base in [".", "..", "../.."]:
            labs_root = os.path.join(base, "Labs")
            if not os.path.isdir(labs_root):
                continue
            for entry in os.listdir(labs_root):
                if entry.startswith(f"_{number:03d}"):
                    return os.path.join(labs_root, entry)
        return None

    def compare(self, cpu_data, gpu_data, tolerance: float = 1e-4):
        """Сравнить CPU и GPU результаты.

        Args:
            cpu_data: numpy array — CPU результат
            gpu_data: numpy array — GPU результат
            tolerance: максимальная допустимая ошибка

        Returns:
            Result
        """
        from _lib import Result
        import numpy as np
        t0 = time.time()
        try:
            cpu_arr = np.asarray(cpu_data)
            gpu_arr = np.asarray(gpu_data)
            max_diff = float(np.abs(cpu_arr - gpu_arr).max())
            mean_diff = float(np.abs(cpu_arr - gpu_arr).mean())
            passed = max_diff <= tolerance
            t1 = time.time()
            return Result(
                status="ok" if passed else "error",
                data={
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "tolerance": tolerance,
                    "passed": passed,
                    "cpu_shape": cpu_arr.shape,
                    "gpu_shape": gpu_arr.shape,
                },
                error="" if passed else f"max_diff={max_diff:.2e} > tolerance={tolerance:.0e}",
                duration=t1 - t0,
                metadata={"driver": self.driver_name, "tolerance": tolerance},
            )
        except Exception as e:
            t1 = time.time()
            return Result(
                status="error",
                error=str(e),
                duration=t1 - t0,
                metadata={"driver": self.driver_name},
            )

    def diff(self, actual, expected, tolerance: float = 1e-10):
        """Сравнить два массива (actual vs expected).

        Args:
            actual: numpy array — полученный результат
            expected: numpy array — ожидаемый результат
            tolerance: максимальная допустимая ошибка

        Returns:
            Result
        """
        from _lib import Result
        import numpy as np
        t0 = time.time()
        try:
            act = np.asarray(actual)
            exp = np.asarray(expected)
            max_err = float(np.abs(act - exp).max())
            mean_err = float(np.abs(act - exp).mean())
            passed = max_err <= tolerance
            t1 = time.time()
            return Result(
                status="ok" if passed else "error",
                data={
                    "max_err": max_err,
                    "mean_err": mean_err,
                    "tolerance": tolerance,
                    "passed": passed,
                    "shape_match": act.shape == exp.shape,
                },
                error="" if passed else f"max_err={max_err:.2e} > tolerance={tolerance:.0e}",
                duration=t1 - t0,
                metadata={"driver": self.driver_name, "tolerance": tolerance},
            )
        except Exception as e:
            t1 = time.time()
            return Result(
                status="error",
                error=str(e),
                duration=t1 - t0,
                metadata={"driver": self.driver_name},
            )
