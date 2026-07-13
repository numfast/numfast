"""Kernel Test Harness — автоматическая проверка ядер.

Любое ядро (EMA, RSI, FFT, Stress, ...) можно проверить:

    report = test_kernel(runtime, "EMA", {"period": 3}, {"Close": data})
    assert report["passed"]
    print(report["summary"])  # "EMA(period=3): OK | NaN=0 Inf=0 range=[1.0..5.0]"

С CPU-эталоном:

    report = test_kernel(runtime, "EMA", {"period": 3}, {"Close": data},
                         expected=reference_output)
    assert report["max_abs_error"] < 1e-12
"""

import numpy as np


class KernelTestHarness:
    """Проверка одного ядра: запуск, валидация, метрики."""

    def __init__(self, runtime):
        self.runtime = runtime
        self._results = []

    def run(self, kernel, params, input_data, expected=None, label=None):
        """Запустить ядро и проверить результат.

        Args:
            kernel: str — alias ("EMA")
            params: dict — параметры ({"period": 3})
            input_data: dict — входные данные ({"Close": np.array})
            expected: np.array или dict[name → np.array] — эталон (опционально)
            label: str — метка для отчёта (опционально)

        Returns:
            dict с метриками:
                passed: bool
                kernel: str
                params: dict
                outputs: dict[name → np.array]
                nan_count: int
                inf_count: int
                finite_pct: float
                output_ranges: dict[name → (min, max)]
                max_abs_error: float (если expected)
                max_rel_error: float (если expected)
                summary: str
        """
        # Compile
        jobs = [{"op": kernel, "inputs": list(input_data.keys()), "params": params}]
        tasks = self.runtime.compile(jobs)

        # Execute
        self.runtime.execute(tasks, input_data)

        # Collect outputs
        outputs = {}
        for t in tasks:
            for name in t.out_names:
                raw = self.runtime.driver.memory.resolve_output(name)
                if raw is not None:
                    outputs[name] = raw

        # Compute metrics
        report = {
            "kernel": kernel,
            "params": params,
            "outputs": outputs,
            "passed": True,
        }

        # Per-output metrics
        output_metrics = {}
        total_nan = 0
        total_inf = 0
        total_finite = 0
        total_elems = 0
        output_ranges = {}

        for name, arr in outputs.items():
            nan_c = int(np.isnan(arr).sum())
            inf_c = int(np.isinf(arr).sum())
            fin_c = int(np.isfinite(arr).sum())
            total_nan += nan_c
            total_inf += inf_c
            total_finite += fin_c
            total_elems += arr.size

            if arr.size > 0:
                output_ranges[name] = (float(arr.min()), float(arr.max()))
            else:
                output_ranges[name] = (None, None)

            output_metrics[name] = {
                "nan": nan_c,
                "inf": inf_c,
                "finite": fin_c,
                "size": arr.size,
            }

        report["nan_count"] = total_nan
        report["inf_count"] = total_inf
        report["finite_pct"] = (total_finite / total_elems * 100) if total_elems > 0 else 100.0
        report["output_ranges"] = output_ranges
        report["output_metrics"] = output_metrics

        # Compare with expected
        max_abs_err = 0.0
        max_rel_err = 0.0
        if expected is not None:
            for name, arr in outputs.items():
                if isinstance(expected, np.ndarray):
                    exp = expected
                elif isinstance(expected, dict) and name in expected:
                    exp = expected[name]
                else:
                    continue

                abs_err = np.abs(arr - exp)
                rel_err = abs_err / (np.abs(exp) + 1e-15)
                max_abs_err = max(max_abs_err, float(abs_err.max()))
                max_rel_err = max(max_rel_err, float(rel_err.max()))

            report["max_abs_error"] = max_abs_err
            report["max_rel_error"] = max_rel_err

            if max_abs_err > 1e-10:
                report["passed"] = False

        # Fail on NaN/Inf
        if total_nan > 0 or total_inf > 0:
            report["passed"] = False

        # Summary string
        label_str = f" [{label}]" if label else ""
        parts = [
            f"{kernel}{label_str}({self._fmt_params(params)})",
            f"{'OK' if report['passed'] else 'FAIL'}",
        ]
        fail_reasons = []
        if total_nan > 0:
            fail_reasons.append(f"NaN={total_nan}")
        if total_inf > 0:
            fail_reasons.append(f"Inf={total_inf}")
        if fail_reasons:
            parts.append("| " + " ".join(fail_reasons))
        if output_ranges:
            ranges_str = " ".join(f"{k}=[{lo:.4g}..{hi:.4g}]" for k, (lo, hi) in output_ranges.items())
            parts.append(f"| range: {ranges_str}")
        if max_abs_err > 0:
            parts.append(f"| err_abs={max_abs_err:.2e} err_rel={max_rel_err:.2e}")
        report["summary"] = " ".join(parts)

        self._results.append(report)
        return report

    def validate(self, kernel, params, input_data, expected=None, label=None):
        """Запустить и assert passed."""
        report = self.run(kernel, params, input_data, expected, label)
        assert report["passed"], f"FAILED: {report['summary']}"
        return report

    def report_all(self):
        """Напечатать отчёт по всем запускам."""
        passed = sum(1 for r in self._results if r["passed"])
        total = len(self._results)
        print(f"\n{'='*60}")
        print(f"Kernel Test Harness Report: {passed}/{total} passed")
        print(f"{'='*60}")
        for r in self._results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {status} {r['summary']}")
        if passed < total:
            print(f"\n  FAILURES:")
            for r in self._results:
                if not r["passed"]:
                    if r.get("max_abs_error", 0) > 1e-10:
                        print(f"    - {r['kernel']}: max_abs_error={r['max_abs_error']:.2e}")
                    if r["nan_count"] > 0:
                        print(f"    - {r['kernel']}: NaN count={r['nan_count']}")
                    if r["inf_count"] > 0:
                        print(f"    - {r['kernel']}: Inf count={r['inf_count']}")
        print(f"{'='*60}\n")
        return passed == total

    @staticmethod
    def _fmt_params(params):
        return ", ".join(f"{k}={v}" for k, v in params.items())


# Convenience functions
def test_kernel(runtime, kernel, params, input_data, expected=None, label=None):
    """Запустить одно ядро и вернуть отчёт."""
    harness = KernelTestHarness(runtime)
    return harness.run(kernel, params, input_data, expected, label)


def validate_kernel(runtime, kernel, params, input_data, expected=None, label=None):
    """Запустить одно ядро с assert."""
    harness = KernelTestHarness(runtime)
    return harness.validate(kernel, params, input_data, expected, label)
