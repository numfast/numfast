import time
import numpy as np
from run import linear_regression

for n in [100, 500, 1000]:
    X = np.random.random((n, 10)).astype(np.float64)
    y = np.random.random(n).astype(np.float64)
    t0 = time.perf_counter()
    for _ in range(10):
        linear_regression(X, y)
    t = (time.perf_counter() - t0) / 10
    print(f"n={n:>5}: {t*1000:.3f} ms  ({n/t/1e6:.2f} M rows/s)")
