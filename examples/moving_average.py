"""Moving average using prefix sum."""
import sys, os
import numpy as np
_examples_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_examples_dir, '..'))
sys.path.insert(0, os.path.join(_examples_dir, '..', '..'))

from numfast import scan


def moving_average(data: np.ndarray, window: int = 3) -> np.ndarray:
    """Compute simple moving average via prefix sum (Cumulative Moving Average).

    Args:
        data: 1D array
        window: window size

    Returns:
        smoothed: 1D array, same length
    """
    prefix = scan(data)
    n = len(data)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i < window - 1:
            result[i] = prefix[i] / (i + 1)
        else:
            result[i] = (prefix[i] - prefix[i - window]) / window
    return result


if __name__ == "__main__":
    np.random.seed(42)
    x = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.random(100) * 0.3
    smoothed = moving_average(x, window=5)
    print(f"Moving average (window=5):")
    print(f"  Input[ 0: 5]:  {x[:5].round(3)}")
    print(f"  Smooth[ 0: 5]: {smoothed[:5].round(3)}")
    print(f"  Input[-5:  ]:  {x[-5:].round(3)}")
    print(f"  Smooth[-5:  ]: {smoothed[-5:].round(3)}")
