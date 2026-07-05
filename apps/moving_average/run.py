"""Moving Average — financial indicator using prefix sum."""
import numpy as np
from numfast import scan

def moving_average(data, window=5):
    prefix = scan(data)
    result = np.zeros_like(data)
    for i in range(len(data)):
        if i < window:
            result[i] = prefix[i] / (i + 1)
        else:
            result[i] = (prefix[i] - prefix[i - window]) / window
    return result

if __name__ == "__main__":
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.random(100) * 2 - 1)
    ma = moving_average(price, window=10)
    print(f"Price[ 0:10]: {price[:10].round(2)}")
    print(f"MA10[ 0:10]: {ma[:10].round(2)}")
    print(f"Price[-10:]: {price[-10:].round(2)}")
    print(f"MA10[-10:]: {ma[-10:].round(2)}")
    print(f"Max error vs numpy: {float(np.abs(ma - np.convolve(price, np.ones(10)/10, 'same')).max()):.2e}")
