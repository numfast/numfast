"""Black-Scholes option pricing — batch evaluation."""
import numpy as np

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option price.
    
    S: asset price
    K: strike price
    T: time to maturity (years)
    r: risk-free rate
    sigma: volatility
    option_type: 'call' or 'put'
    """
    from scipy.stats import norm
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def batch_options_prices():
    """Price multiple options in batch — vectorized."""
    np.random.seed(42)
    n = 100000
    S = np.random.uniform(80, 120, n).astype(np.float64)
    K = np.random.uniform(90, 110, n).astype(np.float64)
    T = np.random.uniform(0.1, 2.0, n).astype(np.float64)
    r = 0.05
    sigma = 0.2
    
    calls = black_scholes(S, K, T, r, sigma, 'call')
    puts = black_scholes(S, K, T, r, sigma, 'put')
    
    return S, K, calls, puts

if __name__ == "__main__":
    import time
    
    # Small batch
    S = np.array([100.0])
    K = np.array([100.0])
    T = np.array([1.0])
    price = black_scholes(S, K, T, 0.05, 0.2, 'call')
    print(f"ATM call (S=K=100, T=1, r=5%, sigma=20%): {price[0]:.4f}")
    
    # Benchmark batch
    t0 = time.perf_counter()
    S, K, calls, puts = batch_options_prices()
    t = time.perf_counter() - t0
    print(f"\nBatch pricing: {len(calls):,} options in {t*1000:.1f} ms")
    print(f"  Call prices: {calls.mean():.4f} +/- {calls.std():.4f}")
    print(f"  Put prices:  {puts.mean():.4f} +/- {puts.std():.4f}")
