# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""P0 regex_replace benchmark: before/after on ~1M rows.

Tests: flat object array (N~1M) + dictionary-encoded (D~100, N~1M).
Cold + warm, best-of-3, stage breakdown, RSS tracking.
Seed 42. CPU-only.
"""

import re
import time
from pathlib import Path

import numpy as np

APP_DIR = str(Path(__file__).resolve().parents[2])
SEED = 42
N_FLAT = 999_978
N_DICT_N = 999_978
D_DICT = 100  # unique values in dictionary


def _build_flat_data(rng, n):
    """Object array of str|None, realistic ClickBench-like URL-ish values."""
    pool = [
        "https://example.com/path/to/resource?id=12345&lang=en",
        "https://cdn.example.org/assets/images/2024/banner_v2.png",
        "http://search.example.com/query?q=python+regex&sort=date",
        "https://api.example.com/v2/users/998877/posts?page=1",
        "https://docs.example.io/guides/advanced/optimization",
        "https://static.cdn.example.net/js/analytics.min.js",
        "https://app.example.dev/dashboard/settings/security",
        "https://mail.example.com/inbox/message?id=5551234",
        "https://news.example.org/article/machine-learning-2024",
        "https://shop.example.com/product/widget-blue-large",
    ]
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.03:
            out.append(None)
        else:
            out.append(rng.choice(pool))
    return out


def _build_dict_data(rng, n, d):
    """Dictionary-encoded: D unique strings + codes int32[N]."""
    pool = [
        "https://example.com/path/to/resource?id=12345&lang=en",
        "https://cdn.example.org/assets/images/2024/banner_v2.png",
        "http://search.example.com/query?q=python+regex&sort=date",
        "https://api.example.com/v2/users/998877/posts?page=1",
        "https://docs.example.io/guides/advanced/optimization",
        "https://static.cdn.example.net/js/analytics.min.js",
        "https://app.example.dev/dashboard/settings/security",
        "https://mail.example.com/inbox/message?id=5551234",
        "https://news.example.org/article/machine-learning-2024",
        "https://shop.example.com/product/widget-blue-large",
        "https://login.example.com/auth/callback?token=abc123",
        "https://cdn2.example.org/fonts/open-sans-v26.woff2",
        "https://metrics.example.io/collect?batch=42&ts=1700000000",
        "https://git.example.com/repo/numfast/commit/abc123def",
        "https://grafana.example.com/d/abc123/executions?from=now-1h",
    ] + [f"https://unique{i:04d}.example.com/path" for i in range(d - 15)]
    pool = pool[:d]
    assert len(set(pool)) == d

    codes = np.array([rng.integers(d) for _ in range(n)], dtype=np.int32)

    # Build utf8 body
    body_parts = [s.encode("utf-8") for s in pool]
    body = b"".join(body_parts)
    offsets = np.zeros(d + 1, dtype=np.int32)
    pos = 0
    for i, bp in enumerate(body_parts):
        offsets[i] = pos
        pos += len(bp)
    offsets[d] = pos

    validity = np.ones(n, dtype=bool)
    for i in range(n):
        if rng.random() < 0.03:
            validity[i] = False

    return {
        "utf8_data": body,
        "offsets": offsets,
        "codes": codes,
        "validity": validity,
    }, pool


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _bench_flat(kernel, vals, pattern, repl, rounds=3, label="flat"):
    """Benchmark flat object array case, best-of-3."""
    a = kernel.alias
    node_fn = a["ir_text_regex_replace"]
    exec_fn = a["cpu_execute"]

    # Cold (first call)
    out_c = exec_fn([node_fn("o", np.asarray(vals, dtype=object), pattern, repl)])
    print(f"  [{label} cold] ", end="")

    results = []
    rss_before = _rss_mb()
    for _ in range(rounds):
        t0 = time.perf_counter()
        graph = exec_fn([node_fn("o", np.asarray(vals, dtype=object), pattern, repl)])
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(elapsed)
    rss_after = _rss_mb()
    best = min(results)
    avg = sum(results) / len(results)
    n = len(vals)
    throughput = n / (best / 1000) if best > 0 else 0
    print(f"best={best:.1f}ms avg={avg:.1f}ms rows/s={throughput/1e6:.2f}M "
          f"RSS_delta={rss_after - rss_before:+.1f}MB")
    return best


def _bench_dict(kernel, dict_vals, pattern, repl, rounds=3, label="dict"):
    """Benchmark dictionary-encoded case, best-of-3.

    Constructs IR node directly with dict in params["values"]
    (simulates Planner passing dictionary body + codes as params).
    """
    a = kernel.alias
    exec_fn = a["cpu_execute"]

    # Build IR node directly (bypass _keep_column which destroys dicts)
    def make_node():
        return {"op": "text_regex_replace", "inputs": [],
                "params": {"values": dict_vals, "pattern": pattern,
                           "repl": repl}, "out": "o"}

    # Cold
    out_c = exec_fn([make_node()])
    print(f"  [{label} cold] ", end="")

    results = []
    rss_before = _rss_mb()
    for _ in range(rounds):
        t0 = time.perf_counter()
        graph = exec_fn([make_node()])
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(elapsed)
    rss_after = _rss_mb()
    best = min(results)
    avg = sum(results) / len(results)
    n = dict_vals["codes"].size
    throughput = n / (best / 1000) if best > 0 else 0
    print(f"best={best:.1f}ms avg={avg:.1f}ms rows/s={throughput/1e6:.2f}M "
          f"RSS_delta={rss_after - rss_before:+.1f}MB")
    return best


def _verify_flat(out, valid, vals, pattern, repl):
    rx = re.compile(pattern)
    for i in range(len(vals)):
        v = vals[i]
        if isinstance(v, str):
            expected = rx.sub(repl, v, count=1)
            assert out[i] == expected, f"flat mismatch at {i}: {out[i]!r} != {expected!r}"
            assert bool(valid[i]) is True
        else:
            assert out[i] is None, f"flat expected None at {i}, got {out[i]!r}"
            assert bool(valid[i]) is False
    print("  Correctness: PASS")


def _verify_dict(out, valid, dict_vals, pattern, repl, pool):
    rx = re.compile(pattern)
    codes = dict_vals["codes"]
    validity = dict_vals["validity"]
    n = len(codes)
    for i in range(n):
        c = int(codes[i])
        v = bool(validity[i]) if validity is not None else True
        if v:
            expected = rx.sub(repl, pool[c], count=1)
            assert out[i] == expected, f"dict mismatch at {i}: {out[i]!r} != {expected!r}"
            assert bool(valid[i]) is True
        else:
            assert out[i] is None, f"dict expected None at {i}, got {out[i]!r}"
            assert bool(valid[i]) is False
    print("  Correctness: PASS")


def main():
    from builder import MAIN

    print(f"Loading kernel from {APP_DIR} ...")
    kernel = MAIN["build"](APP_DIR)
    print("Kernel loaded.\n")

    rng = np.random.default_rng(SEED)

    pattern = r"(https?)://([^/]+)"
    repl = r"\1://[REDACTED]"

    # --- Flat case ---
    print(f"=== Flat object array: N={N_FLAT:,} ===")
    flat_vals = _build_flat_data(rng, N_FLAT)
    t_best_flat = _bench_flat(kernel, flat_vals, pattern, repl)

    a = kernel.alias
    out_f = a["cpu_execute"]([a["ir_text_regex_replace"](
        "o", np.asarray(flat_vals, dtype=object), pattern, repl)])
    _verify_flat(out_f["o"], out_f["o#validity"], flat_vals, pattern, repl)

    # --- Dictionary case ---
    print(f"\n=== Dictionary-encoded: D={D_DICT}, N={N_DICT_N:,} ===")
    dict_vals, pool = _build_dict_data(rng, N_DICT_N, D_DICT)
    t_best_dict = _bench_dict(kernel, dict_vals, pattern, repl)

    # Rebuild to verify
    out_d = a["cpu_execute"]([{"op": "text_regex_replace", "inputs": [],
                               "params": {"values": dict_vals, "pattern": pattern,
                                          "repl": repl}, "out": "o"}])
    _verify_dict(out_d["o"], out_d["o#validity"], dict_vals, pattern, repl, pool)

    # --- Summary ---
    print(f"\n=== Summary (best-of-3, seed={SEED}) ===")
    print(f"  Flat N={N_FLAT:,}:     {t_best_flat:.1f}ms "
          f"({N_FLAT / (t_best_flat / 1000) / 1e6:.2f}M rows/s)")
    print(f"  Dict D={D_DICT} N={N_DICT_N:,}: {t_best_dict:.1f}ms "
          f"({N_DICT_N / (t_best_dict / 1000) / 1e6:.2f}M rows/s)")
    if t_best_dict > 0:
        print(f"  Dict/Flat speedup:   {t_best_flat / t_best_dict:.1f}x")


if __name__ == "__main__":
    main()
