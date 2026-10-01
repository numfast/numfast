#!/usr/bin/env python3
"""Detailed stage breakdown for composite_tuple_index."""
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app-builder"))

from Drivers.CPU._lib.groupindex import composite_tuple_index

SEED = 42
N = 1_000_000

def make_data(n, num_cols, card, rng):
    cols = []
    for i in range(num_cols):
        cols.append(rng.integers(0, card, size=n, dtype=np.int32))
    return cols

def bench_stages(cols, reps=3):
    """Measure each stage separately."""
    # Warmup
    composite_tuple_index(cols)
    
    results = []
    for _ in range(reps):
        t_total_start = time.perf_counter()
        
        # Stage 1: preprocess
        t0 = time.perf_counter()
        cols_proc = [np.ascontiguousarray(np.asarray(c)).ravel() for c in cols]
        n = int(cols_proc[0].size)
        t_preprocess = (time.perf_counter() - t0) * 1000
        
        # Stage 2: lexsort
        t0 = time.perf_counter()
        order = np.lexsort(tuple(cols_proc[::-1]))
        t_lexsort = (time.perf_counter() - t0) * 1000
        
        # Stage 3: fancy-index (sorted columns)
        t0 = time.perf_counter()
        scols = [c[order] for c in cols_proc]
        t_fancy = (time.perf_counter() - t0) * 1000
        
        # Stage 4: new_group detection
        t0 = time.perf_counter()
        new_group = np.zeros(n, dtype=bool)
        new_group[0] = True
        for sc in scols:
            nb = sc[1:] != sc[:-1]
            new_group[1:] |= nb
        t_newgroup = (time.perf_counter() - t0) * 1000
        
        # Stage 5: starts and ukeys
        t0 = time.perf_counter()
        starts = np.flatnonzero(new_group)
        ng = int(starts.size)
        parts = [sc[starts].tolist() for sc in scols]
        ukeys = [tuple(v) for v in zip(*parts)] if ng else []
        t_ukeys = (time.perf_counter() - t0) * 1000
        
        # Stage 6: inverse
        t0 = time.perf_counter()
        gid_sorted = np.cumsum(new_group.astype(np.int64)) - np.int64(1)
        inverse = np.empty(n, dtype=np.int64)
        inverse[order] = gid_sorted
        t_inverse = (time.perf_counter() - t0) * 1000
        
        # Stage 7: counts
        t0 = time.perf_counter()
        counts = np.bincount(gid_sorted, minlength=ng).astype(np.int64)
        t_counts = (time.perf_counter() - t0) * 1000
        
        t_total = (time.perf_counter() - t_total_start) * 1000
        
        results.append({
            'total': t_total,
            'preprocess': t_preprocess,
            'lexsort': t_lexsort,
            'fancy': t_fancy,
            'newgroup': t_newgroup,
            'ukeys': t_ukeys,
            'inverse': t_inverse,
            'counts': t_counts,
        })
    
    # Return best of reps
    best = min(results, key=lambda r: r['total'])
    return best

def main():
    rng = np.random.default_rng(SEED)
    
    for D in [100, 10_000, N]:
        cols = make_data(N, 4, D, rng)
        r = bench_stages(cols)
        print(f"\nD={D:>8}: total={r['total']:>8.2f}ms  groups={len(set(zip(*[c.tolist() for c in cols])))}")
        print(f"  preprocess={r['preprocess']:.2f}ms  lexsort={r['lexsort']:.2f}ms")
        print(f"  fancy-index={r['fancy']:.2f}ms  new_group={r['newgroup']:.2f}ms")
        print(f"  ukeys={r['ukeys']:.2f}ms  inverse={r['inverse']:.2f}ms  counts={r['counts']:.2f}ms")

if __name__ == "__main__":
    main()
