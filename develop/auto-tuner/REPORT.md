# Auto-tuner SEARCH V1 -- reproducible report (research only)
command: `export PYTHONPATH="develop/auto-tuner" && python develop/auto-tuner/AutoTuner.py`
tune fingerprint: {"n": 200000, "seed": 42, "split": "tune(seed42)", "thresh": 50, "selectivity": 0.4888, "nunique_filtered": 9998, "span_filtered": 10000, "is_sorted": false, "keys_dtype": "int32", "values_dtype": "int32", "query": "filter(values>thresh) -> groupby(keys) sum(values)"}
eval fingerprint: {"n": 200000, "seed": 43, "split": "eval(seed43)", "thresh": 50, "selectivity": 0.4892, "nunique_filtered": 10000, "span_filtered": 10000, "is_sorted": false, "keys_dtype": "int32", "values_dtype": "int32", "query": "filter(values>thresh) -> groupby(keys) sum(values)"}
baseline Plan A (ST/hash/materialize): 12.751ms
best Plan F (ST/dense_fused/fused): 5.084ms | reason: eliminated materialization (2->0 intermediate allocs, rows_read 400000->200000), chose dense_fused ST
baseline vs tuned: 12.751->5.084ms (measured, tune split)
eval winner: Plan F | overfit_flag=False
scope A = workload tuning (tune split); scope B = benchmark research (eval split).
candidates (tune split, ms | correct | mem | rows_read | mats | native):
  Plan F: STt1 dense_fused/fused = 5.084ms ok=True mem=200000 rows=200000 mats=0 nat=5
  Plan C: STt1 dense_fused/materialize = 5.428ms ok=True mem=982104 rows=400000 mats=2 nat=7
  Plan P: MTt2 dense_fused/fused = 9.363ms ok=True mem=200000 rows=200000 mats=0 nat=12
  Plan N: MTt4 dense_fused/fused = 11.517ms ok=True mem=200000 rows=200000 mats=0 nat=22
  Plan M: MTt4 dense_fused/materialize = 11.551ms ok=True mem=982104 rows=400000 mats=8 nat=30
  Plan B: STt1 dense_shift/materialize = 12.306ms ok=True mem=982104 rows=400000 mats=2 nat=7
  Plan O: MTt4 hash/fused = 12.479ms ok=True mem=200000 rows=200000 mats=0 nat=22
  Plan G: STt1 dense_shift/fused = 12.488ms ok=True mem=200000 rows=200000 mats=0 nat=5
  Plan E: STt1 unique/materialize = 12.664ms ok=True mem=982104 rows=400000 mats=2 nat=7
  Plan J: STt1 unique/fused = 12.746ms ok=True mem=200000 rows=200000 mats=0 nat=5
  Plan A: STt1 hash/materialize = 12.751ms ok=True mem=982104 rows=400000 mats=2 nat=7
  Plan K: MTt4 hash/materialize = 13.102ms ok=True mem=982104 rows=400000 mats=8 nat=30
  Plan H: STt1 hash/fused = 13.725ms ok=True mem=200000 rows=200000 mats=0 nat=5
  Plan L: MTt4 dense_shift/materialize = 15.187ms ok=True mem=982104 rows=400000 mats=8 nat=30
  Plan D: STt1 sorted/materialize = 15.282ms ok=True mem=982104 rows=400000 mats=2 nat=7
  Plan Q: MTt8 dense_fused/fused = 15.285ms ok=True mem=200000 rows=200000 mats=0 nat=42
  Plan I: STt1 sorted/fused = 15.405ms ok=True mem=200000 rows=200000 mats=0 nat=5
proof: mechanism works = search enumerates 17, oracle gates correctness, min-ms wins, eval split confirms (or flags overfit). Proof stops here.
