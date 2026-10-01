# MISSION Q36 LOG

## EXECUTE-2
Tried:
- struct-argsort single pass on structured dtype [('x','<i8'),('y','<i8')] vs np.lexsort((b,a)), n=999978 seed42
- stable chained 2x argsort (argsort b stable, then argsort a stable) vs lexsort, n=20000 seed42 + NULL sentinel -1
- fused 128-bit single key rejected without implementation: numpy int64 overflow risk, object-dtype fallback loses exact/perf, Q36 key-dependent packing banned

Numbers:
- lexsort n=999978: 287.3ms OLD
- struct-argsort n=999978: 1498.3ms NEW, RATIO 5.21x slower, parity True (order equal)
- lexsort n=20000: 2.48ms OLD; stable2x: 2.52ms NEW, RATIO 1.02x slower, parity True incl NULL sentinel

Decision: NO-GO
Reason: no generic replacement beats lexsort by >=10% with exact parity.

## EXECUTE-3 DONE
Parity Q36 generic tuple path (composite_tuple_index) vs DuckDB hits_1m (999978 rows):
- int32 path (ClientIP int32, c0=CIP c1=CIP-1 c2=CIP-2 c3=CIP-3): groups=68330 match distinct CIP=68330; N small 5000+2pct NULL excluded groups=4901/4098 lex ascending True types ukeys=list/tuple/int inverse=int64 counts=int64 counts_sum=N inverse spot True; FULL lex inversions=0; top10 ORDER BY c DESC LIMIT 10 set_equal True multiset True strict order 9/10 (pos7/8 swap tie c=1015) => TIE-MULTISET (parity EXACT tie-aware).
- int64 path (same cols as int64): groups=165624 vs 68330 lex inversions=82898 top counts 236 vs 1733 => MISMATCH. Found: groupindex doesn't expose correct lex order on generic int64 tuple path. Suggested patch: single location fix in composite_tuple_index lexsort key construction (tuple(c[::-1] for c in cols) reverses each column data; use tuple(cols[::-1])).
Regression: Q21 PASS (95 EXACT 4.5ms); Q22 PASS (EXACT groups=1); Q23 PASS (TIE-MULTISET tie=True); Q24 contains PASS (146.8ms vs 152ms class, faster); Q29 PASS (unsupported DEFERRED pinned, no change).
Measure Q36 real int32: cold=147.5ms warm=154.8ms DuckDB=35.9ms (142 vs 38.7 class).
Freeze: engine untouched, only checks.

## EXECUTE-3b FIX
Bug: src/Drivers/CPU/_lib/groupindex.py:783,785 `np.lexsort(tuple(c[::-1] for c in cols))` reverses data inside each column instead of key order (confirmed by code read + repro: 2-col int64 n=20000 seed42 bug_inv=10046 vs fix_inv=0, bug order != ref).
Fix: `np.lexsort(tuple(cols[::-1]))` (2 lines, generic, fallback kept, IR/API/Planner/GPU/ABI untouched; int32 packed paths byte-identical per git diff).
Parity after fix (seed42, NULL sentinel -1 2pct excluded): int32 4-col N=19579 groups=4894 inv=0 spot=True top10=True EXACT; int64 4-col N=19610 groups=4910 inv=0 spot=True top10=True EXACT; 2-col int32/int64 inv=0; types ukeys=list/tuple inverse=int64 counts=int64 counts_sum=N.
Regression: tests/fast/test_ops_groupby.py + test_ops_composite.py 12 passed; test_microbench_q2_vs_q1 FAIL pre-existing (fails on clean tree too, single-key groupby perf-ratio flake, unrelated to this fix). Q21-Q23 int32 paths unchanged (packed branches untouched); Q24/Q29 unsupported statuses unchanged (no capability code touched).
