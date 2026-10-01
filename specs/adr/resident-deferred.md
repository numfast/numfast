# ADR 16: Public Resident API (proposal — deferred, not normative)

Status: PROPOSAL / DEFERRED. Not normative. No implementation exists.
Frozen components (Runtime, Planner, Drivers, ABIs) are unaffected.

## 1. Known limitation

Public `evaluate(graph, backend, n)` is host-materializing today: each call
returns host buffers, so residence does not cross call boundaries — each
`evaluate()` pays H2D + D2H again. Series chains compile, optimize, evaluate,
materialize, and wrap per operator, i.e. one host round-trip per op.

Why it matters: bytes × crossings dominate these chains, not kernels.
Resident execution would remove inter-stage round-trips (one H2D + one D2H
per chain); it does not speed up kernels. Counter-example on file:
a resident Pack→GroupBy path measured slower than the round-trip path, so
residence must be costed per tract, never assumed faster. Details and numbers
belong to `benchmarks/execution-economics.md` and `benchmarks/h2o/overview.md`,
not to this proposal.

## 2. Proposed object: DeviceValue / ResidentValue

An opaque handle on a device buffer + metadata (logical dtype, n, separate
validity sidecar, placement=resident/host, owner, generation). No numpy data
inside. len()/dtype/schema work without materialization.

Ownership/lifetime: the creator is the Runtime/Driver (leased from the
existing GpuBufferPool); the object owns the lease; Python refcount governs
lifetime; release is close()/__del__/context-manager → return to the pool.
Copying is forbidden (alias views only, with a lifetime shorter than the
original). Transfer between graphs is move/borrow by handle. Double close()
is a no-op. References lost without close() return via finalizer + a
leaked_returned counter in execution_info (observable).

Device loss (init failure/reset/OOM): live ResidentValue → expired; any
operation on it raises a named ResidentExpired error (silent CPU fallback
forbidden). Allocation OOM raises an explicit ResidentOOM(n_bytes, budget).

Do not create: a second pool, a second Runtime, transit wrappers.

## 3. Materialization — explicit only

The only host-read points: resident.materialize() / .to_numpy() / .to_list() /
Series.from_resident(...).to_numpy() and the Planner-ordered final D2H.
Forbidden: hidden GPU→CPU→GPU copies; np.asarray(resident) without materialize
(= TypeError); to_numpy() in the hot path (= REJECT). Each materialize()
reports d2h_bytes + crossings+=["d2h"].

## 4. Chains graph1→graph2→graph3 without host round-trips

A new adjacent Python path: evaluate_resident(graph, hints) → ResidentValue +
chain([g1,g2,g3]) = one device context, intermediate ResidentValues, one H2D
+ one D2H. Planner: propagate_residence gains a residence_from input;
select_backend runs once per graph; the chain carries chain_residence
end-to-end. ping_pong is REJECT by planning (only via an explicit materialize
with a break). execution_info: total h2d/d2h_bytes, crossings list, per-stage
placement, resident_handles{alloc,reuse,returned}. Acceptance bar: crossings=0
between stages + EXPLAIN residence, ≥3 stages, seed 42, stage breakdown.

## 5. Combination with Series/Table

Series stays a host value. Only: (a) ResidentValue as the device side;
(b) adapters nf.from_resident(R) → Series (= materialize) and
nf.to_resident(series) → ResidentValue (= H2D). No lazy-Series or separate
class. Table: columns are Series; a resident table = dict[name → ResidentValue]
inside the chain context only. The LazyExpr/Query path is not duplicated.

## 6. Fallback without GPU

to_resident() without GPU = ResidentValue(placement=host) — a host buffer with
the same interface (materialize = no-op, close = release). Chains run the same
code, crossings=[], actual=cpu, reason observable. Parity resident-cpu ==
evaluate-cpu bit-for-bit on int32.

## 7. WASM — separate

WASM is importless, memory is caller-owned, there are no persistent
device buffers and no async dispatches. Resident on WASM = linear memory +
offset/len, lifetime = call duration. ResidentValue on WASM is always
placement=host-wasm. Chain = sequential calls copying through linear memory
(crossings as mem-copy). WASM is not an auto candidate until a separate spec.
Parity: RNG stages EXACT by hash only.

## 8. Compatibility

Existing evaluate(graph, backend, n) is unchanged. The new API sits alongside:
evaluate_resident / chain / to_resident / materialize in nf.*. Existing callers
are not rewritten. Frozen components and ABIs change nothing; only additive
hints.residence_from + an execution_info extension.

## 9. Native/ABI/kernel — untouched

Python level + Planner contract only. Zero new IR ops, WGSL, kernel tables,
or native libraries. Lookup→IR nodes are separate work.

## 10. Adoption criteria (not a decision)

Adopt only if all hold: (1) an end-to-end cascade of ≥3 stages with crossings=0,
stage breakdown, and EXPLAIN; (2) a byte-level win on a realistic tract with no
regression on the Pack→GroupBy counter-example; (3) leak/expired/OOM tests with
no silent fallback + green Guardian checks; (4) a minimal diff (pool/Planner/IR
reuse, no new Series classes, no native/ABI changes). Otherwise — stay deferred,
with residence driver-internal.

## WHAT MUST NEVER HAPPEN

Silent fallback; per-op routing inside a graph; bit-for-bit float claims;
extrapolation beyond measured transfer coverage presented as measured;
resident claims without crossings+H2D/D2H; hidden copies; a second
Runtime/IR/pool; frozen-ABI edits for numbers.
