// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// THE ERROR CONTRACT, STATED HONESTLY
//
// The kernels report failure in TWO shapes, and neither shape is an
// unrecoverable abort:
//
//   1. RETURN CODE. `0` is ok. Negative codes are frozen by the C ABI
//      (`numfast-native/src/core/errors.rs`) but their MEANING IS PER SYMBOL:
//      `-1` is "null pointer" for `nf_group_*` and "unsorted abort" for
//      `nf_sorted_run_*`. There is no artefact in the repository that maps
//      every symbol to every code, so a raw number is never surfaced here --
//      see `RC_MEANING` in abi.ts for the per-symbol table this package ships.
//
//   2. TRAP. `WebAssembly.RuntimeError`. Measured 2026-10-04: 63 of the 85
//      kernels trap when a length argument is negative, because the kernels
//      do `slice::from_raw_parts(ptr, n)` with `n: usize` and a negative
//      `i32` becomes a ~4e9 element slice. The trap IS catchable, but an
//      unhandled one terminates the host process, so it is not a contract a
//      published package may leave to chance.
//
// This module therefore enforces two rules:
//
//   (a) PREFER PREVENTION. `checkLengths` rejects a negative or
//       non-representable length BEFORE the call, which removes the trap
//       rather than catching it. For the wrapped surface this is total: a
//       non-negative length is the only thing that reaches the slice
//       construction. The trap channel remains reachable for arguments
//       this package cannot model (aliased buffers, `memory.grow` races), so
//       it is caught too, and named.
//
//   (b) NEVER LEAK A MECHANISM. Every call leaves this module either
//       returning a value or throwing `NumFastError` (a return-code failure,
//       resolved through the per-symbol table) or `NumFastTrap` (a caught
//       `WebAssembly.RuntimeError`, carrying the symbol name). A caller who
//       catches `NumFastError | NumFastTrap` has caught everything.

/** A kernel returned a non-zero code. `code` is the raw ABI number; `reason`
 *  is the per-symbol meaning from `RC_MEANING`, never the number alone. */
export class NumFastError extends Error {
  readonly symbol: string;
  readonly code: number;
  readonly reason: string;

  constructor(symbol: string, code: number, reason: string) {
    super(`${symbol}: ${reason} (rc=${code})`);
    this.name = "NumFastError";
    this.symbol = symbol;
    this.code = code;
    this.reason = reason;
  }
}

/** A kernel trapped. The underlying `WebAssembly.RuntimeError` is the `cause`.
 *
 *  On any non-zero code -- here or in `NumFastError` -- outputs written past
 *  the abort point are caller-owned garbage (`errors.rs`: partial writes are
 *  documented kernel behaviour). The wrappers in kernels.ts do not return a
 *  buffer in that case. */
export class NumFastTrap extends Error {
  readonly symbol: string;

  constructor(symbol: string, cause: unknown) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    super(`${symbol}: kernel trapped (${detail}). This is a bounds or aliasing ` +
      `violation in the call, not a returned error code.`);
    this.name = "NumFastTrap";
    this.symbol = symbol;
    this.cause = cause;
  }
}

/** A caller-supplied argument violated the documented contract before any
 *  kernel ran. Never thrown from inside a WASM call. */
export class NumFastArgumentError extends Error {
  readonly symbol: string;

  constructor(symbol: string, detail: string) {
    super(`${symbol}: ${detail}`);
    this.name = "NumFastArgumentError";
    this.symbol = symbol;
  }
}

/** The largest element count a kernel can be asked for. `usize` is 32-bit on
 *  wasm32, and a negative `i32` is reinterpreted as a ~4e9 slice length, so
 *  anything above 2^31-1 is rejected rather than trapped. */
export const MAX_LEN = 0x7fffffff;

/**
 * Validate every length-shaped argument of a call. This is the prevention
 * half of the trap contract: it is what turns "63 of 85 kernels trap" into
 * "0 trap for a length the caller got wrong".
 *
 * @param symbol    kernel name, for the error message
 * @param lengths   the arguments that are element counts, by position
 */
export function checkLengths(symbol: string, ...lengths: readonly number[]): void {
  for (let i = 0; i < lengths.length; i++) {
    const n = lengths[i] as number;
    if (typeof n !== "number" || !Number.isInteger(n)) {
      throw new NumFastArgumentError(symbol,
        `length argument ${i} is ${typeof n === "number" ? String(n) : typeof n}, expected an integer`);
    }
    if (n < 0) {
      throw new NumFastArgumentError(symbol,
        `negative length ${n} at argument ${i}; the kernel would reinterpret it as ` +
        `${(n >>> 0).toString()} elements and trap`);
    }
    if (n > MAX_LEN) {
      throw new NumFastArgumentError(symbol,
        `length ${n} at argument ${i} exceeds ${MAX_LEN}; wasm32 usize is 32-bit`);
    }
  }
}

/**
 * Reject a JavaScript `Number` where the ABI wants an i64. Measured live:
 * passing `12345` to an i64 parameter raises
 * `TypeError: Cannot convert 12345 to a BigInt`; passing `12345n` is accepted.
 * Silently truncating would be worse than the TypeError, so this converts the
 * engine's message into a named one and documents that the boundary is a hard
 * type boundary, not a coercion.
 */
export function requireBigInt(symbol: string, what: string, v: unknown): bigint {
  if (typeof v === "bigint") return v;
  if (typeof v === "number") {
    throw new NumFastArgumentError(symbol,
      `${what} is an i64 on this kernel's ABI and must be a BigInt; ` +
      `Number ${v} would be silently truncated (got ${typeof v})`);
  }
  throw new NumFastArgumentError(symbol,
    `${what} is an i64 on this kernel's ABI and must be a BigInt (got ${typeof v})`);
}

/** The exact signature of `callGuarded`'s callee. */
type RawFn = (...args: never[]) => unknown;

/**
 * Run one kernel with both error channels closed.
 *
 * Nothing else in this package calls `instance.exports` directly: a raw
 * `WebAssembly.RuntimeError` must not be reachable from a library user, and
 * the "outputs past the abort point are garbage" rule means the caller of
 * this function must not be handed a buffer on failure.
 *
 * @param symbol   kernel name, carried on both error types
 * @param fn       the raw export
 * @param args     already-validated arguments
 */
export function callGuarded<T>(symbol: string, fn: RawFn, ...args: number[]): T {
  let rc: unknown;
  try {
    rc = fn(...(args as never[]));
  } catch (e) {
    throw new NumFastTrap(symbol, e);
  }
  const code = typeof rc === "bigint" ? Number(rc) : (rc as number);
  if (typeof code === "number" && Number.isFinite(code) && code !== 0) {
    throw new NumFastError(symbol, code, rcMeaning(symbol, code));
  }
  return rc as T;
}

/**
 * Run one kernel that returns a VALUE, not a status code.
 *
 * `nf_cost_intern` is the case that forces this distinction: it returns the
 * number of interned groups, so a perfectly successful call answers 3. Passing
 * that through the return-code check would turn every success into a
 * `NumFastError`. So the two channels are separated explicitly:
 *
 *   callGuarded  -> returns a STATUS (0 = ok, negative = failure)
 *   callTrapped  -> returns a VALUE (any non-negative value is success)
 *
 * Both close the trap channel. Only the first interprets the return value.
 */
export function callTrapped<T>(symbol: string, fn: RawFn, ...args: number[]): T {
  try {
    return fn(...(args as never[])) as T;
  } catch (e) {
    throw new NumFastTrap(symbol, e);
  }
}

/** Per-symbol meaning of a return code. Overridden by kernels.ts with the
 *  table from abi.ts; kept here so errors.ts has no import cycle. */
let meaningTable: Readonly<Record<string, Readonly<Record<number, string>>>> = {};

/** @internal installed once by kernels.ts */
export function _setRcTable(t: Readonly<Record<string, Readonly<Record<number, string>>>>): void {
  meaningTable = t;
}

function rcMeaning(symbol: string, code: number): string {
  const forSym = meaningTable[symbol];
  const known = forSym?.[code];
  if (known !== undefined) return known;
  return `return code ${code} is not in this package's table for ${symbol}; ` +
    `the ABI assigns a per-symbol meaning to every negative code and no ` +
    `artefact records all of them`;
}