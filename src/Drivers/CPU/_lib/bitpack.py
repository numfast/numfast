# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Bit-packed masks/enums: BoolMask as 1-bit enum + generic width pack (DELTA-5).

Bit suitcase for small integer series: width w in (1, 2, 4, 8) codes packed
LSB-first, k=8//w codes per byte (8 masks per byte at w=1). Vectorized only:
packbits/unpackbits for w=1, strided lane accumulate / repeat-shift-mask for
w in (2, 4, 8). No Python per-row loops at any N.

BitMask ops (and/or/not) run directly on packed bytes, never unpacking to
int32. filter unpacks bits -> bool once (proven-necessary: compacting take
needs a materialized row selector; never int32). Everything else reaches the
packed data via __array__/__array_ufunc__ (bit-exact unpack-on-use, same
semantics as the old bool-ndarray buffers).

STANDALONE module: numpy only, no Builder/Extension imports.
"""

import numpy as np

WIDTHS = (1, 2, 4, 8)


def check_width(width):
    """check_width(w) -> int; only 1/2/4/8 (exact byte subdivision)."""
    w = int(width)
    if w not in WIDTHS:
        raise ValueError(
            f"bitpack width must be one of {list(WIDTHS)}, got {width!r}")
    return w


def _tail_mask(n):
    r = int(n) % 8
    return 0xFF if r == 0 else (1 << r) - 1


def pack_bits(a):
    """bool array -> packed uint8 (LSB-first, zero-padded tail)."""
    a = np.ascontiguousarray(np.asarray(a, dtype=bool))
    return np.packbits(a, bitorder="little")


def unpack_bits(packed, n):
    """packed uint8 + n -> bool array (tail bits beyond n ignored)."""
    p = np.ascontiguousarray(np.asarray(packed, dtype=np.uint8))
    n = int(n)
    if n == 0:
        return np.zeros(0, dtype=bool)
    return np.unpackbits(p, bitorder="little")[:n].astype(bool, copy=False)


def pack_codes(codes, width):
    """int codes in [0, 2**w) -> packed uint8 (LSB-first lane layout)."""
    w = check_width(width)
    c = np.ascontiguousarray(np.asarray(codes).ravel())
    if c.size and (c.min(initial=0) < 0 or c.max(initial=0) > (1 << w) - 1):
        raise ValueError(
            f"bitpack width-{w} code out of range [0, {(1 << w) - 1}]")
    if w == 1:
        return pack_bits(c.astype(bool, copy=False))
    k = 8 // w
    nb = (c.size + k - 1) // k
    out = np.zeros(nb, dtype=np.uint8)
    for j in range(k):
        lane = c[j::k]
        if lane.size > nb:
            lane = lane[:nb]
        if lane.size:
            out[:lane.size] |= (lane.astype(np.uint8) << np.uint8(w * j))
    return out


def unpack_codes(packed, n, width):
    """packed uint8 + n + width -> uint8 code array."""
    w = check_width(width)
    p = np.ascontiguousarray(np.asarray(packed, dtype=np.uint8))
    n = int(n)
    if w == 1:
        return unpack_bits(p, n).view(np.uint8)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    k = 8 // w
    nb = (n + k - 1) // k
    rep = np.repeat(p[:nb], k)[:n]
    shifts = np.tile((np.arange(k) * w).astype(np.uint8), nb)[:n]
    return ((rep >> shifts) & np.uint8((1 << w) - 1)).astype(np.uint8)


class BitPack:
    """Width-w bit-packed column: {width, packed uint8, n}.

    Sequence/buffer protocol mirrors the old bool-ndarray buffers (list(),
    np.asarray(), .size/.shape/.dtype, fancy getitem, numpy operators) so
    existing consumers keep working; hot mask ops use .packed directly.
    """

    __slots__ = ("packed", "n", "width")

    def __init__(self, packed, n, width=1):
        w = check_width(width)
        p = np.ascontiguousarray(np.asarray(packed, dtype=np.uint8))
        object.__setattr__(self, "packed", p)
        object.__setattr__(self, "n", int(n))
        object.__setattr__(self, "width", w)
        need = (int(n) * w + 7) // 8
        if p.size < need:
            raise ValueError(
                f"BitPack needs {need} bytes for n={n} w={w}, got {p.size}")

    @classmethod
    def from_bool(cls, a):
        """bool array-like -> width-1 BitPack (zero-copy impossible: packs)."""
        a = np.ascontiguousarray(np.asarray(a, dtype=bool)).ravel()
        return cls(pack_bits(a), a.size, 1)

    @classmethod
    def from_codes(cls, codes, width):
        """int codes -> width-w BitPack (range-checked, explicit on overflow)."""
        c = np.ascontiguousarray(np.asarray(codes).ravel())
        return cls(pack_codes(c, width), c.size, check_width(width))

    # -- buffer identity (compat with old bool-ndarray buffers) --
    @property
    def dtype(self):
        return np.dtype(bool) if self.width == 1 else np.dtype(np.uint8)

    @property
    def shape(self):
        return (self.n,)

    @property
    def ndim(self):
        return 1

    @property
    def size(self):
        return self.n

    @property
    def nbytes(self):
        return int(self.packed.nbytes)

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"BitPack(w={self.width}, n={self.n})"

    # -- materialization (explicit boundary; filter take is the proven user) --
    def to_array(self):
        """Unpack to bool (w=1) or uint8 codes (w>1). New buffer, packed kept."""
        if self.width == 1:
            return unpack_bits(self.packed, self.n)
        return unpack_codes(self.packed, self.n, self.width)

    def __array__(self, dtype=None, copy=None):
        a = self.to_array()
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        return a

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if any(isinstance(v, BitPack) for v in kwargs.get("out", ()) or ()):
            return NotImplemented  # no in-place into packed: fail loud
        conv = [x.to_array() if isinstance(x, BitPack) else x for x in inputs]
        kw = {k: (v.to_array() if isinstance(v, BitPack) else v)
              for k, v in kwargs.items()}
        return getattr(ufunc, method)(*conv, **kw)

    def tolist(self):
        return self.to_array().tolist()

    def __iter__(self):
        return iter(self.to_array())

    def __getitem__(self, idx):
        if isinstance(idx, BitPack):
            idx = idx.to_array()
        if isinstance(idx, (int, np.integer)):
            i = int(idx) + self.n if int(idx) < 0 else int(idx)
            if not 0 <= i < self.n:
                raise IndexError(f"BitPack index {idx!r} out of range n={self.n}")
            v = self.to_array()[i]
            return bool(v) if self.width == 1 else int(v)
        got = self.to_array()[idx]
        got = np.ascontiguousarray(np.asarray(got).ravel())
        if self.width == 1:
            return BitPack(pack_bits(got.astype(bool, copy=False)), got.size, 1)
        return BitPack(pack_codes(got, self.width), got.size, self.width)

    def astype(self, dtype, copy=True):
        return np.asarray(self).astype(dtype, copy=copy)

    def min(self):
        return self.to_array().min()

    def max(self):
        return self.to_array().max()

    # -- elementwise operators (unpack-on-use, same semantics as bool arrays) --
    def _bo(self, other, op):
        a = self.to_array()
        b = other.to_array() if isinstance(other, BitPack) else other
        return getattr(a, op)(b)

    def __eq__(self, other):
        return self._bo(other, "__eq__")

    def __ne__(self, other):
        return self._bo(other, "__ne__")

    def __lt__(self, other):
        return self._bo(other, "__lt__")

    def __le__(self, other):
        return self._bo(other, "__le__")

    def __gt__(self, other):
        return self._bo(other, "__gt__")

    def __ge__(self, other):
        return self._bo(other, "__ge__")

    def __add__(self, other):
        return self._bo(other, "__add__")

    def __sub__(self, other):
        return self._bo(other, "__sub__")

    def __mul__(self, other):
        return self._bo(other, "__mul__")

    def __truediv__(self, other):
        return self._bo(other, "__truediv__")

    def __pow__(self, other):
        return self._bo(other, "__pow__")

    # -- pure-bit boolean combine (no unpack, never int32) --
    def __and__(self, other):
        return bit_and(self, other)

    def __or__(self, other):
        return bit_or(self, other)

    def __invert__(self):
        return bit_not(self)


BitMask = BitPack  # BoolMask is the width-1 enum in the bit suitcase


def _as_packed_bits(x, what="mask input"):
    """BitPack(w1) or bool array-like -> (packed uint8, n); else TypeError."""
    if isinstance(x, BitPack):
        if x.width != 1:
            raise TypeError(f"{what} must be bool, got width-{x.width} enum")
        return np.ascontiguousarray(x.packed), x.n
    a = np.asarray(x)
    if a.dtype != np.dtype(bool):
        raise TypeError(f"{what} must be bool, got {a.dtype}")
    a = np.ascontiguousarray(a.ravel())
    return pack_bits(a), a.size


def bit_and(a, b):
    """Packed AND (word ops on bytes); size mismatch -> ValueError."""
    pa, na = _as_packed_bits(a)
    pb, nb = _as_packed_bits(b)
    if na != nb:
        raise ValueError(f"mask size mismatch {na} != {nb}")
    return BitPack(np.bitwise_and(pa, pb), na, 1)


def bit_or(a, b):
    """Packed OR (word ops on bytes); size mismatch -> ValueError."""
    pa, na = _as_packed_bits(a)
    pb, nb = _as_packed_bits(b)
    if na != nb:
        raise ValueError(f"mask size mismatch {na} != {nb}")
    return BitPack(np.bitwise_or(pa, pb), na, 1)


def bit_not(a):
    """Packed NOT (word op + tail re-mask so padding stays 0)."""
    pa, na = _as_packed_bits(a)
    out = np.bitwise_not(pa)
    if out.size:
        out[-1] &= np.uint8(_tail_mask(na))
    return BitPack(out, na, 1)


def combine(a, b, op):
    """combine(a, b, op) -> BitPack; same error contract as the bool reference."""
    if op == "and":
        if b is None:
            raise ValueError("mask 'and' needs two masks")
        return bit_and(a, b)
    if op == "or":
        if b is None:
            raise ValueError("mask 'or' needs two masks")
        return bit_or(a, b)
    if op == "not":
        if b is not None:
            raise ValueError("mask 'not' takes a single input")
        return bit_not(a)
    raise ValueError(f"unknown mask op '{op}'")


def to_bool_array(m):
    """Packed or bool mask -> fresh bool ndarray (the one proven unpack)."""
    if isinstance(m, BitPack):
        if m.width != 1:
            raise TypeError(f"filter mask must be bool, got width-{m.width} enum")
        return m.to_array()
    a = np.asarray(m)
    if a.dtype != np.dtype(bool):
        raise TypeError(f"filter mask must be bool, got {a.dtype}")
    return np.ascontiguousarray(a.ravel())
