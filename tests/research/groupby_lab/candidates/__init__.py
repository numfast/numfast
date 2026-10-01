# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Registry: one contract, fifteen candidates (A-F + G-O)."""

from . import a_full_unique as _a
from . import b_partial as _b
from . import c_hashpart as _c
from . import d_sortedmerge as _d
from . import e_tree as _e
from . import f_hybrid as _f
from . import g_robinhood as _g
from . import h_hopscotch as _h
from . import i_swiss as _i
from . import j_quotient as _j
from . import k_twochoice as _k
from . import l_blocked as _l
from . import m_flat as _m
from . import n_batched as _n
from . import o_hybrid as _o

ALGOS = {
    "A": _a,
    "B": _b,
    "C": _c,
    "D": _d,
    "E": _e,
    "F": _f,
    "G": _g,
    "H": _h,
    "I": _i,
    "J": _j,
    "K": _k,
    "L": _l,
    "M": _m,
    "N": _n,
    "O": _o,
}


def get(name):
    return ALGOS[name]
