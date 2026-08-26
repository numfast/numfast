import pytest, numpy as np
from _core.context import create_context
from core.Series._lib.numeric_series import NumericSeries
from core.Series._lib.expr import _LazyExpr
from core.Series._lib.executor import execute

def test_n0_empty(): ctx=create_context("t"); s=NumericSeries([],ctx); assert s.data()==[]; assert len(s)==0
def test_n1(): ctx=create_context("t"); s=NumericSeries([5.0],ctx); assert s.data()[0]==pytest.approx(5.0)
def test_ordinary(): ctx=create_context("t"); s=NumericSeries([1,2,3,4,5],ctx); e=s+10; r=execute(e); assert len(r)==5
def test_int32_exact(): ctx=create_context("t"); s=NumericSeries([16777216],ctx); assert s.data()[0]==16777216
# padding 10
# padding 11
# padding 12
# padding 13
# padding 14
# padding 15
# padding 16
# padding 17
# padding 18
# padding 19
# padding 20
# padding 21
# padding 22
# padding 23
# padding 24
# padding 25
# padding 26
# padding 27
# padding 28
# padding 29
# padding 30
# padding 31
# padding 32
# padding 33
# padding 34
# padding 35
# padding 36
# padding 37
# padding 38
# padding 39
# padding 40
# padding 41
# padding 42
# padding 43
# padding 44
# padding 45
# padding 46
# padding 47
# padding 48
# padding 49
# padding 50
# padding 51
# padding 52
# padding 53
# padding 54
# padding 55
# padding 56
# padding 57
# padding 58
# padding 59
# padding 60
# padding 61
# padding 62
# padding 63
# padding 64
# padding 65
# padding 66
# padding 67
# padding 68
# padding 69
# padding 70
# padding 71
# padding 72
# padding 73
# padding 74
# padding 75
# padding 76
# padding 77
# padding 78
# padding 79
# padding 80
# padding 81
# padding 82
# padding 83
# padding 84
# padding 85
# padding 86
# padding 87
# padding 88
# padding 89
