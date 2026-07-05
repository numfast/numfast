"""numfast Trading — trading indicators.

Каждый индикатор — модуль: describe() + cpu().
Runtime владеет ABI, Trading владеет математикой.
"""

from .EMA import describe as _describe_ema, cpu as _cpu_ema, wgsl as _wgsl_ema
from .RSI import describe as _describe_rsi, cpu as _cpu_rsi
from .SMA import describe as _describe_sma, cpu as _cpu_sma
from .MACD import describe as _describe_macd, cpu as _cpu_macd

# Phase 5 — новые модули
from .ATR import describe as _describe_atr, cpu as _cpu_atr
from .ROC import describe as _describe_roc, cpu as _cpu_roc
from .Momentum import describe as _describe_mom, cpu as _cpu_mom
from .VWAP import describe as _describe_vwap, cpu as _cpu_vwap
from .BollingerBands import describe as _describe_bb, cpu as _cpu_bb
from .Stochastic import describe as _describe_stoch, cpu as _cpu_stoch
from .WilliamsR import describe as _describe_wr, cpu as _cpu_wr
from .OBV import describe as _describe_obv, cpu as _cpu_obv
from .CCI import describe as _describe_cci, cpu as _cpu_cci
from .KeltnerChannels import describe as _describe_kc, cpu as _cpu_kc
from .ADX import describe as _describe_adx, cpu as _cpu_adx
from .SuperTrend import describe as _describe_st, cpu as _cpu_st

# Capabilities for Scheduler
_CAPS = {
    # Phase 3 — базовые
    "EMA":  {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "RSI":  {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "SMA":  {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "MACD": {"streaming": False, "workspace": True,  "multi_input": False, "multi_output": True},
    # Phase 5 — новые
    "ATR":  {"streaming": False, "workspace": True,  "multi_input": True,  "multi_output": False},
    "ROC":  {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "Momentum": {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "VWAP": {"streaming": False, "workspace": True,  "multi_input": True,  "multi_output": False},
    "BollingerBands": {"streaming": False, "workspace": True,  "multi_input": False, "multi_output": True},
    "Stochastic": {"streaming": False, "workspace": True,  "multi_input": True,  "multi_output": True},
    "WilliamsR": {"streaming": False, "workspace": False, "multi_input": True,  "multi_output": False},
    "OBV":  {"streaming": False, "workspace": True,  "multi_input": True,  "multi_output": False},
    "CCI":  {"streaming": False, "workspace": True,  "multi_input": True,  "multi_output": False},
    "KeltnerChannels": {"streaming": False, "workspace": True,  "multi_input": True,  "multi_output": True},
    "ADX":  {"streaming": False, "workspace": True,  "multi_input": True,  "multi_output": True},
    "SuperTrend": {"streaming": False, "workspace": True,  "multi_input": True,  "multi_output": True},
}


def register_all(runtime):
    """Register all Trading kernels in a Runtime instance."""
    kernels = [
        ("EMA",  _describe_ema,  _cpu_ema,  _wgsl_ema),
        ("RSI",  _describe_rsi,  _cpu_rsi,  None),
        ("SMA",  _describe_sma,  _cpu_sma,  None),
        ("MACD", _describe_macd, _cpu_macd, None),
        ("ATR",  _describe_atr,  _cpu_atr,  None),
        ("ROC",  _describe_roc,  _cpu_roc,  None),
        ("Momentum", _describe_mom, _cpu_mom, None),
        ("VWAP", _describe_vwap, _cpu_vwap, None),
        ("BollingerBands", _describe_bb, _cpu_bb, None),
        ("Stochastic",     _describe_stoch, _cpu_stoch, None),
        ("WilliamsR",      _describe_wr,    _cpu_wr,    None),
        ("OBV",  _describe_obv, _cpu_obv, None),
        ("CCI",  _describe_cci, _cpu_cci, None),
        ("KeltnerChannels", _describe_kc, _cpu_kc, None),
        ("ADX",  _describe_adx, _cpu_adx, None),
        ("SuperTrend", _describe_st, _cpu_st, None),
    ]
    for alias, desc, cpu_fn, wgsl_src in kernels:
        runtime.register_kernel(
            alias, describe=desc, cpu=cpu_fn, wgsl=wgsl_src, abi_version=1,
            capabilities=_CAPS.get(alias, {}),
        )


__all__ = ["register_all"]
