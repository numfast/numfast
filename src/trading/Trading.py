"""Trading Extension — market indicators.

Builder entry point.
"""


_CAPS = {
    "EMA":  {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "RSI":  {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "SMA":  {"streaming": False, "workspace": False, "multi_input": False, "multi_output": False},
    "MACD": {"streaming": False, "workspace": True,  "multi_input": False, "multi_output": True},
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


def _kernels():
    from trading._lib.ema import describe as _describe_ema, cpu as _cpu_ema
    from trading._lib.rsi import describe as _describe_rsi, cpu as _cpu_rsi
    from trading._lib.sma import describe as _describe_sma, cpu as _cpu_sma
    from trading._lib.macd import describe as _describe_macd, cpu as _cpu_macd
    from trading._lib.atr import describe as _describe_atr, cpu as _cpu_atr
    from trading._lib.roc import describe as _describe_roc, cpu as _cpu_roc
    from trading._lib.momentum import describe as _describe_mom, cpu as _cpu_mom
    from trading._lib.vwap import describe as _describe_vwap, cpu as _cpu_vwap
    from trading._lib.bollingerbands import describe as _describe_bb, cpu as _cpu_bb
    from trading._lib.stochastic import describe as _describe_stoch, cpu as _cpu_stoch
    from trading._lib.williamsr import describe as _describe_wr, cpu as _cpu_wr
    from trading._lib.obv import describe as _describe_obv, cpu as _cpu_obv
    from trading._lib.cci import describe as _describe_cci, cpu as _cpu_cci, wgsl as _wgsl_cci
    from trading._lib.keltnerchannels import describe as _describe_kc, cpu as _cpu_kc
    from trading._lib.adx import describe as _describe_adx, cpu as _cpu_adx
    from trading._lib.supertrend import describe as _describe_st, cpu as _cpu_st

    return [
        ("EMA",  _describe_ema,  _cpu_ema,  None),
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
        ("CCI",  _describe_cci, _cpu_cci, _wgsl_cci),
        ("KeltnerChannels", _describe_kc, _cpu_kc, None),
        ("ADX",  _describe_adx, _cpu_adx, None),
        ("SuperTrend", _describe_st, _cpu_st, None),
    ]


def register_all(runtime):
    for alias, desc, cpu_fn, wgsl_src in _kernels():
        runtime.register_kernel(
            alias, describe=desc, cpu=cpu_fn, wgsl=wgsl_src, abi_version=1,
            capabilities=_CAPS.get(alias, {}),
        )


def setup(kernel):
    register_fn = kernel.alias.get("register_kernel")
    if register_fn is None:
        return

    class _RuntimeProxy:
        def register_kernel(self, alias, describe=None, cpu=None, wgsl=None, abi_version=1, capabilities=None):
            return register_fn(alias, describe=describe, cpu=cpu, wgsl=wgsl)

    register_all(_RuntimeProxy())