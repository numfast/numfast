"""Vis Extension — GPU-accelerated visualization.

Builder entry point.
"""


def Chart(*args, mode="client", **kwargs):
    from Vis._lib._client import ClientChart, SliceChart
    from Vis._lib._renderer import Chart as _ServerChart
    if mode == "slice":
        return SliceChart(*args, **kwargs)
    if mode == "client":
        return ClientChart(*args, **kwargs)
    return _ServerChart(*args, **kwargs)


__all__ = ["Chart", "ClientChart", "SliceChart"]
