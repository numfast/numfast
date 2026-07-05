"""NumFast Vis — GPU-accelerated / browser-rendered charts."""

from ._renderer import Chart as _ServerChart
from ._client import ClientChart, SliceChart


def Chart(*args, mode="client", **kwargs):
    """Create a chart.

    Parameters
    ----------
    mode : str
        'client' — full data to browser, local viewport (instant interaction).
        'slice' — data on server, CPU-downsampled window slices to browser.
        'server' — render on server GPU via wgpu (for headless/batch).
    """
    if mode == "slice":
        return SliceChart(*args, **kwargs)
    if mode == "client":
        return ClientChart(*args, **kwargs)
    return _ServerChart(*args, **kwargs)


__all__ = ["Chart", "ClientChart", "SliceChart"]
