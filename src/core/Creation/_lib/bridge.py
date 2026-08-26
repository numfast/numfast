"""Мост Creation -> Series (S208).

to_series(created) -> NumericSeries через публичный конструктор (pack_rows).
NumericSeries FROZEN -- не трогаем. Zero-copy GPU-resident view --
deferred D-3 (после unfreeze NumericSeries).
"""


def to_series(created):
    """CreatedArray -> NumericSeries (публичный конструктор, pack_rows)."""
    from _core.backend import set_active
    from _core.context import create_context
    from Series._lib.numeric_series import NumericSeries

    set_active("numpy")
    ctx = create_context("creation")
    return NumericSeries(created.raw.tolist(), ctx)
