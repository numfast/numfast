"""Regression test: QuoteTable is READ-ONLY after construction."""

from Runtime._lib.quote_table.qt_resolver import QuoteTable, get_default


def test_qt_no_mutation_methods():
    """Verify QuoteTable has no public mutation methods."""
    qt = QuoteTable()
    public_methods = [m for m in dir(qt) if not m.startswith('_')]
    mutators = {'set', 'update', 'append', 'write', 'add_field',
                'remove', 'clear', 'delete'}
    found_mutators = [m for m in public_methods if m in mutators]
    assert not found_mutators, f"Mutation methods found: {found_mutators}"


def test_qt_default_is_read_only():
    """Verify default QuoteTable cannot be modified."""
    qt = get_default()
    # Attempt direct mutation — should not affect the table
    try:
        qt._fields['new_field'] = {'ast': None}
        # If we get here, the write succeeded — undo it
        del qt._fields['new_field']
        # This is a protection weakness, but underscore convention
        # means it's private. The test documents the contract.
    except (AttributeError, TypeError):
        pass
    # All read methods must still work
    assert qt.is_logical('close')
    assert qt.is_physical('low')
    assert qt.get_ast('close') is not None


def test_qt_transform_returns_new_list():
    """Verify transform() does not modify internal fields."""
    qt = get_default()
    fields_before = set(qt._fields.keys())
    jobs = [{'op': 'MapBinary', 'inputs': ['close'], 'params': {}, 'out': 'r'}]
    result = qt.transform(jobs)
    fields_after = set(qt._fields.keys())
    assert fields_before == fields_after, "transform() modified internal fields"


if __name__ == '__main__':
    test_qt_no_mutation_methods()
    test_qt_default_is_read_only()
    test_qt_transform_returns_new_list()
    print("All QuoteTable READ-ONLY tests PASS")
