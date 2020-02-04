"""Test sanitizer functions.

"""

import pytest

from pywrangler.util.sanitizer import ensure_iterable


@pytest.mark.parametrize(argnames="seq_type", argvalues=(list, tuple))
def test_ensure_iterable_number(seq_type):
    test_input = 3
    test_output = seq_type([3])

    assert ensure_iterable(test_input, seq_type) == test_output


@pytest.mark.parametrize(argnames="seq_type", argvalues=(list, tuple))
def test_ensure_iterable_string(seq_type):
    test_input = "test_string"
    test_output = seq_type(["test_string"])

    assert ensure_iterable(test_input, seq_type) == test_output


@pytest.mark.parametrize(argnames="seq_type", argvalues=(list, tuple))
def test_ensure_iterable_strings(seq_type):
    test_input = ["test1", "test2"]
    test_output = seq_type(["test1", "test2"])

    assert ensure_iterable(test_input, seq_type) == test_output


@pytest.mark.parametrize(argnames="seq_type", argvalues=(list, tuple))
def test_ensure_iterable_custom_class(seq_type):
    class Dummy:
        pass

    dummy = Dummy()

    test_input = dummy
    test_output = seq_type([dummy])

    assert ensure_iterable(test_input, seq_type) == test_output


@pytest.mark.parametrize(argnames="seq_type", argvalues=(list, tuple))
def test_ensure_iterable_none(seq_type):

    assert ensure_iterable(None, seq_type) == seq_type()
    assert ensure_iterable(None, seq_type, retain_none=True) is None
