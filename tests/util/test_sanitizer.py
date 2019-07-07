"""Test sanitizer functions.

"""

import pytest

from pywrangler.util.sanitizer import ensure_iterable


@pytest.mark.parametrize(argnames="type", argvalues=(list, tuple))
def test_ensure_iterable_number(type):
    test_input = 3
    test_output = type([3])

    assert ensure_iterable(test_input, type) == test_output


@pytest.mark.parametrize(argnames="type", argvalues=(list, tuple))
def test_ensure_iterable_string(type):
    test_input = "test_string"
    test_output = type(["test_string"])

    assert ensure_iterable(test_input, type) == test_output


@pytest.mark.parametrize(argnames="type", argvalues=(list, tuple))
def test_ensure_iterable_strings(type):
    test_input = ["test1", "test2"]
    test_output = type(["test1", "test2"])

    assert ensure_iterable(test_input, type) == test_output


@pytest.mark.parametrize(argnames="type", argvalues=(list, tuple))
def test_ensure_iterable_custom_class(type):
    class Dummy:
        pass

    dummy = Dummy()

    test_input = dummy
    test_output = type([dummy])

    assert ensure_iterable(test_input, type) == test_output


@pytest.mark.parametrize(argnames="type", argvalues=(list, tuple))
def test_ensure_iterable_none(type):
    test_input = None
    test_output = None

    assert ensure_iterable(test_input, type) == test_output
