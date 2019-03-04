"""Test sanitizer functions.

"""

from pywrangler.util.sanitizer import ensure_tuple


def test_ensure_tuple_number():

    test_input = 3
    test_output = (3, )

    assert ensure_tuple(test_input) == test_output


def test_ensure_tuple_string():

    test_input = "test_string"
    test_output = ("test_string", )

    assert ensure_tuple(test_input) == test_output


def test_ensure_tuple_strings():

    test_input = ["test1", "test2"]
    test_output = ("test1", "test2")

    assert ensure_tuple(test_input) == test_output


def test_ensure_tuple_custom_class():
    class Dummy:
        pass

    dummy = Dummy()

    test_input = dummy
    test_output = (dummy, )

    assert ensure_tuple(test_input) == test_output
