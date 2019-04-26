"""This module contains tests for the helper module.

"""

from pywrangler.util.helper import get_param_names


def test_get_param_names():

    def func():
        pass

    assert get_param_names(func) == []

    def func1(a, b=4, c=6):
        pass

    assert get_param_names(func1) == ["a", "b", "c"]
    assert get_param_names(func1, ["a"]) == ["b", "c"]
