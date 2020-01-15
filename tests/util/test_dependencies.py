"""This module contains tests for the dependencies module.

"""
import pytest
from pywrangler.util import dependencies


def test_raise_if_missing():
    # test non raising for available package
    dependencies.raise_if_missing("collections")

    # test raising for missing package
    with pytest.raises(ImportError):
        dependencies.raise_if_missing("not_existent_package_name123")


def test_requires():
    def func(value, a, b=1):
        return value + a + b

    # test non raising for available package
    decorated = dependencies.requires("collections")(func)
    assert decorated(1, 1, 2) == 4

    # test raising for missing package
    decorated = dependencies.requires("not_existent_package_name123")(func)
    with pytest.raises(ImportError):
        decorated(1, 1, 2)


def test_is_available():
    assert dependencies.is_available("collections") is True
    assert dependencies.is_available("not_existent_package_name123") is False
