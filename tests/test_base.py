"""This module contains the BaseWrangler tests.

"""

import pytest

from pywrangler import base
from pywrangler.util.testing.util import concretize_abstract_wrangler


@pytest.fixture(scope="session")
def dummy_wrangler():
    """Create DummyWrangler for testing BaseWrangler.

    """

    class DummyWrangler(base.BaseWrangler):
        def __init__(self, arg1, kwarg1):
            self.arg1 = arg1
            self.kwarg1 = kwarg1

        @property
        def preserves_sample_size(self):
            return True

        @property
        def computation_engine(self):
            return "DummyEngine"

    return concretize_abstract_wrangler(DummyWrangler)("arg_val", "kwarg_val")


def test_base_wrangler_not_implemented():
    with pytest.raises(TypeError):
        base.BaseWrangler()

    empty_wrangler = concretize_abstract_wrangler(base.BaseWrangler)()

    test_attributes = ("preserves_sample_size", "computation_engine")
    test_methods = ("fit", "transform", "fit_transform")

    for test_attribute in test_attributes:
        with pytest.raises(NotImplementedError):
            getattr(empty_wrangler, test_attribute)

    for test_method in test_methods:
        with pytest.raises(NotImplementedError):
            getattr(empty_wrangler, test_method)()


def test_base_wrangler_get_params(dummy_wrangler):
    test_output = {"arg1": "arg_val", "kwarg1": "kwarg_val"}

    assert dummy_wrangler.get_params() == test_output


def test_base_wrangler_get_params_subclassed(dummy_wrangler):
    class SubClass(dummy_wrangler.__class__):
        def __init__(self, *args, new_param=2, **kwargs):
            super().__init__(*args, **kwargs)
            self.new_param = new_param

    test_output = {"arg1": "arg_val", "kwarg1": "kwarg_val", "new_param": 2}

    assert SubClass("arg_val", "kwarg_val").get_params() == test_output


def test_base_wrangler_properties(dummy_wrangler):
    assert dummy_wrangler.preserves_sample_size is True
    assert dummy_wrangler.computation_engine == "DummyEngine"


def test_base_wrangler_set_params(dummy_wrangler):
    dummy_wrangler.set_params(arg1="new_value")

    assert dummy_wrangler.arg1 == "new_value"
    assert dummy_wrangler.kwarg1 == "kwarg_val"


def test_base_wrangler_set_params_exception(dummy_wrangler):
    with pytest.raises(ValueError):
        dummy_wrangler.set_params(not_exist=0)
