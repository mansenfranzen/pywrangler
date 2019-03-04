"""This module contains the BaseWrangler tests.

"""

import pytest

from pywrangler.wranglers import base


@pytest.fixture(scope="module")
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

    return DummyWrangler("arg_val", "kwarg_val")


def test_base_wrangler_not_implemented():

    wrangler = base.BaseWrangler()

    test_attributes = ("preserves_sample_size", "computation_engine")
    test_methods = ("fit", "transform", "fit_transform")

    for test_attribute in test_attributes:
        with pytest.raises(NotImplementedError):
            getattr(wrangler, test_attribute)

    for test_method in test_methods:
        with pytest.raises(NotImplementedError):
            getattr(wrangler, test_method)()


def test_base_wrangler_get_params(dummy_wrangler):

    test_output = {"arg1": "arg_val", "kwarg1": "kwarg_val"}

    assert dummy_wrangler.get_params() == test_output


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


@pytest.fixture(scope="module")
def interval_ident_kwargs():

    return {"marker_column": "marker_col",
            "marker_start": "start",
            "marker_end": "end",
            "order_columns": ("col1", "col2"),
            "groupby_columns": ("col3", "col4"),
            "sort_order": ("ascending", "descending")}


def test_base_interval_identifier_init(interval_ident_kwargs):

    bii = base.BaseIntervalIdentifier(**interval_ident_kwargs)

    assert bii.get_params() == interval_ident_kwargs


def test_base_interval_identifier_sort_length_exc(interval_ident_kwargs):

    incorrect_length = interval_ident_kwargs.copy()
    incorrect_length["sort_order"] = ("descending", )

    with pytest.raises(ValueError):
        base.BaseIntervalIdentifier(**incorrect_length)


def test_base_interval_identifier_sort_keyword_exc(interval_ident_kwargs):

    incorrect_keyword = interval_ident_kwargs.copy()
    incorrect_keyword["sort_order"] = ("wrong keyword", "wrong keyword too")

    with pytest.raises(ValueError):
        base.BaseIntervalIdentifier(**incorrect_keyword)
