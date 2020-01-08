"""Test wrangler interfaces.

"""

import pytest

from pywrangler import wranglers
from pywrangler.util.testing.util import concretize_abstract_wrangler


@pytest.fixture(scope="module")
def ii_kwargs():

    return {"marker_column": "marker_col",
            "marker_start": "start",
            "marker_end": "end",
            "marker_start_use_first": False,
            "marker_end_use_first": True,
            "order_columns": ["col1", "col2"],
            "groupby_columns": ["col3", "col4"],
            "ascending": [True, False],
            "target_column_name": "abc"}


@pytest.fixture()
def concrete_wrangler():

    return concretize_abstract_wrangler(wranglers.IntervalIdentifier)


def test_base_interval_identifier_init(ii_kwargs, concrete_wrangler):

    wrangler = concrete_wrangler
    bii = wrangler(**ii_kwargs)

    assert bii.get_params() == ii_kwargs


def test_base_interval_identifier_sort_length_exc(ii_kwargs, concrete_wrangler):

    incorrect_length = ii_kwargs.copy()
    incorrect_length["ascending"] = (True, )

    wrangler = concrete_wrangler

    with pytest.raises(ValueError):
        wrangler(**incorrect_length)


def test_base_interval_identifier_sort_keyword_exc(ii_kwargs, concrete_wrangler):

    incorrect_keyword = ii_kwargs.copy()
    incorrect_keyword["ascending"] = ("wrong keyword", "wrong keyword too")

    wrangler = concrete_wrangler

    with pytest.raises(ValueError):
        wrangler(**incorrect_keyword)


def test_base_interval_identifier_identical_markers(ii_kwargs, concrete_wrangler):

    kwargs = ii_kwargs.copy()
    del kwargs["marker_end"]

    wrangler = concrete_wrangler(**kwargs)

    assert wrangler._identical_start_end_markers is True


def test_base_interval_identifier_identical_start_end_markers(ii_kwargs, concrete_wrangler):

    kwargs = ii_kwargs.copy()
    kwargs["marker_end"] = kwargs["marker_start"]

    wrangler = concrete_wrangler(**kwargs)

    assert wrangler._identical_start_end_markers is True


def test_base_interval_identifier_non_identical_markers(ii_kwargs, concrete_wrangler):

    wrangler = concrete_wrangler(**ii_kwargs)

    assert wrangler._identical_start_end_markers is False
