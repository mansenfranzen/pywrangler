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
            "orderby_columns": ["col1", "col2"],
            "groupby_columns": ["col3", "col4"],
            "ascending": [True, False],
            "target_column_name": "abc",
            "result_type": "raw"}


@pytest.fixture()
def interval_identifier():
    return concretize_abstract_wrangler(wranglers.IntervalIdentifier)


def test_base_interval_identifier_init(ii_kwargs, interval_identifier):
    wrangler = interval_identifier
    bii = wrangler(**ii_kwargs)

    assert bii.get_params() == ii_kwargs


def test_base_interval_identifier_forced_ascending(ii_kwargs,
                                                   interval_identifier):
    forced_ascending = ii_kwargs.copy()
    forced_ascending["ascending"] = None

    wrangler = interval_identifier
    bii = wrangler(**forced_ascending)

    assert bii.ascending == [True, True]


def test_base_interval_identifier_sort_length_exc(ii_kwargs,
                                                  interval_identifier):
    incorrect_length = ii_kwargs.copy()
    incorrect_length["ascending"] = (True,)

    wrangler = interval_identifier

    with pytest.raises(ValueError):
        wrangler(**incorrect_length)


def test_base_interval_identifier_sort_keyword_exc(ii_kwargs,
                                                   interval_identifier):
    incorrect_keyword = ii_kwargs.copy()
    incorrect_keyword["ascending"] = ("wrong keyword", "wrong keyword too")

    wrangler = interval_identifier

    with pytest.raises(ValueError):
        wrangler(**incorrect_keyword)


def test_base_interval_identifier_identical_markers(ii_kwargs,
                                                    interval_identifier):
    kwargs = ii_kwargs.copy()
    del kwargs["marker_end"]

    wrangler = interval_identifier(**kwargs)

    assert wrangler._identical_start_end_markers is True


def test_base_interval_identifier_identical_start_end_markers(ii_kwargs,
                                                              interval_identifier):
    kwargs = ii_kwargs.copy()
    kwargs["marker_end"] = kwargs["marker_start"]

    wrangler = interval_identifier(**kwargs)

    assert wrangler._identical_start_end_markers is True


def test_base_interval_identifier_non_identical_markers(ii_kwargs,
                                                        interval_identifier):
    wrangler = interval_identifier(**ii_kwargs)

    assert wrangler._identical_start_end_markers is False


def test_base_interval_identifier_result_type(ii_kwargs, interval_identifier):
    kwargs = ii_kwargs.copy()
    kwargs["result_type"] = "does not exist"

    with pytest.raises(ValueError):
        interval_identifier(**kwargs)
