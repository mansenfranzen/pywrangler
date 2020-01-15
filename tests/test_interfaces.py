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
            "order_columns": ["col1", "col2"],
            "groupby_columns": ["col3", "col4"],
            "ascending": [True, False],
            "target_column_name": "abc"}


def test_base_interval_identifier_init(ii_kwargs):

    wrangler = concretize_abstract_wrangler(wranglers.IntervalIdentifier)
    bii = wrangler(**ii_kwargs)

    assert bii.get_params() == ii_kwargs


def test_base_interval_identifier_sort_length_exc(ii_kwargs):

    incorrect_length = ii_kwargs.copy()
    incorrect_length["ascending"] = (True, )

    wrangler = concretize_abstract_wrangler(wranglers.IntervalIdentifier)

    with pytest.raises(ValueError):
        wrangler(**incorrect_length)


def test_base_interval_identifier_sort_keyword_exc(ii_kwargs):

    incorrect_keyword = ii_kwargs.copy()
    incorrect_keyword["ascending"] = ("wrong keyword", "wrong keyword too")

    wrangler = concretize_abstract_wrangler(wranglers.IntervalIdentifier)

    with pytest.raises(ValueError):
        wrangler(**incorrect_keyword)
