"""Test wrangler interfaces.

"""

import pytest

from pywrangler.wranglers import interfaces


@pytest.fixture(scope="module")
def interval_ident_kwargs():

    return {"marker_column": "marker_col",
            "marker_start": "start",
            "marker_end": "end",
            "order_columns": ("col1", "col2"),
            "groupby_columns": ("col3", "col4"),
            "sort_order": ("ascending", "descending"),
            "target_column_name": "abc"}


def test_base_interval_identifier_init(interval_ident_kwargs):

    bii = interfaces.IntervalIdentifier(**interval_ident_kwargs)

    assert bii.get_params() == interval_ident_kwargs


def test_base_interval_identifier_sort_length_exc(interval_ident_kwargs):

    incorrect_length = interval_ident_kwargs.copy()
    incorrect_length["sort_order"] = ("descending", )

    with pytest.raises(ValueError):
        interfaces.IntervalIdentifier(**incorrect_length)


def test_base_interval_identifier_sort_keyword_exc(interval_ident_kwargs):

    incorrect_keyword = interval_ident_kwargs.copy()
    incorrect_keyword["sort_order"] = ("wrong keyword", "wrong keyword too")

    with pytest.raises(ValueError):
        interfaces.IntervalIdentifier(**incorrect_keyword)
