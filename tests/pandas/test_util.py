"""This module contains pandas wrangler utility tests.

"""

import pytest

import pandas as pd

from pywrangler.pandas import util


def test_validate_empty_df_raises():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        util.validate_empty_df(df)


def test_validate_empty_df_not_raises():
    df = pd.DataFrame([0, 0])

    util.validate_empty_df(df)


def test_validate_columns_raises():
    df = pd.DataFrame(columns=["col1", "col2"])

    with pytest.raises(ValueError):
        util.validate_columns(df, ("col3", "col1"))


def test_validate_columns_not_raises():
    df = pd.DataFrame(columns=["col1", "col2"])

    util.validate_columns(df, ("col1", "col2"))


def test_sort_values():
    values = list(range(10))
    df = pd.DataFrame({"col1": values,
                       "col2": values})

    # no sort order given
    assert df is util.sort_values(df, [], [])

    # sort order
    computed = util.sort_values(df, ["col1"], [False])
    assert df.sort_values("col1", ascending=False).equals(computed)


def test_groupby():
    values = list(range(10))
    df = pd.DataFrame({"col1": values,
                       "col2": values})

    conv = lambda x: {key: value.values.tolist()
                      for key, value in x.groups.items()}

    # no groupby given
    expected = df.groupby([0]*len(values))
    given = util.groupby(df, [])
    assert conv(expected) == conv(given)

    # with groupby
    expected = df.groupby("col1")
    given = util.groupby(df, ["col1"])
    assert conv(expected) == conv(given)

