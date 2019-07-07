"""This module contains pandas wrangler utility tests.

"""

import pytest

import pandas as pd

from pywrangler.pandas import util


def test_pandas_wrangler_validate_empty_df_raises():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        util.validate_empty_df(df)


def test_pandas_wrangler_validate_empty_df_not_raises():
    df = pd.DataFrame([0, 0])

    util.validate_empty_df(df)


def test_pandas_wrangler_validate_columns_raises():
    df = pd.DataFrame(columns=["col1", "col2"])

    with pytest.raises(ValueError):
        util.validate_columns(df, ("col3", "col1"))


def test_pandas_wrangler_validate_columns_not_raises():
    df = pd.DataFrame(columns=["col1", "col2"])

    util.validate_columns(df, ("col1", "col2"))
