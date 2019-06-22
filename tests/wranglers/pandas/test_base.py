"""Test pandas base wrangler.

"""

import pytest

import pandas as pd

from pywrangler.wranglers.pandas.base import PandasWrangler


def test_pandas_base_wrangler_engine():

    wrangler = PandasWrangler()

    assert wrangler.computation_engine == "pandas"


@pytest.mark.parametrize("preserves_sample_size", [True, False])
def test_pandas_wrangler_validate_output_shape_raises(preserves_sample_size):

    class DummyWrangler(PandasWrangler):
        @property
        def preserves_sample_size(self):
            return preserves_sample_size

    wrangler = DummyWrangler()

    df1 = pd.DataFrame([0]*10)
    df2 = pd.DataFrame([0]*20)

    if preserves_sample_size:
        with pytest.raises(ValueError):
            wrangler.validate_output_shape(df1, df2)
    else:
        wrangler.validate_output_shape(df1, df2)


def test_pandas_wrangler_validate_empty_df_raises():

    df = pd.DataFrame()

    with pytest.raises(ValueError):
        PandasWrangler.validate_empty_df(df)


def test_pandas_wrangler_validate_empty_df_not_raises():

    df = pd.DataFrame([0, 0])

    PandasWrangler.validate_empty_df(df)


def test_pandas_wrangler_validate_columns_raises():

    df = pd.DataFrame(columns=["col1", "col2"])

    with pytest.raises(ValueError):
        PandasWrangler.validate_columns(df, ("col3", "col1"))


def test_pandas_wrangler_validate_columns_not_raises():

    df = pd.DataFrame(columns=["col1", "col2"])

    PandasWrangler.validate_columns(df, ("col1", "col2"))
