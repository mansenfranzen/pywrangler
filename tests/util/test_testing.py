"""This module contains tests for testing utilities.

"""

import datetime

import pytest

import pandas as pd
from pandas.api import types

from pywrangler.base import BaseWrangler
from pywrangler.util.testing import (
    NULL,
    NaN,
    TestDataTable,
    concretize_abstract_wrangler
)


def test_concretize_abstract_wrangler():
    class Dummy(BaseWrangler):
        """Doc"""

        @property
        def computation_engine(self) -> str:
            return "engine"

    concrete_class = concretize_abstract_wrangler(Dummy)
    instance = concrete_class()

    assert instance.computation_engine == "engine"
    assert instance.__doc__ == "Doc"
    assert instance.__class__.__name__ == "Dummy"

    with pytest.raises(NotImplementedError):
        instance.preserves_sample_size


@pytest.fixture
def data_table_miss():
    cols = ["int", "float", "bool", "str", "datetime"]
    data = [[1, 1.1, True, "string", "2019-01-01 10:00:00"],
            [2, NaN, False, "string2", "2019-02-01 10:00:00"],
            [NULL, NULL, NULL, NULL, NULL]]

    return TestDataTable(data=data, dtypes=cols, columns=cols)


@pytest.fixture
def data_table():
    cols = ["int", "float", "bool", "str", "datetime"]
    data = [[1, 1.1, True, "string", "2019-01-01 10:00:00"],
            [2, 2, False, "string2", "2019-02-01 10:00:00"]]

    return TestDataTable(data=data, dtypes=cols, columns=cols)


def test_spark_converter(data_table_miss):
    df = data_table_miss.to_pyspark()

    dtypes = dict(df.dtypes)
    assert dtypes["int"] == "int"
    assert dtypes["float"] == "double"
    assert dtypes["bool"] == "boolean"
    assert dtypes["str"] == "string"
    assert dtypes["datetime"] == "timestamp"

    res = df.collect()
    assert res[0].int == 1
    assert res[2].int is None

    assert res[0].float == 1.1
    assert pd.isnull(res[1].float)
    assert res[2].float is None

    assert res[0].bool is True
    assert res[2].bool is None

    assert res[0].str == "string"
    assert res[2].str is None

    assert res[0].datetime == datetime.datetime(2019, 1, 1, 10)
    assert res[2].datetime is None


def test_pandas_converter(data_table):
    df = data_table.to_pandas()

    assert types.is_integer_dtype(df["int"])
    assert df["int"][0] == 1
    assert df["int"].isnull().sum() == 0

    assert types.is_float_dtype(df["float"])
    assert df["float"].isnull().sum() == 0
    assert df["float"][1] == 2.0

    assert types.is_bool_dtype(df["bool"])
    assert df["bool"][0] is True
    assert df["bool"].isnull().sum() == 0

    assert types.is_object_dtype(df["str"])
    assert df["str"].isnull().sum() == 0
    assert df["str"][0] == "string"

    assert types.is_datetime64_dtype(df["datetime"])
    assert df["datetime"].isnull().sum() == 0
    assert df["datetime"][0] == pd.Timestamp("2019-01-01 10:00:00")


def test_pandas_converter_missings(data_table_miss):
    df = data_table_miss.to_pandas()

    assert types.is_float_dtype(df["int"])
    assert df["int"][0] == 1.0
    assert pd.isnull(df["int"][2])
    assert df["int"].isnull().sum() == 1

    assert df["float"].isnull().sum() == 2
    assert df["float"][0] == 1.1
    assert pd.isnull(df["float"][2])

    assert types.is_float_dtype(df["bool"])
    assert df["bool"][0] == 1.0
    assert df["bool"].isnull().sum() == 1

    assert types.is_object_dtype(df["str"])
    assert df["str"].isnull().sum() == 1
    assert df["str"][0] == "string"

    assert types.is_datetime64_dtype(df["datetime"])
    assert df["datetime"].isnull().sum() == 1
    assert df["datetime"][0] == pd.Timestamp("2019-01-01 10:00:00")
    assert df["datetime"][2] is pd.NaT


def test_data_table_assertions():
    # unequal elements per row
    with pytest.raises(ValueError):
        TestDataTable(data=[[1, 2],
                            [1]],
                      columns=["a", "b"],
                      dtypes=["int", "int"])

    # mismatch between number of columns and entries per row
    with pytest.raises(ValueError):
        TestDataTable(data=[[1, 2],
                            [1, 2]],
                      columns=["a"],
                      dtypes=["int", "int"])

    # mismatch between number of dtypes and entries per row
    with pytest.raises(ValueError):
        TestDataTable(data=[[1, 2],
                            [1, 2]],
                      columns=["a", "b"],
                      dtypes=["int"])

    # incorrect dtypes
    with pytest.raises(ValueError):
        TestDataTable(data=[[1, 2],
                            [1, 2]],
                      columns=["a", "b"],
                      dtypes=["int", "bad_type"])

    # type errors conversion
    with pytest.raises(TypeError):
        TestDataTable(data=[[1, 2],
                            [1, 2]],
                      columns=["a", "b"],
                      dtypes=["int", "str"])

    with pytest.raises(TypeError):
        TestDataTable(data=[[1, 2],
                            [1, 2]],
                      columns=["a", "b"],
                      dtypes=["int", "bool"])

    with pytest.raises(TypeError):
        TestDataTable(data=[["1", 2],
                            ["1", 2]],
                      columns=["a", "b"],
                      dtypes=["float", "int"])

    with pytest.raises(TypeError):
        TestDataTable(data=[["1", 2],
                            ["1", 2]],
                      columns=["a", "b"],
                      dtypes=["str", "str"])

    with pytest.raises(TypeError):
        TestDataTable(data=[[True, 2],
                            [False, 2]],
                      columns=["a", "b"],
                      dtypes=["datetime", "int"])

    # correct implementation should not raise
    TestDataTable(data=[[1, 2],
                        [1, 2]],
                  columns=["a", "b"],
                  dtypes=["int", "int"])
