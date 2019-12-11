"""This module contains tests for testing utilities.

"""

import datetime

import pytest

import pandas as pd
from numpy.testing import assert_equal as np_assert_equal
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
    np_assert_equal(df["bool"][0], True)
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


def create_test_table(cols, rows, reverse_cols=False, reverse_rows=False):
    """Helper function to automatically create instances of TestDataTable.

    """

    if reverse_cols:
        cols = cols[::-1]

    dtypes, columns = zip(*[col.split("_") for col in cols])

    values = list(range(1, rows + 1))
    mapping = {"str": list(map(str, values)),
               "int": values,
               "float": list(map(float, values)),
               "bool": list([x % 2 == 0 for x in values]),
               "datetime": ["2019-01-{:02} 10:00:00".format(x) for x in
                            values]}

    data = [mapping[dtype] for dtype in dtypes]
    data = list(zip(*data))

    if reverse_rows:
        data = data[::-1]

    return TestDataTable(data=data,
                         dtypes=dtypes,
                         columns=columns)


def create_test_table_special(values, dtype):
    """Create some special scenarios more easily. Always assumes a single
    column with identical name. Only values and dtype varies.

    """

    data = [[x] for x in values]
    dtypes = [dtype]
    columns = ["name"]

    return TestDataTable(data=data, dtypes=dtypes, columns=columns)


def test_assert_equal_basics():
    # equal values should be equal
    left = create_test_table(["int_a", "int_b"], 10)
    right = create_test_table(["int_a", "int_b"], 10)
    left.assert_equal(right)

    # different values should not be equal
    left = create_test_table_special([1, 2], "int")
    right = create_test_table_special([2, 3], "int")
    with pytest.raises(AssertionError):
        left.assert_equal(right)

    # incorrect number of rows
    with pytest.raises(AssertionError):
        left = create_test_table(["int_a", "int_b"], 10)
        right = create_test_table(["int_a", "int_b"], 5)
        left.assert_equal(right)

    # incorrect number of columns
    with pytest.raises(AssertionError):
        left = create_test_table(["int_a"], 10)
        right = create_test_table(["int_a", "int_b"], 10)
        left.assert_equal(right)

    # incorrect column_names
    with pytest.raises(AssertionError):
        left = create_test_table(["int_a", "int_c"], 10)
        right = create_test_table(["int_a", "int_b"], 10)
        left.assert_equal(right)

    # incorrect dtypes
    with pytest.raises(AssertionError):
        left = create_test_table(["int_a", "str_b"], 10)
        right = create_test_table(["int_a", "int_b"], 10)
        left.assert_equal(right)

    # check column order
    left = create_test_table(["int_a", "int_b"], 10, reverse_cols=True)
    right = create_test_table(["int_a", "int_b"], 10)
    left.assert_equal(right, assert_column_order=False)

    with pytest.raises(AssertionError):
        left.assert_equal(right, assert_column_order=True)

    # check row order
    left = create_test_table(["int_a", "int_b"], 10, reverse_rows=True)
    right = create_test_table(["int_a", "int_b"], 10)
    left.assert_equal(right, assert_row_order=False)

    with pytest.raises(AssertionError):
        left.assert_equal(right, assert_row_order=True)


def test_assert_equal_special():
    # nan should be equal
    left = create_test_table_special([NaN, 1], "float")
    right = create_test_table_special([NaN, 1], "float")
    left.assert_equal(right)

    # Null should be equal
    left = create_test_table_special([NULL, 1], "float")
    right = create_test_table_special([NULL, 1], "float")
    left.assert_equal(right)

    # null should be different from other values
    with pytest.raises(AssertionError):
        left = create_test_table_special(["2019-01-01"], "datetime")
        right = create_test_table_special([NULL], "datetime")
        left.assert_equal(right)

    # nan should be different from other values
    with pytest.raises(AssertionError):
        left = create_test_table_special([1.1], "float")
        right = create_test_table_special([NaN], "float")
        left.assert_equal(right)
