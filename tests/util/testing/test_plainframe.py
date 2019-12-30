"""This module contains PlainFrame and PlainColumn tests.

"""
import collections
import datetime

import pandas as pd
import numpy as np
import pytest
from pywrangler.util.testing.plainframe import PlainFrame, NULL, NaN, \
    ConverterFromPandas

from numpy.testing import assert_equal as np_assert_equal


@pytest.fixture
def plainframe_standard():
    cols = ["int", "float", "bool", "str", "datetime"]
    data = [[1, 1.1, True, "string", "2019-01-01 10:00:00"],
            [2, 2, False, "string2", "2019-02-01 10:00:00"]]

    return PlainFrame(data=data, dtypes=cols, columns=cols)


@pytest.fixture
def plainframe_missings():
    cols = ["int", "float", "bool", "str", "datetime"]
    data = [[1, 1.1, True, "string", "2019-01-01 10:00:00"],
            [2, NaN, False, "string2", "2019-02-01 10:00:00"],
            [NULL, NULL, NULL, NULL, NULL]]

    return PlainFrame(data=data, dtypes=cols, columns=cols)


@pytest.fixture
def df_from_pandas():
    df = pd.DataFrame(
        {"int": [1, 2],
         "int_na": [1, np.NaN],
         "bool": [True, False],
         "bool_na": [True, np.NaN],
         "float": [1.2, 1.3],
         "float_na": [1.2, np.NaN],
         "str": ["foo", "bar"],
         "str_na": ["foo", np.NaN],
         "datetime": [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-02")],
         "datetime_na": [pd.Timestamp("2019-01-01"), pd.NaT]})

    return df


@pytest.fixture
def df_from_spark(spark):
    from pyspark.sql import types

    values = collections.OrderedDict(
        {"int": [1, 2, None],
         "smallint": [1, 2, None],
         "bigint": [1, 2, None],
         "bool": [True, False, None],
         "single": [1.0, NaN, None],
         "double": [1.0, NaN, None],
         "str": ["foo", "bar", None],
         "datetime": [datetime.datetime(2019, 1, 1),
                      datetime.datetime(2019, 1, 2),
                      None],
         "date": [datetime.date(2019, 1, 1),
                  datetime.date(2019, 1, 2),
                  None],
         "map": [{"foo": "bar"}, {"bar": "foo"}, None],
         "array": [[1, 2, 3], [3, 4, 5], None]}
    )

    data = list(zip(*values.values()))

    c = types.StructField
    columns = [c("int", types.IntegerType()),
               c("smallint", types.ShortType()),
               c("bigint", types.LongType()),
               c("bool", types.BooleanType()),
               c("single", types.FloatType()),
               c("double", types.DoubleType()),
               c("str", types.StringType()),
               c("datetime", types.TimestampType()),
               c("date", types.DateType()),
               c("map", types.MapType(types.StringType(), types.StringType())),
               c("array", types.ArrayType(types.IntegerType()))]

    schema = types.StructType(columns)

    return spark.createDataFrame(data, schema)


def create_plain_frame(cols, rows, reverse_cols=False, reverse_rows=False):
    """Helper function to automatically create instances of PlainFrame.

    `cols` contains typed column annotations like "col1:int".
    """

    if reverse_cols:
        cols = cols[::-1]

    columns, dtypes = zip(*[col.split(":") for col in cols])

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

    return PlainFrame(data=data,
                      dtypes=dtypes,
                      columns=columns)


def create_plainframe_single(values, dtype):
    """Create some special scenarios more easily. Always assumes a single
    column with identical name. Only values and dtype varies.

    """

    data = [[x] for x in values]
    dtypes = [dtype]
    columns = ["name"]

    return PlainFrame(data=data, dtypes=dtypes, columns=columns)


def test_plainframe_init(plainframe_missings):
    df = plainframe_missings
    col_values = lambda x: df.get_column(x).values

    assert df.columns == ["int", "float", "bool", "str", "datetime"]
    assert df.dtypes == ["int", "float", "bool", "str", "datetime"]
    assert col_values("int") == (1, 2, NULL)
    assert col_values("str") == ("string", "string2", NULL)
    assert col_values("datetime")[0] == datetime.datetime(2019, 1, 1, 10)


def test_plainframe_init_assertions():
    # unequal elements per row
    with pytest.raises(ValueError):
        PlainFrame(data=[[1, 2],
                         [1]],
                   columns=["a", "b"],
                   dtypes=["int", "int"])

    # mismatch between number of columns and entries per row
    with pytest.raises(ValueError):
        PlainFrame(data=[[1, 2],
                         [1, 2]],
                   columns=["a"],
                   dtypes=["int", "int"])

    # mismatch between number of dtypes and entries per row
    with pytest.raises(ValueError):
        PlainFrame(data=[[1, 2],
                         [1, 2]],
                   columns=["a", "b"],
                   dtypes=["int"])

    # incorrect dtypes
    with pytest.raises(ValueError):
        PlainFrame(data=[[1, 2],
                         [1, 2]],
                   columns=["a", "b"],
                   dtypes=["int", "bad_type"])

    # type errors conversion
    with pytest.raises(TypeError):
        PlainFrame(data=[[1, 2],
                         [1, 2]],
                   columns=["a", "b"],
                   dtypes=["int", "str"])

    with pytest.raises(TypeError):
        PlainFrame(data=[[1, 2],
                         [1, 2]],
                   columns=["a", "b"],
                   dtypes=["int", "bool"])

    with pytest.raises(TypeError):
        PlainFrame(data=[["1", 2],
                         ["1", 2]],
                   columns=["a", "b"],
                   dtypes=["float", "int"])

    with pytest.raises(TypeError):
        PlainFrame(data=[["1", 2],
                         ["1", 2]],
                   columns=["a", "b"],
                   dtypes=["str", "str"])

    with pytest.raises(TypeError):
        PlainFrame(data=[[True, 2],
                         [False, 2]],
                   columns=["a", "b"],
                   dtypes=["datetime", "int"])

    # correct implementation should not raise
    PlainFrame(data=[[1, 2],
                     [1, 2]],
               columns=["a", "b"],
               dtypes=["int", "int"])


def test_assert_equal_basics():
    # equal values should be equal
    left = create_plain_frame(["a:int", "b:int"], 10)
    right = create_plain_frame(["a:int", "b:int"], 10)
    left.assert_equal(right)

    # different values should not be equal
    left = create_plainframe_single([1, 2], "int")
    right = create_plainframe_single([2, 3], "int")
    with pytest.raises(AssertionError):
        left.assert_equal(right)

    # incorrect number of rows
    with pytest.raises(AssertionError):
        left = create_plain_frame(["a:int", "b:int"], 10)
        right = create_plain_frame(["a:int", "b:int"], 5)
        left.assert_equal(right)

    # incorrect number of columns
    with pytest.raises(AssertionError):
        left = create_plain_frame(["a:int"], 10)
        right = create_plain_frame(["a:int", "b:int"], 10)
        left.assert_equal(right)

    # incorrect column_names
    with pytest.raises(AssertionError):
        left = create_plain_frame(["a:int", "c:int"], 10)
        right = create_plain_frame(["a:int", "b:int"], 10)
        left.assert_equal(right)

    # incorrect dtypes
    with pytest.raises(AssertionError):
        left = create_plain_frame(["a:int", "b:str"], 10)
        right = create_plain_frame(["a:int", "b:int"], 10)
        left.assert_equal(right)

    # check column order
    left = create_plain_frame(["a:int", "b:int"], 10, reverse_cols=True)
    right = create_plain_frame(["a:int", "b:int"], 10)
    left.assert_equal(right, assert_column_order=False)

    with pytest.raises(AssertionError):
        left.assert_equal(right, assert_column_order=True)

    # check row order
    left = create_plain_frame(["a:int", "b:int"], 10, reverse_rows=True)
    right = create_plain_frame(["a:int", "b:int"], 10)
    left.assert_equal(right, assert_row_order=False)

    with pytest.raises(AssertionError):
        left.assert_equal(right, assert_row_order=True)


def test_assert_equal_special():
    # nan should be equal
    left = create_plainframe_single([NaN, 1], "float")
    right = create_plainframe_single([NaN, 1], "float")
    left.assert_equal(right)

    # Null should be equal
    left = create_plainframe_single([NULL, 1], "float")
    right = create_plainframe_single([NULL, 1], "float")
    left.assert_equal(right)

    # null should be different from other values
    with pytest.raises(AssertionError):
        left = create_plainframe_single(["2019-01-01"], "datetime")
        right = create_plainframe_single([NULL], "datetime")
        left.assert_equal(right)

    # nan should be different from other values
    with pytest.raises(AssertionError):
        left = create_plainframe_single([1.1], "float")
        right = create_plainframe_single([NaN], "float")
        left.assert_equal(right)


def test_testdatatable_from_dict():
    data = collections.OrderedDict(
        [("col1:int", [1, 2, 3]),
         ("col2:s", ["a", "b", "c"])]
    )

    df = PlainFrame.from_dict(data)

    # check correct column order and dtypes
    np_assert_equal(df.columns, ("col1", "col2"))
    np_assert_equal(df.dtypes, ["int", "str"])

    # check correct values
    np_assert_equal(df.get_column("col1").values, (1, 2, 3))
    np_assert_equal(df.get_column("col2").values, ("a", "b", "c"))


def test_testdatatable_to_dict():
    df = create_plain_frame(["col2:str", "col1:int"], 2)

    to_dict = df.to_dict()
    keys = list(to_dict.keys())
    values = list(to_dict.values())

    # check column order and dtypes
    np_assert_equal(keys, ["col2:str", "col1:int"])

    # check values
    np_assert_equal(values[0], ["1", "2"])
    np_assert_equal(values[1], [1, 2])


def test_testdatatable_getitem_subset():
    df = create_plain_frame(["col1:str", "col2:int", "col3:int"], 2)
    df_sub = create_plain_frame(["col1:str", "col2:int"], 2)

    cmp_kwargs = dict(assert_column_order=True,
                      assert_row_order=True)

    # test list of strings, slice and string
    df["col1", "col2"].assert_equal(df_sub, **cmp_kwargs)
    df["col1":"col2"].assert_equal(df_sub, **cmp_kwargs)
    df["col1"].assert_equal(df_sub["col1"], **cmp_kwargs)

    # test incorrect type
    with pytest.raises(ValueError):
        df[{"col1"}]

    # test invalid column name
    with pytest.raises(ValueError):
        df["non_existant"]


def test_testdatatable_get_column():
    df = create_plain_frame(["col1:str", "col2:int"], 2)
    assert df.get_column("col1") is df._col_dict["col1"]


def test_testdatatable_parse_typed_columns():
    parse = PlainFrame._parse_typed_columns

    # invalid splits
    cols = ["col1:int", "col2"]
    with pytest.raises(ValueError):
        parse(cols)

    # invalid types
    cols = ["col1:asd"]
    with pytest.raises(ValueError):
        parse(cols)

    # invalid abbreviations
    cols = ["col1:a"]
    with pytest.raises(ValueError):
        parse(cols)

    # correct types and columns
    cols = ["col1:str", "col2:s",
            "col3:int", "col4:i",
            "col5:float", "col6:f",
            "col7:bool", "col8:b",
            "col9:datetime", "col10:d"]

    names = ["col{}".format(x) for x in range(1, 11)]
    dtypes = ["str", "str",
              "int", "int",
              "float", "float",
              "bool", "bool",
              "datetime", "datetime"]

    result = (names, dtypes)
    np_assert_equal(parse(cols), result)


def test_inspect_dtype_from_pandas():
    inspect = ConverterFromPandas.inspect_dtype

    # raise if incorret type
    ser = pd.Series("asd", dtype=object)
    with pytest.raises(TypeError):
        inspect(ser)


def test_testdatatable_from_pandas(df_from_pandas):
    df = df_from_pandas
    df_conv = PlainFrame.from_pandas(df)

    # check int to int
    assert df_conv.get_column("int").dtype == "int"
    assert df_conv.get_column("int").values == (1, 2)

    # check bool to bool
    assert df_conv.get_column("bool").dtype == "bool"
    assert df_conv.get_column("bool").values == (True, False)

    # check bool (object) to bool with nan
    assert df_conv.get_column("bool_na").dtype == "bool"
    assert df_conv.get_column("bool_na").values == (True, NULL)

    # check float to float
    assert df_conv.get_column("float").dtype == "float"
    assert df_conv.get_column("float").values == (1.2, 1.3)

    # check float to float with nan
    assert df_conv.get_column("float_na").dtype == "float"
    np_assert_equal(df_conv.get_column("float_na").values, (1.2, NaN))

    # check str to str
    assert df_conv.get_column("str").dtype == "str"
    assert df_conv.get_column("str").values == ("foo", "bar")

    # check str to str with nan
    assert df_conv.get_column("str_na").dtype == "str"
    assert df_conv.get_column("str_na").values == ("foo", NULL)

    # check datetime to datetime
    assert df_conv.get_column("datetime").dtype == "datetime"
    assert df_conv.get_column("datetime").values == \
           (datetime.datetime(2019, 1, 1), datetime.datetime(2019, 1, 2))
    # check datetime to datetime with nan
    assert df_conv.get_column("datetime_na").dtype == "datetime"
    assert df_conv.get_column("datetime_na").values == (
        datetime.datetime(2019, 1, 1), NULL)


def test_testdatatable_from_pandas_special():
    # check mixed dtype raise
    df = pd.DataFrame({"mixed": [1, "foo bar"]})
    with pytest.raises(TypeError):
        PlainFrame.from_pandas(df)

    # check assertion for incorrect forces
    # too many types provided
    with pytest.raises(ValueError):
        PlainFrame.from_pandas(df, dtypes=["int", "str"])

    with pytest.raises(ValueError):
        PlainFrame.from_pandas(df, dtypes={"mixed": "str",
                                           "dummy": "int"})

    # invalid dtypes provided
    with pytest.raises(ValueError):
        PlainFrame.from_pandas(df, dtypes=["not existant type"])

    with pytest.raises(ValueError):
        PlainFrame.from_pandas(df, dtypes={"mixed": "not existant type"})

    # invalid column names provided
    with pytest.raises(ValueError):
        PlainFrame.from_pandas(df, dtypes={"dummy": "str"})

    # check int to forced int with nan
    df = pd.DataFrame({"int": [1, np.NaN]})
    df_conv = PlainFrame.from_pandas(df, dtypes=["int"])
    assert df_conv.get_column("int").dtype == "int"
    assert df_conv.get_column("int").values == (1, NULL)

    # check force int to float
    df = pd.DataFrame({"int": [1, 2]})
    df_conv = PlainFrame.from_pandas(df, dtypes=["float"])
    assert df_conv.get_column("int").dtype == "float"
    assert df_conv.get_column("int").values == (1.0, 2.0)

    # check force float to int
    df = pd.DataFrame({"float": [1.0, 2.0]})
    df_conv = PlainFrame.from_pandas(df, dtypes=["int"])
    assert df_conv.get_column("float").dtype == "int"
    assert df_conv.get_column("float").values == (1, 2)

    # check force str to datetime
    df = pd.DataFrame({"datetime": ["2019-01-01", "2019-01-02"]})
    df_conv = PlainFrame.from_pandas(df, dtypes=["datetime"])
    assert df_conv.get_column("datetime").dtype == "datetime"
    assert df_conv.get_column("datetime").values == \
           (datetime.datetime(2019, 1, 1), datetime.datetime(2019, 1, 2))

    # dtype object with strings and nan should pass correctly
    df = pd.DataFrame({"str": ["foo", "bar", NaN]}, dtype=object)
    df_conv = PlainFrame.from_pandas(df)
    assert df_conv.get_column("str").dtype == "str"
    assert df_conv.get_column("str").values == ("foo", "bar", NULL)


def test_testdatatable_from_pyspark(df_from_spark):
    def select(x):
        from_pyspark = PlainFrame.from_pyspark
        return from_pyspark(df_from_spark.select(x))

    # int columns
    int_columns = ["int", "smallint", "bigint"]
    df = select(int_columns)
    for int_column in int_columns:
        assert df.get_column(int_column).dtype == "int"
        assert df.get_column(int_column).values == (1, 2, NULL)

    # bool column
    df = select("bool")
    assert df.get_column("bool").dtype == "bool"
    assert df.get_column("bool").values == (True, False, NULL)

    # float columns
    float_columns = ["single", "double"]
    df = select(float_columns)
    for float_column in float_columns:
        assert df.get_column(float_column).dtype == "float"
        np_assert_equal(df.get_column(float_column).values, (1.0, NaN, NULL))

    # string column
    df = select("str")
    assert df.get_column("str").dtype == "str"
    assert df.get_column("str").values == ("foo", "bar", NULL)

    # datetime columns
    datetime_columns = ["datetime", "date"]
    df = select(datetime_columns)
    for datetime_column in datetime_columns:
        col = df.get_column(datetime_column)
        assert col.dtype == "datetime"
        assert col.values == (datetime.datetime(2019, 1, 1),
                              datetime.datetime(2019, 1, 2),
                              NULL)

    # unsupported columns
    unsupported_columns = ["map", "array"]
    for unsupported_column in unsupported_columns:
        with pytest.raises(ValueError):
            df = select(unsupported_column)


def test_spark_converter(plainframe_missings):
    df = plainframe_missings.to_pyspark()

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


def test_pandas_converter(plainframe_standard):
    from pandas.api import types
    df = plainframe_standard.to_pandas()

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


def test_pandas_converter_missings(plainframe_missings):
    from pandas.api import types

    df = plainframe_missings.to_pandas()

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
