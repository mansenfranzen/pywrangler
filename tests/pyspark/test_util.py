"""This module contains pyspark wrangler utility tests.

isort:skip_file
"""

import pytest
import pandas as pd
from pywrangler.pyspark.util import ColumnCacher
from pyspark.sql import functions as F

pytestmark = pytest.mark.pyspark  # noqa: E402
pyspark = pytest.importorskip("pyspark")  # noqa: E402

from pywrangler.pyspark import util


def test_ensure_column(spark):
    assert str(F.col("a")) == str(util.ensure_column("a"))
    assert str(F.col("a")) == str(util.ensure_column(F.col("a")))


def test_spark_wrangler_validate_columns_raises(spark):

    data = {"col1": [1, 2], "col2": [3, 4]}
    df = spark.createDataFrame(pd.DataFrame(data))

    with pytest.raises(ValueError):
        util.validate_columns(df, ("col3", "col1"))


def test_spark_wrangler_validate_columns_not_raises(spark):

    data = {"col1": [1, 2], "col2": [3, 4]}
    df = spark.createDataFrame(pd.DataFrame(data))

    util.validate_columns(df, ("col1", "col2"))
    util.validate_columns(df, None)


def test_prepare_orderby(spark):

    columns = ["a", "b"]

    # test empty input
    assert util.prepare_orderby(None) == []

    # test broadcast
    result = [F.col("a").asc(), F.col("b").asc()]
    assert str(result) == str(util.prepare_orderby(columns, True))

    # test exact
    result = [F.col("a").asc(), F.col("b").desc()]
    assert str(result) == str(util.prepare_orderby(columns, [True, False]))

    # test reverse
    result = [F.col("a").asc(), F.col("b").desc()]
    assert str(result) == str(util.prepare_orderby(columns, [False, True],
                                                   reverse=True))

    # raise unequal lengths
    with pytest.raises(ValueError):
        util.prepare_orderby(columns, [True, False, True])


def test_column_cacher(spark):

    data = {"col1": [1, 2], "col2": [3, 4]}
    df = spark.createDataFrame(pd.DataFrame(data))

    # check invalid mode argument
    with pytest.raises(ValueError):
        ColumnCacher(df, mode="incorrect argument")

    # check added column for mode = True
    cc = ColumnCacher(df, mode=True)
    cc.add("col3", F.lit(None))
    assert cc.columns["col3"] in cc.df.columns

    # check added column for mode = debug
    cc = ColumnCacher(df, mode="debug")
    cc.add("col3", F.lit(None))
    assert cc.columns["col3"] in cc.df.columns

    # check missing column for mode = False
    cc = ColumnCacher(df, mode=False)
    cc.add("col3", F.lit(None))
    assert "col3" not in cc.columns

    # check added column for force = True and mode = False
    cc = ColumnCacher(df, mode=False)
    cc.add("col3", F.lit(None), force=True)
    assert cc.columns["col3"] in cc.df.columns

    # check removed columns after finish with mode = True/False
    cc = ColumnCacher(df, mode=False)
    cc.add("col3", F.lit(None))
    df_result = cc.finish("col4", F.lit(None))
    assert "col3" not in cc.columns
    assert "col4" in df_result.columns

    # check remaining column after finish with mode debug
    cc = ColumnCacher(df, mode="debug")
    cc.add("col3", F.lit(None))
    df_result = cc.finish("col4", F.lit(None))
    assert cc.columns["col3"] in df_result.columns
    assert "col4" in df_result.columns

