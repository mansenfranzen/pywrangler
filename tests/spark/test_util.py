"""This module contains pyspark wrangler utility tests.

isort:skip_file
"""

import pytest
import pandas as pd

pytestmark = pytest.mark.pyspark  # noqa: E402
pyspark = pytest.importorskip("pyspark")  # noqa: E402

from pywrangler.spark import util


def test_spark_wrangler_validate_columns_raises(spark):

    data = {"col1": [1, 2], "col2": [3, 4]}
    df = spark.createDataFrame(pd.DataFrame(data))

    with pytest.raises(ValueError):
        util.validate_columns(df, ("col3", "col1"))


def test_spark_wrangler_validate_columns_not_raises(spark):

    data = {"col1": [1, 2], "col2": [3, 4]}
    df = spark.createDataFrame(pd.DataFrame(data))

    util.validate_columns(df, ("col1", "col2"))