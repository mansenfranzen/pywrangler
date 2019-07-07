"""This module contains tests for pyspark testing utility.

isort:skip_file
"""

import pytest
import pandas as pd

pytestmark = pytest.mark.pyspark  # noqa: E402
pyspark = pytest.importorskip("pyspark")  # noqa: E402

from pywrangler.pyspark.testing import assert_pyspark_pandas_equality


def test_assert_spark_pandas_equality_no_assert(spark):
    data = [[1, 2, 3, 4],
            [5, 6, 7, 8]]
    columns = list("abcd")
    index = [0, 1]
    test_data = pd.DataFrame(data=data, columns=columns, index=index)

    test_input = spark.createDataFrame(test_data)
    test_output = test_data.reindex([1, 0])
    test_output = test_output[["b", "c", "a", "d"]]

    assert_pyspark_pandas_equality(test_input, test_output)
    assert_pyspark_pandas_equality(test_input, test_output, orderby=["a"])


def test_assert_spark_pandas_equality_assert(spark):
    data = [[1, 2, 3, 4],
            [5, 6, 7, 8]]
    columns = list("abcd")
    index = [0, 1]
    test_data = pd.DataFrame(data=data, columns=columns, index=index)

    test_input = spark.createDataFrame(test_data)
    test_output = test_data.copy(deep=True)
    test_output.iloc[0, 0] = 100

    with pytest.raises(AssertionError):
        assert_pyspark_pandas_equality(test_input, test_output)

    with pytest.raises(AssertionError):
        assert_pyspark_pandas_equality(test_input, test_output, orderby=["a"])
