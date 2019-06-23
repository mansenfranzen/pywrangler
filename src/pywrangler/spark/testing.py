"""This module contains helper functions for testing.

"""

import pandas as pd
from pyspark.sql import DataFrame

from pywrangler.util.types import TYPE_COLUMNS

try:
    from pandas.testing import assert_frame_equal
except ImportError:
    from pandas.util.testing import assert_frame_equal


def assert_spark_pandas_equality(df_spark: DataFrame,
                                 df_pandas: pd.DataFrame,
                                 orderby: TYPE_COLUMNS = None):
    """Compare a spark and pandas dataframe in regard to content equality.
    Spark dataframes do not have a specific index or column order due to their
    distributed nature. In contrast, a test for equality for pandas dataframes
    respects index and column order. Therefore, the test for equality between a
    spark and pandas dataframe will ignore index and column order on purpose.

    Testing spark dataframes content is most simple while converting to pandas
    dataframes and having test data as pandas dataframes, too.

    To ensure index order is ignored, both dataframes need be sorted by all or
    given columns `orderby`.

    Parameters
    ----------
    df_spark: pyspark.sql.DataFrame
        Spark dataframe to be tested for equality.
    df_pandas: pd.DataFrame
        Pandas dataframe to be tested for equality.
    orderby: iterable, optional
        Columns to be sorted for correct index order.

    Returns
    -------
    None but asserts if dataframes are not equal.

    """

    orderby = orderby or df_pandas.columns.tolist()

    def prepare_compare(df):
        return df.sort_values(orderby).reset_index(drop=True)

    df_spark = prepare_compare(df_spark.toPandas())
    df_pandas = prepare_compare(df_pandas)

    assert_frame_equal(df_spark, df_pandas, check_like=True)
