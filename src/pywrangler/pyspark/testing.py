"""This module contains helper functions for testing.

"""

import pandas as pd
from pyspark.sql import DataFrame

from pywrangler.util.types import TYPE_COLUMNS

try:
    from pandas.testing import assert_frame_equal
except ImportError:
    from pandas.util.testing import assert_frame_equal

# constant for pandas missing NULL values
PANDAS_NULL = object()


def prepare_spark_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas does not distinguish NULL and NaN values. Everything null-like
    is converted to NaN. However, spark does distinguish NULL and NaN for
    example. To enable correct spark dataframe creation with NULL and NaN
    values, the `PANDAS_NULL` constant is used as a workaround to enforce NULL
    values in pyspark dataframes. Pyspark treats `None` values as NULL.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe to be prepared.

    Returns
    -------
    df_prepared: pd.DataFrame
        Prepared dataframe for spark conversion.

    """

    return df.where(df.ne(PANDAS_NULL), None)


def assert_pyspark_pandas_equality(df_spark: DataFrame,
                                   df_pandas: pd.DataFrame,
                                   orderby: TYPE_COLUMNS = None):
    """Compare a pyspark and pandas dataframe in regard to content equality.
    Pyspark dataframes don't have a specific index or column order due to their
    distributed nature. In contrast, a test for equality for pandas dataframes
    respects index and column order. Therefore, the test for equality between a
    pyspark and pandas dataframe will ignore index and column order on purpose.

    Testing pyspark dataframes content is most simple while converting to
    pandas dataframes and having test data as pandas dataframes, too.

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

    df_spark = df_spark.toPandas()

    # check for non matching columns and enforce identical column order
    mismatch_columns = df_pandas.columns.symmetric_difference(df_spark.columns)
    if not mismatch_columns.empty:
        raise AssertionError("Column names do not match: {}"
                             .format(mismatch_columns.tolist()))
    else:
        df_spark = df_spark[df_pandas.columns]

    # enforce identical row order
    orderby = orderby or df_pandas.columns.tolist()

    def prepare_compare(df):
        df = df.sort_values(orderby).reset_index(drop=True)
        df = df.where((pd.notnull(df)), None)
        return df

    df_pandas = prepare_compare(df_pandas)
    df_spark = prepare_compare(df_spark)

    assert_frame_equal(df_spark,
                       df_pandas,
                       check_like=False,
                       check_dtype=False)
