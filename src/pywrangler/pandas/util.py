"""This module contains utility functions (e.g. validation) commonly used by
pandas wranglers.

"""

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from pywrangler.util.sanitizer import ensure_iterable
from pywrangler.util.types import TYPE_ASCENDING, TYPE_COLUMNS


def validate_empty_df(df: pd.DataFrame):
    """Check for empty dataframe. By definition, wranglers operate on non
    empty dataframe. Therefore, raise error if dataframe is empty.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to check against.

    """

    if df.empty:
        raise ValueError('Dataframe is empty.')


def validate_columns(df: pd.DataFrame, columns: TYPE_COLUMNS):
    """Check that columns exist in dataframe and raise error if otherwise.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to check against.
    columns: iterable[str]
        Columns to be validated.

    """

    columns = ensure_iterable(columns)

    for column in columns:
        if column not in df.columns:
            raise ValueError('Column with name `{}` does not exist. '
                             'Please check parameter settings.'
                             .format(column))


def sort_values(df: pd.DataFrame,
                order_columns: TYPE_COLUMNS,
                ascending: TYPE_ASCENDING) -> pd.DataFrame:
    """Convenient function to return sorted dataframe while taking care of
     optional order columns and order (ascending/descending).

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to check against.
    order_columns: TYPE_COLUMNS
        Columns to be sorted.
    ascending: TYPE_ASCENDING
        Column order.

    Returns
    -------
    df_sorted: pd.DataFrame

    """

    if order_columns:
        return df.sort_values(order_columns, ascending=ascending)
    else:
        return df


def groupby(df: pd.DataFrame,
            groupby_columns: TYPE_COLUMNS) -> DataFrameGroupBy:
    """Convenient function to group by a dataframe while taking care of
     optional groupby columns. Always returns a `DataFrameGroupBy` object.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to check against.
    groupby_columns: TYPE_COLUMNS
        Columns to be grouped by.

    Returns
    -------
    groupby: DataFrameGroupBy

    """

    if groupby_columns:
        return df.groupby(groupby_columns)
    else:
        return df.groupby(np.zeros(df.shape[0]))
