"""This module contains utility functions (e.g. validation) commonly used by
spark wranglers.

"""

from typing import Iterable, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from pywrangler.util.sanitizer import ensure_iterable
from pywrangler.util.types import TYPE_ASCENDING, TYPE_COLUMNS

TYPE_OPT_COLUMN = Union[None, Iterable[Column]]


def validate_columns(df: DataFrame, columns: TYPE_COLUMNS):
    """Check that columns exist in dataframe and raise error if otherwise.

    Parameters
    ----------
    df: pyspark.sql.DataFrame
        Dataframe to check against.
    columns: Tuple[str]
        Columns to be validated.

    """

    if not columns:
        return

    columns = ensure_iterable(columns)

    for column in columns:
        if column not in df.columns:
            raise ValueError('Column with name `{}` does not exist. '
                             'Please check parameter settings.'
                             .format(column))


def prepare_orderby(order_columns: TYPE_COLUMNS,
                    ascending: TYPE_ASCENDING) -> TYPE_OPT_COLUMN:
    """Convenient function to return orderby columns in correct
    ascending/descending order.

    """

    if order_columns is None:
        return []

    zipped = zip(order_columns, ascending)
    return [column if ascending else F.desc(column)
            for column, ascending in zipped]
