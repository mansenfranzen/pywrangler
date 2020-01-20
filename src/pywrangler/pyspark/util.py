"""This module contains utility functions (e.g. validation) commonly used by
pyspark wranglers.

"""

from typing import Iterable, Union, Optional

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
                    ascending: TYPE_ASCENDING,
                    reverse: bool = False) -> TYPE_OPT_COLUMN:
    """Convenient function to return orderby columns in correct
    ascending/descending order.

    """

    if order_columns is None:
        return []

    zipped = zip(order_columns, ascending)

    def boolify(sort_ascending: Optional[bool]) -> bool:
        return bool(sort_ascending) != reverse

    return [column if boolify(sort_ascending) else F.desc(column)
            for column, sort_ascending in zipped]


class ColumnCacher:
    """Composite of PySparkWrangler which enables storing of intermediate
    column expressions. PySpark column expressions can be stacked/chained. For
    example, a column expression may be a result of a conjunction of window
    functions and boolean masks for which the intermediate results are not
    stored because they are not needed for the final outcome.

    There are two valid reasons to store intermediate results. First,
    debugging requires to inspect intermediate results. Second, stacking
    column expressions seem to create more complex and less performant
    computation graphs.

    TODO: Add link to example for more complex graphs

    """

    def __init__(self, df: DataFrame, mode: Union[bool, str]):
        """Initialize column cacher. Set reference to dataframe.

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            DataFrame for which column caching will be activated.
        mode: bool, str
            If True, enables caching. If False, disables caching. If 'debug',
            enables caching and keeps intermediate columns (does not drop
            columns).

        """

        self.df = df
        self.mode = mode

        self.columns = {}

        valid_modes = {True, False, "debug"}
        if mode not in valid_modes:
            raise ValueError("Parameter `mode` has to be one of the "
                             "following: {}."
                             .format(valid_modes))

    def add(self, name: str, column: Column, force=False) -> Column:
        """Add given column to dataframe. Return referenced column. Creates
        unique name which is not yet present in dataframe.

        Parameters
        ----------
        name: str
            Name of the column.
        column: pyspark.sql.column.Column
            PySpark column expression to be explicitly added to dataframe.

        Returns
        -------
        reference: pyspark.sql.column.Column

        """

        if (self.mode is False) and (force is not True):
            return column

        if (self.mode is True) and (force == "debug"):
            return column

        col_name = "{}_{}".format(name, len(self.columns))
        while col_name in self.df.columns:
            col_name += "_"

        self.columns[name] = col_name
        self.df = self.df.withColumn(col_name, column)

        return F.col(col_name)

    def finish(self, name, column) -> DataFrame:
        """Closes column cacher and returns dataframe representation.

        """

        self.df = self.df.withColumn(name, column)

        if self.mode != "debug":
            self.df = self.df.drop(*self.columns.values())

        return self.df
