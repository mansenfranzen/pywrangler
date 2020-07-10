"""This module contains utility functions (e.g. validation) commonly used by
pyspark wranglers.

"""

from typing import Union, Optional, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from pywrangler.util.sanitizer import ensure_iterable
from pywrangler.util.types import TYPE_ASCENDING, TYPE_COLUMNS
from pywrangler.pyspark.types import TYPE_PYSPARK_COLUMNS


def ensure_column(column: Union[Column, str]) -> Column:
    """Helper function to ensure that provided column will be of type
    `pyspark.sql.column.Column`.

    Parameters
    ----------
    column: str, Column
        Column object to be casted if required.

    Returns
    -------
    ensured: Column

    """

    if isinstance(column, Column):
        return column
    else:
        return F.col(column)


def validate_columns(df: DataFrame, columns: TYPE_COLUMNS):
    """Check that columns exist in dataframe and raise error if otherwise.

    Parameters
    ----------
    df: pyspark.sql.DataFrame
        Dataframe to check against.
    columns: Tuple[str]
        Columns to be validated.

    """

    columns = ensure_iterable(columns)
    compare_columns = {column.lower() for column in df.columns}

    for column in columns:
        if column.lower() not in compare_columns:
            raise ValueError('Column with name `{}` does not exist. '
                             'Please check parameter settings.'
                             .format(column))


def prepare_orderby(orderby_columns: TYPE_PYSPARK_COLUMNS,
                    ascending: TYPE_ASCENDING = True,
                    reverse: bool = False) -> List[Column]:
    """Convenient function to return orderby columns in correct
    ascending/descending order.

    Parameters
    ----------
    orderby_columns: TYPE_PYSPARK_COLUMNS
        Columns to explicitly apply an order to.
    ascending: TYPE_ASCENDING, optional
        Define order of columns via bools. True and False refer to ascending
        and descending, respectively.
    reverse: bool, optional
        Reverse the given order. By default, not activated.

    Returns
    -------
    ordered: list
        List of order columns.

    """

    # ensure columns
    orderby_columns = ensure_iterable(orderby_columns)
    orderby_columns = [ensure_column(column) for column in orderby_columns]

    # check if only True/False is given broadcast
    if isinstance(ascending, bool):
        ascending = [ascending] * len(orderby_columns)

    # ensure equal lengths, otherwise raise
    elif len(orderby_columns) != len(ascending):
        raise ValueError('`orderby_columns` and `ascending` must have '
                         'equal number of items.')

    zipped = zip(orderby_columns, ascending)

    def boolify(sort_ascending: Optional[bool]) -> bool:
        return bool(sort_ascending) != reverse

    return [column.asc() if boolify(sort_ascending) else column.desc()
            for column, sort_ascending in zipped]


class ColumnCacher:
    """Pyspark column expression cacher which enables storing of intermediate
    column expressions. PySpark column expressions can be stacked/chained. For
    example, a column expression may be a result of a conjunction of window
    functions and boolean masks for which the intermediate results are not
    stored because they are not needed for the final outcome.

    There are two valid reasons to store intermediate results. First,
    debugging requires to inspect intermediate results. Second, stacking
    column expressions seem to create more complex computation graphs. Storing
    intermediate results may help to decrease DAG complexity.

    For more, see Spark Jira: https://issues.apache.org/jira/browse/SPARK-30552

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
        force: bool, optional
            You may need to force to add a column temporarily for a given
            computation to finish even though you do not want to store
            intermediate results. This may be the case for window specs which
            rely on computed columns.

        Returns
        -------
        reference: pyspark.sql.column.Column

        """

        if (self.mode is False) and (force is not True):
            return column

        col_name = "{}_{}".format(name, len(self.columns))
        while col_name in self.df.columns:
            col_name += "_"

        self.columns[name] = col_name
        self.df = self.df.withColumn(col_name, column)

        return F.col(col_name)

    def finish(self, name, column) -> DataFrame:
        """Closes column cacher and returns dataframe representation with
        provided final result column. Intermediate columns will be dropped
        based on `mode`.

        Parameters
        ----------
        name: str
            Name of the final result column.
        column: pyspark.sql.column.Column
            Content of the final result column.

        Returns
        -------
        df: pyspark.sql.DataFrame
            Original dataframe with added column.

        """

        self.df = self.df.withColumn(name, column)

        if self.mode != "debug":
            self.df = self.df.drop(*self.columns.values())

        return self.df
