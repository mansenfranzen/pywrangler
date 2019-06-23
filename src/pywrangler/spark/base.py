"""This module contains the dask base wrangler.

"""

from typing import Iterable, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from pywrangler.base import BaseWrangler
from pywrangler.util.sanitizer import ensure_iterable
from pywrangler.util.types import TYPE_ASCENDING, TYPE_COLUMNS

TYPE_OPT_COLUMN = Union[None, Iterable[Column]]


class SparkWrangler(BaseWrangler):
    """Contains methods common to all spark based wranglers.

    """

    @property
    def computation_engine(self):
        return "spark"

    @staticmethod
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

    @staticmethod
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


class SparkSingleNoFit(SparkWrangler):
    """Mixin class defining `fit` and `fit_transform` for all wranglers with
    a single data frame input and output with no fitting necessary.

    """

    def fit(self, df: DataFrame):
        """Do nothing and return the wrangler unchanged.

        This method is just there to implement the usual API and hence work in
        pipelines.

        Parameters
        ----------
        df: pyspark.sql.DataFrame

        """

        return self

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """Apply fit and transform in sequence at once.

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        result: pyspark.sql.DataFrame

        """

        return self.fit(df).transform(df)
