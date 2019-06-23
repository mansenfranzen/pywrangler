"""This module contains the pandas base wrangler.

"""

import numpy as np
import pandas as pd

from pywrangler.base import BaseWrangler
from pywrangler.util.sanitizer import ensure_tuple
from pywrangler.util.types import TYPE_ASCENDING, TYPE_COLUMNS


class PandasWrangler(BaseWrangler):
    """Contains methods common to all pandas based wranglers.

    """

    @property
    def computation_engine(self):
        return "pandas"

    def validate_output_shape(self, df_in: pd.DataFrame, df_out: pd.DataFrame):
        """If wrangler implementation preserves sample size, assert equal
        sample sizes between input and output dataframe.

        Using pandas, all data is in memory. Hence, getting shape information
        is cheap and this check can be done regularly (in contrast to pyspark
        where `df.count()` can be very expensive).

        Parameters
        ----------
        df_in: pd.DataFrame
            Input dataframe.
        df_out: pd.DataFrame
            Output dataframe.

        """

        if self.preserves_sample_size:
            shape_in = df_in.shape[0]
            shape_out = df_out.shape[0]

            if shape_in != shape_out:
                raise ValueError('Number of input samples ({}) does not match '
                                 'number of ouput samples ({}) which should '
                                 'be the case because wrangler is supposed to '
                                 'preserve the number of samples.'
                                 .format(shape_in, shape_out))

    @staticmethod
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

    @staticmethod
    def validate_columns(df: pd.DataFrame, columns: TYPE_COLUMNS):
        """Check that columns exist in dataframe and raise error if otherwise.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to check against.
        columns: Tuple[str]
            Columns to be validated.

        """

        if not columns:
            return

        columns = ensure_tuple(columns)

        for column in columns:
            if column not in df.columns:
                raise ValueError('Column with name `{}` does not exist. '
                                 'Please check parameter settings.'
                                 .format(column))

    @staticmethod
    def sort_values(df: pd.DataFrame,
                    order_columns: TYPE_COLUMNS,
                    ascending: TYPE_ASCENDING) -> pd.DataFrame:
        """Convenient function to return sorted dataframe while taking care of
         optional order columns and order (ascending/descending).

         """

        if order_columns:
            return df.sort_values(list(order_columns),
                                  ascending=list(ascending))
        else:
            return df

    @staticmethod
    def groupby(df: pd.DataFrame, groupby_columns: TYPE_COLUMNS):
        """Convenient function to group by a dataframe while taking care of
         optional groupby columns. Always returns a `DataFrameGroupBy` object.

         """

        if groupby_columns:
            return df.groupby(list(groupby_columns))
        else:
            return df.groupby(np.zeros(df.shape[0]))


class PandasSingleNoFit(PandasWrangler):
    """Mixin class defining `fit` and `fit_transform` for all wranglers with
    a single data frame input and output with no fitting necessary.

    """

    def fit(self, df: pd.DataFrame):
        """Do nothing and return the wrangler unchanged.

        This method is just there to implement the usual API and hence work in
        pipelines.

        Parameters
        ----------
        df: pd.DataFrame

        """

        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fit and transform in sequence at once.

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        result: pd.DataFrame

        """

        return self.fit(df).transform(df)
