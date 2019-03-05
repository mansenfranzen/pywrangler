"""This module contains the pandas base wrangler.

"""

from typing import Tuple

import pandas as pd

from pywrangler.wranglers.base import BaseWrangler


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
    def validate_columns(df: pd.DataFrame, columns: Tuple[str]):
        """Check that columns exist in dataframe and raise error if otherwise.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to check against.
        columns: Tuple[str]
            Columns to be validated.

        """

        for column in columns:
            if column not in df.columns:
                raise ValueError('Column with name `{}` does not exist. '
                                 'Please check parameter settings.')
