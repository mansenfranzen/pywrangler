"""This module contains the pandas base wrangler.

"""

import pandas as pd

from pywrangler.base import BaseWrangler


class PandasWrangler(BaseWrangler):
    """Pandas wrangler base class.

    """

    @property
    def computation_engine(self):
        return "pandas"

    def _validate_output_shape(self, df_in: pd.DataFrame,
                               df_out: pd.DataFrame):
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
