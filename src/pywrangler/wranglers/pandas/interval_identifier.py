"""This module contains implementations of the interval identifier wrangler.

"""

from typing import List

import pandas as pd

from pywrangler.wranglers.interfaces import IntervalIdentifier
from pywrangler.wranglers.pandas.base import PandasWrangler


class NaiveIterator(IntervalIdentifier, PandasWrangler):
    """Most simple, sequential implementation which iterates over values while
    remembering the state of start and end markers.

    The `_native_iterator` method extracts intervals using plain python
    assuming values to be already ordered and grouped correctly. Ordering and
    grouping while retaining the original index is left to pandas within the
    `transform` method.

    """

    def _naive_iterator(self, series: pd.Series) -> List[int]:
        """Iterates given `series` value by value and extracts interval id.
        Assumes that series is already ordered and grouped.

        """

        counter = 0  # counts the current interval id
        active = 0  # 0 in case no active interval, otherwise equals counter
        intermediate = []  # stores intermediate results
        result = []  # keeps track of all results

        def is_begin(value):
            return value == self.marker_start

        def is_close(value):
            return value == self.marker_end

        def is_valid_begin(value, active):
            """A valid begin occurs if there is no active interval present (no
            start marker was seen since last end marker).

            """

            return is_begin(value) and not active

        def is_invalid_begin(value, active):
            """An invalid begin occurs if there is already an active interval
            present (start marker was seen since last end marker).

            """

            return is_begin(value) and active

        def is_valid_close(value, active):
            """A valid close is defined with `value` begin equal to the close
            marker and `active` being unqual to 0 which means there is an
            active interval.

            """

            return is_close(value) and active

        for value in series.values:

            if is_invalid_begin(value, active):
                # add invalid values to result (from previous begin marker)
                result.extend([0] * len(intermediate))

                # start new intermediate list
                intermediate = [active]

            elif is_valid_begin(value, active):
                active = counter + 1
                intermediate.append(active)

            elif is_valid_close(value, active):
                # add valid interval to result
                result.extend(intermediate)
                result.append(active)

                # empty intermediate list
                intermediate = []
                active = 0

                # increase id counter since valid interval was closed
                counter += 1

            else:
                intermediate.append(active)

        else:
            # finally, add rest to result
            result.extend(intermediate)

        return result

    def fit(self, df: pd.DataFrame):
        """Do nothing and return the wrangler unchanged.

        This method is just there to implement the usual API and hence work in
        pipelines.

        Parameters
        ----------
        df: pd.DataFrame

        """

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract interval ids from given dataframe.

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        result: pd.DataFrame
            Single columned dataframe with same index as `df`.

        """

        return df.sort_values(list(self.order_columns))\
                 .groupby(list(self.groupby_columns))[self.marker_column]\
                 .transform(self._naive_iterator)\
                 .reindex(df.index)\
                 .to_frame(self.target_column_name)\
                 .astype(int)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fit and transform in sequence at once.

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        result: pd.DataFrame
            Single columned dataframe with same index as `df`.

        """
        return self.fit(df).transform(df)
