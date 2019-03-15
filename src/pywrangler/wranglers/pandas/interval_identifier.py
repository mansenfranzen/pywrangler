"""This module contains implementations of the interval identifier wrangler.

"""

from typing import List

import pandas as pd

from pywrangler.wranglers.interfaces import IntervalIdentifier
from pywrangler.wranglers.pandas.base import PandasWrangler


class NaiveIterator(IntervalIdentifier, PandasWrangler):
    """Most simple, sequential implementation which iterates values while
    remembering the state of start and end markers.

    """

    def _naive_iterator(self, series: pd.Series) -> List[int]:
        """Iterates given `series` testing each value against start and end
        markers while keeping track of already instantiated intervals to
        separate valid from invalid intervals.

        Assumes that series is already ordered and grouped.

        """

        counter = 0  # counts the current interval id
        active = 0  # 0 in case no active interval, otherwise equals counter
        intermediate = []  # stores intermediate results
        result = []  # keeps track of all results

        def is_start(value):
            return value == self.marker_start

        def is_end(value):
            return value == self.marker_end

        def is_valid_start(value, active):
            """A valid start occurs if there is no active interval present (no
            start marker was seen since last end marker).

            """

            return is_start(value) and not active

        def is_invalid_start(value, active):
            """An invalid start occurs if there is already an active interval
            present (start marker was seen since last end marker).

            """

            return is_start(value) and active

        def is_valid_end(value, active):
            """A valid end is defined with `value` begin equal to the close
            marker and `active` being unqual to 0 which means there is an
            active interval.

            """

            return is_end(value) and active

        for value in series.values:

            if is_invalid_start(value, active):
                # add invalid values to result (from previous begin marker)
                result.extend([0] * len(intermediate))

                # start new intermediate list
                intermediate = [active]

            elif is_valid_start(value, active):
                active = counter + 1
                intermediate.append(active)

            elif is_valid_end(value, active):
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
            # finally, add rest to result which must be invalid
            result.extend([0] * len(intermediate))

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

        return df.sort_values(list(self.order_columns),
                              ascending=self.ascending)\
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


class VectorizedCumSum(IntervalIdentifier, PandasWrangler):
    """Sophisticated approach using multiple, vectorized operations. Using
    cumulative sum allows enumeration of intervals to avoid looping.

    """

    def _vectorized_cumsum(self, series: pd.Series) -> List[int]:
        """First, get enumeration of all intervals (valid and invalid). Every
        time a start or end marker is encountered, increase interval id by one.
        However, shift the end marker by one to include the end marker in the
        current interval. This is realized via the cumulative sum of boolean
        series of start markers and shifted end markers.

        Second, separate valid from invalid intervals by ensuring the presence
        of both start and end markers per interval id.

        Third, numerate valid intervals starting with 1 and set invalid
        intervals to 0.

        Assumes that series is already ordered and grouped.

        """

        # get boolean series with start and end markers
        bool_start = series.eq(self.marker_start)
        bool_end = series.eq(self.marker_end)

        # shifting the close marker allows cumulative sum to include the end
        bool_end_shift = bool_end.shift().fillna(False)

        # get increasing ids for intervals (in/valid) with cumsum
        ser_ids = (bool_start + bool_end_shift).cumsum()

        # separate valid vs invalid: ids with start AND end marker are valid
        bool_valid_ids = (bool_start + bool_end).groupby(ser_ids).sum().eq(2)

        valid_ids = bool_valid_ids.index[bool_valid_ids].values
        bool_valid = ser_ids.isin(valid_ids)

        # re-numerate ids from 1 to x and fill invalid with 0
        result = ser_ids[bool_valid].diff().ne(0).cumsum()
        return result.reindex(series.index).fillna(0).values

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

        return df.sort_values(list(self.order_columns),
                              ascending=self.ascending)\
                 .groupby(list(self.groupby_columns))[self.marker_column]\
                 .transform(self._vectorized_cumsum)\
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
