"""This module contains implementations of the interval identifier wrangler.

"""

from typing import List

import pandas as pd

from pywrangler.wranglers.interfaces import IntervalIdentifier
from pywrangler.wranglers.pandas.base import PandasSingleNoFit


class _BaseIntervalIdentifier(PandasSingleNoFit, IntervalIdentifier):
    """Provides `transform` and `validate_input` methods  common to more than
    one implementation of the pandas interval identification wrangler.

    The `transform` has several shared responsibilities. It sorts and groups
    the data before applying the `_transform` method which needs to be
    implemented by every wrangler subclassing this mixin. In addition, it
    remains the original index of the input dataframe, ensures the resulting
    column to be of type integer and converts output to a data frame with
    parametrized target column name.

    """

    def validate_input(self,  df: pd.DataFrame):
        """Checks input data frame in regard to column names and empty data.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to be validated.

        """

        self.validate_columns(df, self.marker_column)
        self.validate_columns(df, self.order_columns)
        self.validate_columns(df, self.groupby_columns)
        self.validate_empty_df(df)

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

        # check input
        self.validate_input(df)

        # transform
        df_ordered = self.sort_values(df, self.order_columns, self.ascending)
        df_grouped = self.groupby(df_ordered, self.groupby_columns)

        df_result = df_grouped[self.marker_column]\
            .transform(self._transform)\
            .astype(int)\
            .reindex(df.index)\
            .to_frame(self.target_column_name)

        # check output
        self.validate_output_shape(df, df_result)

        return df_result

    def _transform(self, series: pd.Series) -> List[int]:
        """Needs to be implemented.

        """

        raise NotImplementedError


class NaiveIterator(_BaseIntervalIdentifier):
    """Most simple, sequential implementation which iterates values while
    remembering the state of start and end markers.

    """

    def _transform(self, series: pd.Series) -> List[int]:
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


class VectorizedCumSum(_BaseIntervalIdentifier):
    """Sophisticated approach using multiple, vectorized operations. Using
    cumulative sum allows enumeration of intervals to avoid looping.

    """

    def _transform(self, series: pd.Series) -> List[int]:
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
        ser_ids = bool_start.add(bool_end_shift).cumsum()

        # separate valid vs invalid: ids with start AND end marker are valid
        bool_valid_ids = bool_start.add(bool_end).groupby(ser_ids).sum().eq(2)

        valid_ids = bool_valid_ids.index[bool_valid_ids].values
        bool_valid = ser_ids.isin(valid_ids)

        # re-numerate ids from 1 to x and fill invalid with 0
        result = ser_ids[bool_valid].diff().ne(0).cumsum()
        return result.reindex(series.index).fillna(0).values
