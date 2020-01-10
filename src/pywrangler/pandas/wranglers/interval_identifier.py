"""This module contains implementations of the interval identifier wrangler.

"""

from typing import List, Callable

import pandas as pd
from pywrangler.pandas import util
from pywrangler.pandas.base import PandasSingleNoFit
from pywrangler.wranglers import IntervalIdentifier


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

    def validate_input(self, df: pd.DataFrame):
        """Checks input data frame in regard to column names and empty data.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to be validated.

        """

        util.validate_columns(df, self.marker_column)
        util.validate_columns(df, self.order_columns)
        util.validate_columns(df, self.groupby_columns)
        util.validate_empty_df(df)

    def _transform(self, df: pd.DataFrame, transformer: Callable) -> pd.DataFrame:
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
        df_ordered = util.sort_values(df, self.order_columns, self.ascending)
        df_grouped = util.groupby(df_ordered, self.groupby_columns)

        df_result = df_grouped[self.marker_column] \
            .transform(transformer) \
            .astype(int) \
            .reindex(df.index) \
            .to_frame(self.target_column_name)

        # check output
        self._validate_output_shape(df, df_result)

        return df_result


class NaiveIterator(_BaseIntervalIdentifier):
    """Most simple, sequential implementation which iterates values while
    remembering the state of start and end markers.

    """

    def transform(self, df: pd.DataFrame) -> List[int]:
        """Selects appropriate algorithm depending on identical/different
        start and end markers.

        """

        start_first = self.marker_start_use_first
        end_first = self.marker_end_use_first

        if self._identical_start_end_markers:
            transformer = self._agg_identical_start_end_markers

        elif ~start_first & end_first:
            transformer = self._last_start_first_end

        elif start_first & ~end_first:
            raise NotImplementedError

        elif start_first & end_first:
            raise NotImplementedError

        else:
            raise NotImplementedError

        return self._transform(df, transformer)

    def _is_start(self, value):
        return value == self.marker_start

    def _is_end(self, value):
        return value == self.marker_end

    def _is_valid_start(self, value, active):
        """A valid start occurs if there is no active interval present (no
        start marker was seen since last end marker).

        """

        return self._is_start(value) and not active

    def _is_invalid_start(self, value, active):
        """An invalid start occurs if there is already an active interval
        present (start marker was seen since last end marker).

        """

        return self._is_start(value) and active

    def _is_valid_end(self, value, active):
        """A valid end is defined with `value` begin equal to the close
        marker and `active` being unqual to 0 which means there is an
        active interval.

        """

        return self._is_end(value) and active

    def _agg_identical_start_end_markers(self, series: pd.Series) -> List[int]:
        """Iterates given `series` testing each value against start marker
        while increasing counter each time start marker is encountered.

        Assumes that series is already ordered and grouped.

        """

        result = []
        counter = 0

        for value in series.values:
            if self._is_start(value):
                counter += 1

            result.append(counter)

        return result

    def _last_start_first_end(self, series: pd.Series) -> List[int]:
        """Iterates given `series` testing each value against start and end
        markers while keeping track of already instantiated intervals to
        separate valid from invalid intervals.

        Assumes that series is already ordered and grouped.

        """

        counter = 0  # counts the current interval id
        active = 0  # 0 in case no active interval, otherwise equals counter
        intermediate = []  # stores intermediate results
        result = []  # keeps track of all results

        for value in series.values:

            if self._is_invalid_start(value, active):
                # add invalid values to result (from previous begin marker)
                result.extend([0] * len(intermediate))

                # start new intermediate list
                intermediate = [active]

            elif self._is_valid_start(value, active):
                active = counter + 1
                intermediate.append(active)

            elif self._is_valid_end(value, active):
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

        # finally, add rest to result which must be invalid
        result.extend([0] * len(intermediate))

        return result


class VectorizedCumSum(_BaseIntervalIdentifier):
    """Sophisticated approach using multiple, vectorized operations. Using
    cumulative sum allows enumeration of intervals to avoid looping.

    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        start_first = self.marker_start_use_first
        end_first = self.marker_end_use_first

        if self._identical_start_end_markers:
            transformer = self._agg_identical_start_end_markers

        elif ~start_first & end_first:
            transformer = self._last_start_first_end

        elif start_first & ~end_first:
            raise NotImplementedError

        elif start_first & end_first:
            raise NotImplementedError

        else:
            raise NotImplementedError

        return self._transform(df, transformer)

    def _last_start_first_end(self, series: pd.Series) -> List[int]:
        """Extract shortest intervals from given dataFrame as ids.
        First, get enumeration of all intervals (valid and invalid). Every
        time a start or end marker is encountered, increase interval id by one.
        The end marker is shifted by one to include the end marker in the
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

    def _agg_identical_start_end_markers(self, series: pd.Series) -> List[int]:

        bool_start = series.eq(self.marker_start)
        return bool_start.cumsum()
