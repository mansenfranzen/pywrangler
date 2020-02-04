"""This module contains implementations of the interval identifier wrangler.

"""

from typing import List

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

    def _validate_input(self, df: pd.DataFrame):
        """Checks input data frame in regard to column names and empty data.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to be validated.

        """

        util.validate_columns(df, self.marker_column)
        util.validate_columns(df, self.orderby_columns)
        util.validate_columns(df, self.groupby_columns)
        util.validate_empty_df(df)

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
        self._validate_input(df)

        # transform
        df_ordered = util.sort_values(df, self.orderby_columns, self.ascending)
        df_grouped = util.groupby(df_ordered, self.groupby_columns)

        df_result = df_grouped[self.marker_column] \
            .transform(self._transform) \
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

    def _transform(self, values: pd.Series) -> List[int]:
        """Selects appropriate algorithm depending on identical/different
        start and end markers.

        """

        start_first = self.marker_start_use_first
        end_first = self.marker_end_use_first

        if self._identical_start_end_markers:
            return self._agg_identical_start_end_markers(values)
        elif self.result_type == "raw":
            return self._agg_raw_iids(values)
        elif not start_first and end_first:
            return self._generic_start_first_end(values, False)
        elif start_first and not end_first:
            return self._generic_start_last_end(values, True)
        elif start_first and end_first:
            return self._generic_start_first_end(values, True)
        elif not start_first and not end_first:
            return self._generic_start_last_end(values, False)

    def _is_start(self, value):
        return value == self.marker_start

    def _is_end(self, value):
        return value == self.marker_end

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

    def _agg_raw_iids(self, series: pd.Series) -> List[int]:
        """Iterates given `series` testing each value against start marker
        while increasing counter each time start or end marker (shifted) is
        encountered.

        Assumes that series is already ordered and grouped.

        """

        result = []
        counter = 0
        lag = False

        for value in series.values:
            if lag:
                counter += 1
                lag = False

            if self._is_start(value):
                counter += 1
            elif self._is_end(value):
                lag = True

            result.append(counter)

        return result

    def _generic_start_first_end(self, series: pd.Series, first_start: bool) \
            -> List[int]:
        """Iterates given `series` testing each value against start and end
        markers while keeping track of already instantiated intervals to
        separate valid from invalid intervals.

        Assumes that series is already ordered and grouped.

        Parameters
        ----------
        series: pd.Series
            Sorted values which contain interval data.
        first_start: bool
            Indicates if first or last start is required. If True, generates
            ids for first start. If False, generates ids for last start.

        """

        counter = 0  # counts the current interval id
        active = 0  # 0 in case no active interval, otherwise equals counter
        intermediate = []  # stores intermediate results
        result = []  # keeps track of all results

        for value in series.values:

            if self._is_start(value):
                if active and not first_start:
                    # add invalid values to result (from previous begin marker)
                    result.extend([0] * len(intermediate))

                    # start new intermediate list
                    intermediate = []

                if not active:
                    active = counter + 1

                intermediate.append(active)

            elif self._is_end(value) and active:
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

    def _generic_start_last_end(self, series: pd.Series, first_start: bool) \
            -> List[int]:
        """Iterates given `series` testing each value against start and end
        markers while keeping track of already instantiated intervals to
        separate valid from invalid intervals.

        Requires state for opened start/end markers and number of noise values
        since last end marker.

        Assumes that series is already ordered and grouped.

        Parameters
        ----------
        series: pd.Series
            Sorted values which contain interval data.
        first_start: bool
            Indicates if first or last start is required. If True, generates
            ids for first start. If False, generates ids for last start.

        """

        counter = 0  # counts the current interval id
        active_start = False  # remember opened start marker
        active_end = False  # remember opened end marker
        noise_counter = 0  # store number of noises after end marker
        intermediate = []  # store intermediate results
        result = []  # keeps track of all results

        for value in series.values:
            # handle start marker
            if self._is_start(value):
                # closing valid interval
                if active_start & active_end:
                    result.extend(intermediate)
                    result.extend([0] * noise_counter)
                    counter += 1

                    noise_counter = 0
                    active_end = False
                    intermediate = []

                # increase counter only if start was not active previously
                elif not active_start:
                    counter += 1

                # handle last start
                elif not active_end and not first_start:
                    result.extend([0] * len(intermediate))
                    intermediate = []

                active_start = True
                intermediate.append(counter)

            # handle end marker
            elif self._is_end(value):
                if not active_start:
                    result.append(0)
                else:
                    active_end = True
                    count = len(intermediate) + noise_counter + 1
                    result.extend([counter] * count)

                    intermediate = []
                    noise_counter = 0

            # handle noise
            else:
                if active_end:
                    noise_counter += 1
                elif active_start:
                    intermediate.append(counter)
                else:
                    result.append(0)

        # handle remaining values
        if active_start & ~active_end:
            result.extend([0] * len(intermediate))
        elif active_end:
            intermediate.extend([0] * noise_counter)
            result.extend(intermediate)

        return result


class VectorizedCumSum(_BaseIntervalIdentifier):
    """Sophisticated approach using multiple, vectorized operations. Using
    cumulative sum allows enumeration of intervals to avoid looping.

    """

    def _transform(self, values: pd.Series) -> List[int]:
        """Selects appropriate algorithm depending on identical/different
        start and end markers.

        """

        if self._identical_start_end_markers:
            return self._agg_identical_start_end_markers(values)

        if self.marker_start_use_first and not self.result_type == "raw":
            values = self._drop_duplicated_marker(values, True)

        if not self.marker_end_use_first and not self.result_type == "raw":
            values = self._drop_duplicated_marker(values, False)

        return self._last_start_first_end(values)

    def _drop_duplicated_marker(self, marker_column: pd.Series,
                                start: bool = True):
        """Modify marker column to keep only first start marker or last end
        marker.

        Parameters
        ----------
        marker_column: pd.Series
            Values for which duplicated markers will be removed.
        start: bool, optional
            Indicate which duplicates should be dropped. If True, only first
            start marker is kept. If False, only last end marker is kept.

        Returns
        -------
        dropped: pd.Series

        """

        valid_values = [self.marker_start, self.marker_end]
        denoised = marker_column.where(marker_column.isin(valid_values))

        if start:
            fill = denoised.ffill()
            marker = 1
            shift = 1
        else:
            fill = denoised.bfill()
            marker = 2
            shift = -1

        shifted = fill.shift(shift)
        shifted_start_only = shifted.where(fill.eq(marker))

        mask_drop = (shifted_start_only == marker_column)
        dropped = marker_column.where(~mask_drop)

        return dropped

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
        iids_raw = bool_start.add(bool_end_shift).cumsum()
        if self.result_type == "raw":
            return iids_raw

        # separate valid vs invalid: ids with start AND end marker are valid
        mask_valid_ids = bool_start.add(bool_end).groupby(iids_raw).sum().eq(2)
        valid_ids = mask_valid_ids.index[mask_valid_ids].values
        mask = iids_raw.isin(valid_ids)

        if self.result_type == "valid":
            return iids_raw.where(mask, 0)

        # re-numerate ids from 1 to x and fill invalid with 0
        result = iids_raw[mask].diff().ne(0).cumsum()
        return result.reindex(series.index).fillna(0).values

    def _agg_identical_start_end_markers(self, series: pd.Series) -> List[int]:
        """Iterates given `series` testing each value against start marker
        while increasing counter each time start marker is encountered.

        Assumes that series is already ordered and grouped.

        """

        bool_start = series.eq(self.marker_start)
        return bool_start.cumsum()
