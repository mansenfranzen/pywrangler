"""This module contains implementations of the interval identifier wrangler.

"""
import operator
from typing import NamedTuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import Column

from pywrangler.pyspark import util
from pywrangler.pyspark.base import PySparkSingleNoFit
from pywrangler.wranglers import IntervalIdentifier


class VectorizedCumSum(PySparkSingleNoFit, IntervalIdentifier):
    """Sophisticated approach avoiding python UDFs. However multiple windows
    are necessary.

    First, get enumeration of all intervals (valid and invalid). Every
    time a start or end marker is encountered, increase interval id by one.
    The end marker is shifted by one to include the end marker in the
    current interval. This is realized via the cumulative sum of boolean
    series of start markers and shifted end markers.

    Second, separate valid from invalid intervals by ensuring the presence
    of both start and end markers per interval id.

    Third, numerate valid intervals starting with 1 and set invalid
    intervals to 0.

    """

    def validate_input(self, df: DataFrame):
        """Checks input data frame in regard to column names.

        Parameters
        ----------
        df: pyspark.sql.Dataframe
            Dataframe to be validated.

        """

        util.validate_columns(df, self.marker_column)
        util.validate_columns(df, self.order_columns)
        util.validate_columns(df, self.groupby_columns)

        if self.order_columns is None:
            raise ValueError("Please define an order column. Pyspark "
                             "dataframes have no implicit order unlike pandas "
                             "dataframes.")

    def _boolify_marker(self, marker_column, start=True):
        """Helper function to create an integer casted boolean column
        expression of start/end marker.

        """

        if start:
            marker = self.marker_start
        else:
            marker = self.marker_end

        return marker_column.eqNullSafe(marker).cast("integer")

    def _denoise_marker_column(self, window, start=True):
        """Return marker column with noises removed and forward/backwards
         filled.

        Parameters
        ----------
        window: pyspark.sql.Window
            Resembles a window specification according to groupby/order.
        start: bool, optional
            Indicate fill order. If True, forward fill for start markers. If
            False, backwards fill for end markers.

        Returns
        -------
        denoised: pyspark.sql.column.Column
            Return spark column expression with denoised values.

        """

        marker_column = F.col(self.marker_column)

        # remove noise values
        valid_values = [self.marker_start, self.marker_end]
        mask_no_noise = marker_column.isin(valid_values)
        denoised = F.when(mask_no_noise, marker_column)

        # forward fill with remaining start/end markers
        if start:
            ffill_window = window.rowsBetween(Window.unboundedPreceding, 0)
            fill = F.last(denoised, ignorenulls=True).over(ffill_window)
        else:
            bfill_window = window.rowsBetween(0, Window.unboundedFollowing)
            fill = F.first(denoised, ignorenulls=True).over(bfill_window)

        return fill

    def _drop_duplicated_marker(self, marker_column, window, start=True):
        """Modify marker column to keep only first start marker or last end
        marker.

        Parameters
        ----------
        marker_column: pyspark.sql.column.Column
            Column expression for which duplicated markers will be removed.
        window: pyspark.sql.Window
            Resembles a window specification according to groupby/order.
        start: bool, optional
            Indicate which duplicates should be dropped. If True, only first
            start marker is kept. If False, only last end marker is kept.

        Returns
        -------
        dropped: pyspark.sql.column.Column

        """

        if start:
            marker = self.marker_start
            count_lag = 1
        else:
            marker = self.marker_end
            count_lag = -1

        denoised = self._denoise_marker_column(window, start)

        # apply modification only to marker values
        mask_only = denoised == marker

        # use shifted column to identify subsequent duplicates
        shifted = F.lag(denoised, count=count_lag).over(window)
        shifted_start_only = F.when(mask_only, shifted)

        # nullify duplicates
        mask_drop = (shifted_start_only == marker_column)
        dropped = F.when(mask_drop, F.lit(None)).otherwise(marker_column)

        return dropped

    def _generate_windows(self):
        """Generate pyspark sql windows which are required by all subsequent
        computational steps. Two windows are relevant. First, `w_lag`
        represents the main window corresponding to the given groupby and
        orderby columns. Second, `w_id` is identical to `w_lag` except for an
        addition column which resembles raw iids.

        Returns
        -------
        w_lag, w_id: pyspark.sql.Window

        """

        orderby = util.prepare_orderby(self.order_columns, self.ascending)
        groupby = self.groupby_columns or []

        w_lag = Window.partitionBy(groupby).orderBy(orderby)
        w_id = Window.partitionBy(groupby + [self.target_column_name])

        return w_lag, w_id

    def _preprocess_marker_column(self, window):
        """If required, removes duplicated start/end markers and returns
        modified marker column which is ready to be further processed.

        Parameters
        ----------
        window: pyspark.sql.Window
            Resembles a window specification according to groupby/order.

        Returns
        -------
        col: pyspark.sql.column.Column
            Modified marker column.

        """

        col = F.col(self.marker_column)
        if self._identical_start_end_markers:
            return col

        if self.marker_start_use_first:
            col = self._drop_duplicated_marker(col, window)

        if not self.marker_end_use_first:
            col = self._drop_duplicated_marker(col, window, False)

        return col

    def _generate_raw_iids(self, marker_column, window):
        """Create sequence of interval ids in increasing order regardless of
        their validity.

        Parameters
        ----------
        marker_column: pyspark.sql.column.Column
            Column expression resembling the marker column.
        window: pyspark.sql.Window
            Resembles a window specification according to groupby/order.

        Returns
        -------
        raw_iids: pyspark.sql.column.Column

        """

        bool_start = self._boolify_marker(marker_column, True)
        bool_end = self._boolify_marker(marker_column, False)

        # shifting the end marker allows cumulative sum to include the end
        bool_end_shift = F.lag(bool_end, default=1).over(window)
        bool_start_end_shift = bool_start + bool_end_shift

        # get increasing ids for intervals (in/valid) with cumsum
        raw_iids = F.sum(bool_start_end_shift).over(window)

        return raw_iids

    def _generate_valid_iids(self, marker_column, raw_iids, window):
        """Create sequence of interval identifier ids in increasing order
        with invalid intervals being removed. Invalid iids will be set to 0.

        Parameters
        ----------
        marker_column: pyspark.sql.column.Column
            Column expression resembling the marker column.
        raw_iids: pyspark.sql.column.Column
            Column expression with raw iids.
        window: pyspark.sql.Window
            Resembles a window specification according to groupby/order + the
            column name of the raw iids.

        Returns
        -------
        valid_iids: pyspark.sql.column.Column

        """

        bool_start = self._boolify_marker(marker_column, True)
        bool_end = self._boolify_marker(marker_column, False)
        bool_start_end = bool_start + bool_end

        bool_valid = F.sum(bool_start_end).over(window) == 2
        valid_iids = F.when(bool_valid, raw_iids).otherwise(0)

        return valid_iids

    def _generate_renumerated_iids(self, valid_iids, window):
        """Create sequence of interval identifier ids in increasing order
        starting with 1 in steps of 1. Invalid intervals are marked with 0.

        Parameters
        ----------
        valid_iids: pyspark.sql.column.Column
            Column expression resembling valid iids.
        window: pyspark.sql.Window
            Resembles a window specification according to groupby/order.

        Returns
        -------
        raw_iids: pyspark.sql.column.Column

        """

        valid_ids_shift = F.lag(valid_iids, default=0).over(window)
        valid_ids_diff = valid_ids_shift - valid_iids
        valid_ids_increase = (valid_ids_diff < 0).cast("integer")

        renumerate = F.sum(valid_ids_increase).over(window)

        bool_valid = valid_iids != 0
        renumerated_iids = F.when(bool_valid, renumerate).otherwise(0)

        return renumerated_iids

    def _generate_iids_identical(self, marker_col, window):
        """Compute interval ids for identical start and end markers.

        Parameters
        ----------
        marker_column: pyspark.sql.column.Column
            Column expression resembling the marker column.
        window: pyspark.sql.Window
            Resembles a window specification according to groupby/order.

        Returns
        -------
        iids: pyspark.sql.column.Column

        """

        bool_start = self._boolify_marker(marker_col)
        iids = F.sum(bool_start).over(window)

        return iids

    def transform(self, df: DataFrame) -> DataFrame:
        """Extract interval ids from given dataframe.

        Parameters
        ----------
        df: pyspark.sql.Dataframe

        Returns
        -------
        result: pyspark.sql.Dataframe
            Same columns as original dataframe plus the new interval id column.

        """

        # check input
        self.validate_input(df)

        # define window specs
        w_lag, w_id = self._generate_windows()

        # get preprocessed marker col
        marker_col = self._preprocess_marker_column(w_lag)

        # early exit for identical start/end markers
        if self._identical_start_end_markers:
            iids = self._generate_iids_identical(marker_col, w_lag)
            return df.withColumn(self.target_column_name, iids)

        #df = df.withColumn("processed", marker_col)
        #marker_col = F.col("processed")

        # get iids
        iids_raw = self._generate_raw_iids(marker_col, w_lag)

        #df = df.withColumn(self.target_column_name, iids_raw)
        #df = df.withColumn("iids_raw", iids_raw)
        #iids_raw = F.col("iids_raw")

        iids_valid = self._generate_valid_iids(marker_col, iids_raw, w_id)
        #df = df.withColumn("iids_valid", iids_valid)
        #iids_valid = F.col("iids_valid")

        iids_renumerated = self._generate_renumerated_iids(iids_valid, w_lag)
        #df = df.withColumn("iids_renumerated", iids_renumerated)
        #iids_renumerated = F.col("iids_renumerated")

        # apply expressions
        df = df.withColumn(self.target_column_name, iids_raw)
        df = df.withColumn(self.target_column_name, iids_renumerated)

        return df


class VectorizedCumSumAdjusted(VectorizedCumSum):
    """Modifies 2 versions (last start/last end and first start/first end)
    which may be faster.

    """

    def _marker_bool_start_end(self):
        """get boolean series with start and end markers
        """

        marker_col = F.col(self.marker_column)
        bool_start = marker_col.eqNullSafe(self.marker_start).cast("integer")
        bool_end = marker_col.eqNullSafe(self.marker_end).cast("integer")
        bool_start_end = bool_start + bool_end
        return bool_start_end, marker_col

    def _identify_valids(self, raw_iids, w_id):
        """Identifies valid/invalid intervals

        Parameters
        ----------
        raw_iids: pyspark.sql.Column
            interval ids for intervals
        w_id: pyspark.sql.Window
            window function of the interval ids

        Returns
        -------
        bool_valid: pyspark.sql.Column
            valid intervals
        valis_ids: pysparl.sql.Column
            interval ids

        """

        col = F.col(self.marker_column)
        bool_start = col.eqNullSafe(self.marker_start).cast("integer")
        bool_end = col.eqNullSafe(self.marker_end).cast("integer")
        start = F.max(bool_start).over(w_id)
        end = F.max(bool_end).over(w_id)
        bool_valid = (start + end) == 2
        valid_ids = F.when(bool_valid, raw_iids).otherwise(0)

        return bool_valid, valid_ids

    def _prepare_iids(self, marker_col, w_lag, start_first,
                      add_negate_shift_col):
        """

        Parameters
        ----------
        marker_col: pyspark.sql.Column
            Column where the markers are in
        w_lag: pyspark.sql.Window
            Defining the window
        start_first: bool
            Defines if the first start is used for intervals
        add_negate_shift_col: bool
            True if the shift col have to be negated

        Returns
        -------
        result: NamedTuple
            returns the raw interval ids and the forward filled marker Column
        """
        # generate forward fill depending on interval
        if start_first:
            default = 0
            forward_fill = F.when(marker_col == self.marker_start, 1) \
                .when(marker_col == self.marker_end, 0) \
                .otherwise(None)
        else:
            default = 1
            forward_fill = F.when(marker_col == self.marker_end, 1) \
                .when(marker_col == self.marker_start, 0) \
                .otherwise(None)

        window = w_lag.rowsBetween(Window.unboundedPreceding, 0)
        forward_fill_col = F.last(forward_fill, ignorenulls=True).over(window)

        # shifting marker_col forward
        shift_col = F.lag(forward_fill_col, default=default, count=1) \
            .over(w_lag) \
            .cast("integer")

        # compare forward fill col and shifted forward fill col, if equal set to 0
        end_marker_null_col = F.when(shift_col == forward_fill_col, 0) \
            .otherwise(forward_fill_col)

        if add_negate_shift_col:
            # negate shift_col
            shift_col_negated = F.when(shift_col == 0, 1).otherwise(0)
            add_col = end_marker_null_col + shift_col_negated
        else:
            add_col = end_marker_null_col

        # build cum sum over window
        nt = NamedTuple("iids_ffill",
                        [("raw_iids", Column), ("forward_fill_col", Column)])
        return nt(F.sum(add_col).over(w_lag), forward_fill_col)

    def _continuous_renumeration(self, bool_valid, valid_ids, w_lag):
        """re-numerate ids from 1 to x and fill invalid with 0

        Parameters
        ----------
        bool_valid: pyspark.sql.Column
            Marked as valid
        valid_ids: pyspark.sql.Column
            values for valid ids
        w_lag: pyspark.sql.Window
            Defining window

        Returns
        -------
        result: pyspark.sql.Column
            Continuous renumerated interval ids. Can be increasing oder decreasing.

        """

        valid_ids_shift = F.lag(valid_ids, default=0).over(w_lag)
        valid_ids_diff = valid_ids_shift - valid_ids
        valid_ids_increase = (valid_ids_diff < 0).cast("integer")

        renumerate = F.sum(valid_ids_increase).over(w_lag)
        renumerate_adjusted = F.when(bool_valid, renumerate).otherwise(0)
        return renumerate_adjusted

    def transform(self, df: DataFrame) -> DataFrame:
        """Extract interval ids from given dataframe.

        Parameters
        ----------
        df: pyspark.sql.Dataframe

        Returns
        -------
        result: pyspark.sql.Dataframe
            Same columns as original dataframe plus the new interval id column.

        """

        start_first = self.marker_start_use_first
        end_first = self.marker_end_use_first

        pass_on = (self._identical_start_end_markers or
                   (start_first & ~end_first) or
                   (~start_first & end_first))

        if pass_on:
            super().transform(df)

        # check input
        self.validate_input(df)

        if start_first & end_first:
            return self._first_start_first_end(df)

        else:
            return self._last_start_last_end(df)

    def _first_start_first_end(self, df: DataFrame) -> DataFrame:
        """Extract interval ids from given dataframe.

        Parameters
        ----------
        df: pyspark.sql.Dataframe

        Returns
        -------
        result: pyspark.sql.Dataframe
            Same columns as original dataframe plus the new interval id column.

        """

        # define window specs
        orderby = util.prepare_orderby(self.order_columns, self.ascending)
        groupby = self.groupby_columns or []

        w_lag = Window.partitionBy(groupby).orderBy(orderby)
        w_id = Window.partitionBy(groupby + [self.target_column_name])

        bool_start_end, marker_col = self._marker_bool_start_end()

        raw_ffill = self._prepare_iids(marker_col, w_lag,
                                       start_first=True,
                                       add_negate_shift_col=True)
        df = df.withColumn(self.target_column_name, raw_ffill.raw_iids)

        # separate valid vs invalid: ids with start AND end marker are valid
        bool_valid, valid_ids = self._identify_valids(bool_start_end,
                                                      raw_ffill.raw_iids, w_id,
                                                      operator.ge)

        renumerate_adjusted = self._continuous_renumeration(bool_valid,
                                                            valid_ids, w_lag)

        return df.withColumn(self.target_column_name, renumerate_adjusted)

    def _last_start_last_end(self, df: DataFrame) -> DataFrame:
        """Extract interval ids from given dataframe.
        The ids are in continuously decreasing order. Invalid ids are 0.

        Parameters
        ----------
        df: pyspark.sql.Dataframe

        Returns
        -------
        result: pyspark.sql.Dataframe
            Same columns as original dataframe plus the new interval id column.

        """

        # define window specs
        orderby_reverse = util.prepare_orderby(self.order_columns,
                                               self.ascending, reverse=True)
        groupby = self.groupby_columns or []

        w_lag = Window.partitionBy(groupby).orderBy(orderby_reverse)
        w_id = Window.partitionBy(groupby + [self.target_column_name])

        bool_start_end, marker_col = self._marker_bool_start_end()

        raw_fill = self._prepare_iids(marker_col, w_lag,
                                      start_first=False,
                                      add_negate_shift_col=True)
        raw_iids = raw_fill.raw_iids + 1
        df = df.withColumn(self.target_column_name, raw_iids)

        # separate valid vs invalid: ids with start AND end marker are valid
        bool_valid, valid_ids = self._identify_valids(bool_start_end, raw_iids,
                                                      w_id, operator.ge)

        renumerate_adjusted = self._continuous_renumeration(bool_valid,
                                                            valid_ids, w_lag)

        return df.withColumn(self.target_column_name, renumerate_adjusted)
