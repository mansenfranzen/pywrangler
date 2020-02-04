"""This module contains implementations of the interval identifier wrangler.

"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import Column

from pywrangler.pyspark import util
from pywrangler.pyspark.base import PySparkSingleNoFit
from pywrangler.wranglers import IntervalIdentifier


class VectorizedCumSum(PySparkSingleNoFit, IntervalIdentifier):
    """Sophisticated approach without using python UDFs which is based on
    multiple window functions.

    The algorithm is specialized to work with the shortest sequence possible.
    This addresses intervals with last start and first end marker. If first
    start marker or last end marker is required, a preprocessing step is
    executed to remove multiple start/end markers accordingly.

    After preprocessing, the algorithm works as follows: First, get increasing
    ids (numbers) for all intervals with no regard to valid or invalid
    intervals. Every time a start or end marker is encountered, increase
    interval id by one. The end marker is shifted by one to include the end
    marker in the current interval. This is realized via the cumulative sum of
    boolean series of start markers and shifted end markers. This resembles the
    raw iids.

    Second, separate valid from invalid intervals by ensuring the presence
    of both start and end markers per interval id.

    Third, numerate valid intervals starting with 1 and set invalid
    intervals to 0.

    """

    def _validate_input(self, df: DataFrame):
        """Checks input data frame in regard to column names.

        Parameters
        ----------
        df: pyspark.sql.Dataframe
            Dataframe to be validated.

        """

        util.validate_columns(df, self.marker_column)
        util.validate_columns(df, self.orderby_columns)
        util.validate_columns(df, self.groupby_columns)

        if self.orderby_columns is None:
            raise ValueError("Please define an order column. Pyspark "
                             "dataframes have no implicit order unlike pandas "
                             "dataframes.")

    def _boolify_marker(self, marker_column, start=True) -> Column:
        """Helper function to create an integer casted boolean column
        expression of start/end marker.

        """

        if start:
            marker = self.marker_start
        else:
            marker = self.marker_end

        return marker_column.eqNullSafe(marker).cast("integer")

    @staticmethod
    def _reverse_enumeration(column: Column, window: Window) -> Column:
        """Helper function to reverse enumeration of given column.

        Parameters
        ----------
        column: pyspark.sql.column.Column
            Column containing enumerated values.
        window: pyspark.sql.Window
            Window spec for which the reversing should take place.

        Returns
        -------
        reverse: pyspark.sql.column.Column

        """

        window_max = window.rowsBetween(Window.unboundedPreceding,
                                        Window.unboundedFollowing)

        diff_score = F.max(column).over(window_max) + 1
        reverse = (column - diff_score) * -1

        return reverse

    def _denoise_marker_column(self, window, start=True) -> Column:
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

    def _drop_duplicated_marker(self, marker_column: Column, window: Window,
                                start: bool = True) -> Column:
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

    def _window_groupby(self, reverse: bool = False) -> Window:
        """Generate pyspark sql window which represents the main window
        corresponding to the given groupby and orderby columns.

        Parameters
        ----------
        reverse: bool, optional
            Define order by to be reversed or not.

        Returns
        -------
        window pyspark.sql.Window

        """

        orderby = util.prepare_orderby(self.orderby_columns, self.ascending,
                                       reverse=reverse)
        groupby = self.groupby_columns or []

        return Window.partitionBy(groupby).orderBy(orderby)

    def _window_raw_iids(self, column_name: str) -> Window:
        """Generate pyspark sql window which represents the raw iids window
        corresponding to the given groupby columns plus the intermediate
        column representation of the raw iids.

        Parameters
        ----------
        column_name: str
            Name of the column which contains the raw iids.

        Returns
        -------
        window pyspark.sql.Window

        """

        groupby = self.groupby_columns or []

        return Window.partitionBy(groupby + [column_name])

    def _preprocess_marker_column(self) -> Column:
        """If required, removes duplicated start/end markers and returns
        modified marker column which is ready to be further processed.

        Returns
        -------
        col: pyspark.sql.column.Column
            Modified marker column.

        """

        window = self._window_groupby()

        col = F.col(self.marker_column)
        if self._identical_start_end_markers:
            return col

        if self.marker_start_use_first:
            col = self._drop_duplicated_marker(col, window)

        if not self.marker_end_use_first:
            col = self._drop_duplicated_marker(col, window, False)

        return col

    def _generate_raw_iids(self, marker_column: Column) -> Column:
        """Create sequence of interval ids in increasing order regardless of
        their validity.

        Parameters
        ----------
        marker_column: pyspark.sql.column.Column
            Column expression resembling the marker column.

        Returns
        -------
        raw_iids: pyspark.sql.column.Column

        """

        window = self._window_groupby()

        bool_start = self._boolify_marker(marker_column, True)
        bool_end = self._boolify_marker(marker_column, False)

        # shifting the end marker allows cumulative sum to include the end
        bool_end_shift = F.lag(bool_end, default=1).over(window)
        bool_start_end_shift = bool_start + bool_end_shift

        # get increasing ids for intervals (in/valid) with cumsum
        raw_iids = F.sum(bool_start_end_shift).over(window)

        return raw_iids

    def _generate_valid_iids(self, marker_column: Column,
                             raw_iids: Column,
                             exact: bool,
                             cc: util.ColumnCacher) -> Column:
        """Create sequence of interval identifier ids in increasing order
        with invalid intervals being removed. Invalid iids will be set to 0.

        Parameters
        ----------
        marker_column: pyspark.sql.column.Column
            Column expression resembling the marker column.
        raw_iids: pyspark.sql.column.Column
            Column expression with raw iids.
        exact: bool
            If True, start and end markers need to appear exactly once for each
            valid interval. If False, start and end markers may appear more
            than one time (but at least one time) per valid interval.

        Returns
        -------
        valid_iids: pyspark.sql.column.Column

        """

        bool_start = self._boolify_marker(marker_column, True)
        bool_end = self._boolify_marker(marker_column, False)

        raw_iids_name = cc.columns["raw_iids"]
        window = self._window_raw_iids(raw_iids_name)

        if exact:
            summed = F.sum(bool_start + bool_end).over(window)
        else:
            max_start = F.max(bool_start).over(window)
            max_end = F.max(bool_end).over(window)
            summed = max_start + max_end

        bool_valid = (summed == 2)
        valid_iids = F.when(bool_valid, raw_iids).otherwise(0)

        return valid_iids

    def _generate_renumerated_iids(self, valid_iids: Column,
                                   reverse: bool = False) -> Column:
        """Create sequence of interval identifier ids in increasing order
        starting with 1 in steps of 1. Invalid intervals are marked with 0.

        Parameters
        ----------
        valid_iids: pyspark.sql.column.Column
            Column expression resembling valid iids.
        reverse: bool, optional
            Reverse order of enumeration. Highest number will be lowest and
            vice versa.

        Returns
        -------
        raw_iids: pyspark.sql.column.Column

        """

        window = self._window_groupby(reverse=reverse)

        # get interval changes
        valid_ids_shift = F.lag(valid_iids, default=0).over(window)
        valid_ids_diff = valid_ids_shift - valid_iids
        valid_ids_increase = (valid_ids_diff < 0).cast("integer")

        # renumerate based on changes
        renumerate = F.sum(valid_ids_increase).over(window)
        if reverse:
            renumerate = self._reverse_enumeration(renumerate, window)

        # reset invalids
        bool_valid = valid_iids != 0
        renumerated_iids = F.when(bool_valid, renumerate).otherwise(0)

        return renumerated_iids

    def _generate_iids_identical(self, marker_col: Column) -> Column:
        """Compute interval ids for identical start and end markers.

        Parameters
        ----------
        marker_column: pyspark.sql.column.Column
            Column expression resembling the marker column.

        Returns
        -------
        iids: pyspark.sql.column.Column

        """

        window = self._window_groupby()

        bool_start = self._boolify_marker(marker_col)
        iids = F.sum(bool_start).over(window)

        return iids

    def _compute_valid_renumerated_iids(self,
                                        marker_col: Column,
                                        iids_raw: Column,
                                        cc: util.ColumnCacher,
                                        valid_iids_exact: bool,
                                        renumerated_iids_reverse: bool) \
            -> DataFrame:
        """Compute valid and renumerated iids based on given parameters.

        Parameters
        ----------
        marker_col: pyspark.sql.column.Column
            Pyspark column expression representing the preprocessed/raw marker
            column.
        iids_raw: pyspark.sql.column.Column
            Pyspark column expression representing the raw iids.
        cc: util.ColumnCacher
            Column cacher.
        valid_iids_exact: bool
            Define if valid iid identification requires exact start/end-sum
            match.
        renumerated_iids_reverse: bool
            Define if renumeration is required to be in reverse order.

        Returns
        -------
        result: pyspark.sql.Dataframe
            Same columns as original dataframe plus the new interval id column.

        """

        if self.result_type == "raw":
            return cc.finish(self.target_column_name, iids_raw)

        # valid iids
        iids_raw = cc.add("raw_iids", iids_raw)
        iids_valid = self._generate_valid_iids(marker_col,
                                               iids_raw,
                                               valid_iids_exact,
                                               cc)

        if self.result_type == "valid":
            return cc.finish(self.target_column_name, iids_valid)

        # renumerated iids
        iids_valid = cc.add("valid_ids", iids_valid)
        iids_renumerated = self._generate_renumerated_iids(
            iids_valid,
            renumerated_iids_reverse)

        return cc.finish(self.target_column_name, iids_renumerated)

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
        self._validate_input(df)

        # cacher
        cc = util.ColumnCacher(df, True)

        # get preprocessed marker col
        marker_col = self._preprocess_marker_column()

        # early exit for identical start/end markers
        if self._identical_start_end_markers:
            iids = self._generate_iids_identical(marker_col)
            return df.withColumn(self.target_column_name, iids)

        # raw iids
        iids_raw = self._generate_raw_iids(marker_col)

        return self._compute_valid_renumerated_iids(marker_col,
                                                    iids_raw,
                                                    cc,
                                                    True,
                                                    False)


class VectorizedCumSumAdjusted(VectorizedCumSum):
    """Extends generic solution and provides faster implementations for last
    start / last end and  first start / first end which do not require
    proprocessing of the marker column. The DAGs of these implementations have
    2-3 steps less.

    """

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

        # delegate to super for unhandled implementations
        if (self._identical_start_end_markers or
                self.result_type == "raw" or
                (start_first & ~end_first) or
                (~start_first & end_first)):
            return super().transform(df)

        # check input
        self._validate_input(df)

        if start_first & end_first:
            return self._first_start_first_end(df)

        else:
            return self._last_start_last_end(df)

    def _generate_raw_iids_special(self, start_first: bool,
                                   add_negate_shift_col: bool,
                                   reverse=False) -> Column:
        """Create sequence of interval ids in increasing order regardless of
        their validity.

        Parameters
        ----------
        start_first: bool
            Defines if the first start is used for intervals.
        add_negate_shift_col: bool
            True if the shift col have to be negated.
        reverse: bool, optional
            Define order by.

        Returns
        -------
        raw_iids: pyspark.sql.column.Column

        """

        marker_col = F.col(self.marker_column)
        window = self._window_groupby(reverse)

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

        ff_window = window.rowsBetween(Window.unboundedPreceding, 0)
        forward_fill_col = F.last(forward_fill, ignorenulls=True).over(
            ff_window)

        # shifting marker_col forward
        shift_col = F.lag(forward_fill_col, default=default, count=1) \
            .over(window) \
            .cast("integer")

        # compare forward fill col and shifted forward fill col
        end_marker_null_col = F.when(shift_col == forward_fill_col, 0) \
            .otherwise(forward_fill_col)

        if add_negate_shift_col:
            shift_col_negated = F.when(shift_col == 0, 1).otherwise(0)
            add_col = end_marker_null_col + shift_col_negated
        else:
            add_col = end_marker_null_col

        # build cum sum over window
        raw_iids = F.sum(add_col).over(window)

        return raw_iids

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

        # cacher
        cc = util.ColumnCacher(df, True)

        # raw iids
        iids_raw = self._generate_raw_iids_special(start_first=True,
                                                   add_negate_shift_col=True)

        marker_col = F.col(self.marker_column)

        return self._compute_valid_renumerated_iids(marker_col,
                                                    iids_raw,
                                                    cc,
                                                    False,
                                                    False)

    def _last_start_last_end(self, df: DataFrame) -> DataFrame:
        """Extract interval ids from given dataframe.

        Parameters
        ----------
        df: pyspark.sql.Dataframe

        Returns
        -------
        result: pyspark.sql.Dataframe
            Same columns as original dataframe plus the new interval id column.

        """

        # cacher
        cc = util.ColumnCacher(df, True)

        # raw iids
        iids_raw = self._generate_raw_iids_special(start_first=False,
                                                   add_negate_shift_col=True,
                                                   reverse=True)
        iids_raw = iids_raw + 1
        marker_col = F.col(self.marker_column)

        return self._compute_valid_renumerated_iids(marker_col,
                                                    iids_raw,
                                                    cc,
                                                    False,
                                                    True)
