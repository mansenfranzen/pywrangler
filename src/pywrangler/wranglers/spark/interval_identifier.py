"""This module contains implementations of the interval identifier wrangler.

"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from pywrangler.wranglers.interfaces import IntervalIdentifier
from pywrangler.wranglers.spark.base import SparkSingleNoFit


class VectorizedCumSum(SparkSingleNoFit, IntervalIdentifier):
    """Sophisticated approach avoiding python UDFs. However multiple windows
    are necessary.

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

        # shifting the close marker allows cumulative sum to include the end
        window_lag = Window.partitionBy(list(self.groupby_columns)) \
            .orderBy(list(self.order_columns))

        window_id = Window.partitionBy(list(self.groupby_columns) +
                                       [self.target_column_name])

        # get boolean series with start and end markers
        marker_col = F.col(self.marker_column)
        bool_start = (marker_col == self.marker_start).cast("integer")
        bool_end = (marker_col == self.marker_end).cast("integer")
        bool_end_shift = F.lag(bool_end, default=1) \
            .over(window_lag) \
            .cast("integer")

        ser_id = F.sum(bool_start + bool_end_shift) \
            .over(window_lag) \
            .alias(self.target_column_name)
        valid = F.sum(bool_start + bool_end).over(window_id) == 2

        zero = F.when(valid, ser_id).otherwise(0)
        lag_zero = (F.lag(zero, default=0).over(window_lag) - zero) < 0
        change = F.sum(lag_zero.cast("integer")).over(window_lag)
        change_adjusted = F.when(valid, change).otherwise(0)

        df = df.withColumn(self.target_column_name, ser_id)

        return df.withColumn(self.target_column_name, change_adjusted)
