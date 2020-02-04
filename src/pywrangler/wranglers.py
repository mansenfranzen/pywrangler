"""This module contains computation engine independent wrangler interfaces
and corresponding descriptions.

"""
from typing import Any

from pywrangler.base import BaseWrangler
from pywrangler.util import sanitizer
from pywrangler.util.types import TYPE_ASCENDING, TYPE_COLUMNS

NONEVALUE = object()


class IntervalIdentifier(BaseWrangler):
    """Defines the reference interface for the interval identification
    wrangler.

    An interval is defined as a range of values beginning with an opening
    marker and ending with a closing marker (e.g. the interval daylight may be
    defined as all events/values occurring between sunrise and sunset). Start
    and end marker may be identical.

    The interval identification wrangler assigns ids to values such that values
    belonging to the same interval share the same interval id. For example, all
    values of the first daylight interval are assigned with id 1. All values of
    the second daylight interval will be assigned with id 2 and so on.

    By default, values which do not belong to any valid interval, are assigned
    the value 0 by definition (please refer to `result_type` for different
    result types). If start and end marker are identical or the end marker is
    not provided, invalid values are only possible before the first start
    marker is encountered.

    Due to messy data, start and end marker may occur multiple times in
    sequence until its counterpart is reached. Therefore, intervals may have
    different spans based on different task requirements. For example, the very
    first start or very last start marker may define the correct start of an
    interval. Accordingly, four intervals can be selected by setting
    `marker_start_use_first` and `marker_end_use_first`. The resulting
    intervals are as follows:

        - first start / first end
        - first start / last end (longest interval)
        - last start / first end (shortest interval)
        - last start / last end

    Opening and closing markers are included in their corresponding interval.

    Parameters
    ----------
    marker_column: str
        Name of column which contains the opening and closing markers.
    marker_start: Any
        A value defining the start of an interval.
    marker_end: Any, optional
        A value defining the end of an interval. This value is optional. If not
        given, the end marker equals the start marker.
    marker_start_use_first: bool
        Identifies if the first occurring `marker_start` of an interval is used.
        Otherwise the last occurring `marker_start` is used. Default is False.
    marker_end_use_first: bool
        Identifies if the first occurring `marker_end` of an interval is used.
        Otherwise the last occurring `marker_end` is used. Default is True.
    orderby_columns: str, Iterable[str], optional
        Column names which define the order of the data (e.g. a timestamp
        column). Sort order can be defined with the parameter `ascending`.
    groupby_columns: str, Iterable[str], optional
        Column names which define how the data should be grouped/split into
        separate entities. For distributed computation engines, groupby columns
        should ideally reference partition keys to avoid data shuffling.
    ascending: bool, Iterable[bool], optional
        Sort ascending vs. descending. Specify list for multiple sort orders.
        If a list is specified, length of the list must equal length of
        `order_columns`. Default is True.
    result_type: str, optional
        Defines the content of the returned result. If 'raw', interval ids
        will be in arbitrary order with no distinction made between valid and
        invalid intervals. Intervals are distinguishable by interval id but the
        interval id may not provide any more information. If 'valid', the
        result is the same as 'raw' but all invalid intervals are set to 0.
        If 'enumerated', the result is the same as 'valid' but interval ids
        increase in ascending order (as defined by order) in steps of one.
    target_column_name: str, optional
        Name of the resulting target column.

    """

    def __init__(self,
                 marker_column: str,
                 marker_start: Any,
                 marker_end: Any = NONEVALUE,
                 marker_start_use_first: bool = False,
                 marker_end_use_first: bool = True,
                 orderby_columns: TYPE_COLUMNS = None,
                 groupby_columns: TYPE_COLUMNS = None,
                 ascending: TYPE_ASCENDING = None,
                 result_type: str = "enumerated",
                 target_column_name: str = "iids"):

        self.marker_column = marker_column
        self.marker_start = marker_start
        self.marker_end = marker_end
        self.marker_start_use_first = marker_start_use_first
        self.marker_end_use_first = marker_end_use_first
        self.orderby_columns = sanitizer.ensure_iterable(orderby_columns)
        self.groupby_columns = sanitizer.ensure_iterable(groupby_columns)
        self.ascending = sanitizer.ensure_iterable(ascending)
        self.result_type = result_type
        self.target_column_name = target_column_name

        # check correct result type
        valid_result_types = {"raw", "valid", "enumerated"}
        if result_type not in valid_result_types:
            raise ValueError("Parameter `result_type` is invalid with: {}. "
                             "Allowed arguments are: {}"
                             .format(result_type, valid_result_types))

        # check for identical start and end values
        self._identical_start_end_markers = ((marker_end == NONEVALUE) or
                                             (marker_start == marker_end))

        # sanity checks for sort order
        if self.ascending:

            # check for equal number of items of order and sort columns
            if len(self.orderby_columns) != len(self.ascending):
                raise ValueError('`order_columns` and `ascending` must have '
                                 'equal number of items.')

            # check for correct sorting keywords
            if not all([isinstance(x, bool) for x in self.ascending]):
                raise ValueError('Only `True` and `False` are '
                                 'allowed arguments for `ascending`')

        # set default sort order if None is given
        elif self.orderby_columns:
            self.ascending = [True] * len(self.orderby_columns)

    @property
    def preserves_sample_size(self) -> bool:
        return True
