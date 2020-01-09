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

    Values which do not belong to any valid interval are assigned the value 0
    by definition (if start and end marker are identical, there are only
    invalid values possible before the first start marker is encountered).

    Four Intervals can be selected by using the following params:
    `marker_start_use_first` and `marker_end_use_first`. The resulting
    intervals are:
        - identical start and end markers
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
    order_columns: str, Iterable[str], optional
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
    target_column_name: str, optional
        Name of the resulting target column.

    """

    def __init__(self,
                 marker_column: str,
                 marker_start,
                 marker_end: Any = NONEVALUE,
                 marker_start_use_first=False,
                 marker_end_use_first=True,
                 order_columns: TYPE_COLUMNS = None,
                 groupby_columns: TYPE_COLUMNS = None,
                 ascending: TYPE_ASCENDING = None,
                 target_column_name: str = "iids"):

        self.marker_column = marker_column
        self.marker_start = marker_start
        self.marker_end = marker_end
        self.marker_start_use_first = marker_start_use_first
        self.marker_end_use_first = marker_end_use_first
        self.order_columns = sanitizer.ensure_iterable(order_columns)
        self.groupby_columns = sanitizer.ensure_iterable(groupby_columns)
        self.ascending = sanitizer.ensure_iterable(ascending)
        self.target_column_name = target_column_name

        # check for identical start and end values
        self._identical_start_end_markers = ((marker_end == NONEVALUE) or
                                             (marker_start == marker_end))

        # sanity checks for sort order
        if self.ascending:

            # check for equal number of items of order and sort columns
            if len(self.order_columns) != len(self.ascending):
                raise ValueError('`order_columns` and `ascending` must have '
                                 'equal number of items.')

            # check for correct sorting keywords
            if not all([isinstance(x, bool) for x in self.ascending]):
                raise ValueError('Only `True` and `False` are '
                                 'as arguments for `ascending`')

        # set default sort order if None is given
        elif self.order_columns:
            self.ascending = tuple([True] * len(self.order_columns))

    @property
    def preserves_sample_size(self) -> bool:
        return True
