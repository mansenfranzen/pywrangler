"""This module contains computation engine independent wrangler interfaces
and corresponding descriptions.

"""

from typing import Any, Iterable, Union

from pywrangler.util import sanitizer
from pywrangler.wranglers.base import BaseWrangler

TYPE_COLUMNS = Union[str, Iterable[str]]


class IntervalIdentifier(BaseWrangler):
    """Defines the reference interface for the interval identification
    wrangler.

    An interval is defined as a range of values beginning with an opening
    marker and ending with a closing marker (e.g. the interval daylight may be
    defined as all events/values occurring between sunrise and sunset).

    The interval identification wrangler assigns ids to values such that values
    belonging to the same interval share the same interval id. For example, all
    values of the first daylight interval are assigned with id 1. All values of
    the second daylight interval will be assigned with id 2 and so on.

    Values which do not belong to any valid interval are assigned the value 0
    by definition.

    Only the shortest valid interval is identified. Given multiple opening
    markers in sequence without an intermittent closing marker, only the last
    opening marker is relevant and the rest is ignored. Given multiple
    closing markers in sequence without an intermittent opening marker, only
    the first closing marker is relevant and the rest is ignored.

    Opening and closing markers are included in their corresponding interval.

    Parameters
    ----------
    marker_column: str
        Name of column which contains the opening and closing markers.
    marker_start: Any
        A value defining the start of an interval.
    marker_end: Any
        A value defining the end of an interval.
    order_columns: str, Iterable[str], optional
        Column names which define the order of the data (e.g. a timestamp
        column). Sort order can be defined with the parameter `sort_order`.
    groupby_columns: str, Iterable[str], optional
        Column names which define how the data should be grouped/split into
        separate entities. For distributed computation engines, groupby columns
        should ideally reference partition keys to avoid data shuffling.
    sort_order: str, Iterable[str], optional
        Explicitly define the sort order of given `order_columns` with
        `ascending` and `descending`.
    target_column_name: str, optional
        Name of the resulting target column.

    """

    def __init__(self,
                 marker_column: str,
                 marker_start: Any,
                 marker_end: Any,
                 order_columns: TYPE_COLUMNS = None,
                 groupby_columns: TYPE_COLUMNS = None,
                 sort_order: TYPE_COLUMNS = None,
                 target_column_name: str = "iids"):

        self.marker_column = marker_column
        self.marker_start = marker_start
        self.marker_end = marker_end
        self.order_columns = sanitizer.ensure_tuple(order_columns)
        self.groupby_columns = sanitizer.ensure_tuple(groupby_columns)
        self.sort_order = sanitizer.ensure_tuple(sort_order)
        self.target_column_name = target_column_name

        # sanity checks for sort order
        if self.sort_order:

            # check for equal number of items of order and sort columns
            if len(self.order_columns) != len(self.sort_order):
                raise ValueError('`order_columns` and `sort_order` must have '
                                 'equal number of items.')

            # check for correct sorting keywords
            allow_values = ('ascending', 'descending')
            if any([x not in allow_values for x in self.sort_order]):
                raise ValueError('Only `ascending` and `descencing` are '
                                 'allowed as keywords for `sort_order`')

    @property
    def preserves_sample_size(self) -> bool:
        return True
