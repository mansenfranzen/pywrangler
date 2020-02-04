"""This module contains the test data relevant for the interval identifier
wrangler.

"""

from pywrangler.util.testing import NaN, NULL, DataTestCase, PlainFrame, \
    TestCollection


MARKER_USE = {"FirstStartFirstEnd": {"marker_start_use_first": True,
                                    "marker_end_use_first": True},
             "FirstStartLastEnd": {"marker_start_use_first": True,
                                   "marker_end_use_first": False},
             "LastStartLastEnd": {"marker_start_use_first": False,
                                  "marker_end_use_first": False},
             "LastStartFirstEnd": {"marker_start_use_first": False,
                                   "marker_end_use_first": True}}


MARKER_USE_KWARGS = dict(argnames="marker_use",
                         argvalues=list(MARKER_USE.values()),
                         ids=list(MARKER_USE.keys()))


class _IntervalIdentifierTestCase(DataTestCase):

    marker_start = 1
    marker_end = 2
    marker_noise = 0
    marker_dtype = "int"

    marker_column = "marker"
    target_column_name = "iid"

    marker_start_use_first = False
    marker_end_use_first = True

    @property
    def test_dtypes(self):
        dtypes = ["int"] * len(self.test_columns)
        dtypes[-2] = self.marker_dtype
        return dtypes

    @property
    def test_columns(self):
        return (self.orderby_columns +
                self.groupby_columns +
                [self.marker_column, self.target_column_name])

    @property
    def test_kwargs(self):
        return dict(
            orderby_columns=self.orderby_columns,
            groupby_columns=self.groupby_columns,
            marker_column=self.marker_column,
            ascending=self.ascending,
            marker_start=self.marker_start,
            marker_end=self.marker_end,
            target_column_name=self.target_column_name,
            marker_start_use_first = self.marker_start_use_first,
            marker_end_use_first = self.marker_end_use_first
        )

    def input(self):
        return self.output[:-1]

    def output(self):
        data = self.data()

        return PlainFrame.from_plain(data=data,
                                     columns=self.test_columns,
                                     dtypes=self.test_dtypes)


class _SingleOrderGroup(_IntervalIdentifierTestCase):
    """Resembles test data with integer value and one orderby/groupby column.

    """

    orderby_columns = ["order"]
    groupby_columns = ["groupby"]
    ascending = [True]


class _SingleOrderGroupReverse(_SingleOrderGroup):
    """Resembles test data with integer value and one orderby/groupby column
    and reverse sort order.

    """

    ascending = [False]


class _SingleOrderGroupFloat(_SingleOrderGroup):
    """Resembles test data with float values.

    """

    marker_start = 0.1
    marker_end = 0.2
    marker_noise = 0.3
    marker_dtype = "float"


class _SingleOrderGroupString(_SingleOrderGroup):
    """Resembles test data with string values.

    """

    marker_start = "start"
    marker_end = "end"
    marker_noise = "noise"
    marker_dtype = "str"


class _MultiOrderGroup(_IntervalIdentifierTestCase):
    """Resembles test data with integer value and multiple orderby/groupby
    columns.

    """

    orderby_columns = ["order1", "order2"]
    groupby_columns = ["groupby1", "groupby2"]
    ascending = [True, True]


class _MultiOrderGroupReverse(_MultiOrderGroup):
    """Resembles test data with integer value and multiple orderby/groupby
    columns and reversed order.

    """

    ascending = [False, False]


class _FirstStartFirstEnd:
    marker_start_use_first = True
    marker_end_use_first = True


class _FirstStartLastEnd:
    marker_start_use_first = True
    marker_end_use_first = False


class _LastStartLastEnd:
    marker_start_use_first = False
    marker_end_use_first = False


class _LastStartFirstEnd:
    marker_start_use_first = False
    marker_end_use_first = True


class NoInterval(_SingleOrderGroup):
    def data(self):
        noise = self.marker_noise

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       noise,  0],
                [2,     1,       noise,  0],
                [3,     1,       noise,  0],
                [4,     1,       noise,  0]]

        return data


class NoIntervalInvalidStart(_SingleOrderGroup):
    def data(self):
        noise = self.marker_noise
        start = self.marker_start

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       noise,  0],
                [2,     1,       noise,  0],
                [3,     1,       start,  0],
                [4,     1,       noise,  0]]

        return data


class NoIntervalInvalidEnd(_SingleOrderGroup):
    def data(self):
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       noise,  0],
                [2,     1,       noise,  0],
                [3,     1,       end,    0],
                [4,     1,       noise,  0]]

        return data


class SingleInterval(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       end,    1],
                [4,     1,       noise,  0]]

        return data


class SingleIntervalStartsWith(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       start,  1],
                [2,     1,       end,    1],
                [3,     1,       noise,  0],
                [4,     1,       noise,  0]]

        return data


class SingleIntervalEndsWith(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       noise,  0],
                [2,     1,       noise,  0],
                [3,     1,       start,  1],
                [4,     1,       end,    1]]

        return data


class SingleIntervalSpanning(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       noise,  1],
                [4,     1,       end,    1],
                [5,     1,       noise,  0]]

        return data


class SingleIntervalSpanningGroupby(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       noise,  1],
                [4,     1,       end,    1],
                [5,     1,       noise,  0],
                [6,     2,       noise,  0],
                [7,     2,       noise,  0],
                [8,     2,       noise,  0]]

        return data


class SingleIntervalUnsorted(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[4,     1,       end,    1],
                [3,     1,       noise,  1],
                [2,     1,       start,  1],
                [5,     1,       noise,  0],
                [1,     1,       noise,  0]]

        return data


class SingleIntervalMissings(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       NULL,   0],
                [2,     1,       start,  1],
                [3,     1,       NULL,   1],
                [4,     1,       end,    1],
                [5,     1,       NULL,   0]]

        return data


class MultipleIntervals(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid"""
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       end,    1],
                [4,     1,       noise,  0],
                [5,     1,       start,  2],
                [6,     1,       end,    2],
                [7,     1,       noise,  0]]

        return data


class MultipleIntervalsReverse(_SingleOrderGroupReverse):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid"""
        data = [[1,     1,       noise,  0],
                [2,     1,       end,    2],
                [3,     1,       start,  2],
                [4,     1,       end,    1],
                [5,     1,       noise,  1],
                [6,     1,       start,  1],
                [7,     1,       noise,  0]]

        return data


class MultipleIntervalsSpanning(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid"""
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       end,    1],
                [4,     1,       noise,  0],
                [5,     1,       start,  2],
                [6,     1,       noise,  2],
                [7,     1,       end,    2],
                [8,     1,       noise,  0]]

        return data


class MultipleIntervalsSpanningFloat(MultipleIntervalsSpanning,
                                     _SingleOrderGroupFloat):
    pass


class MultipleIntervalsSpanningFloatNAN(MultipleIntervalsSpanning,
                                        _SingleOrderGroupFloat):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid"""
        data = [[1,     1,       NaN,    0],
                [2,     1,       start,  1],
                [3,     1,       end,    1],
                [4,     1,       NaN,    0],
                [5,     1,       start,  2],
                [6,     1,       NaN,    2],
                [7,     1,       end,    2],
                [8,     1,       noise,  0]]

        return data


class MultipleIntervalsSpanningString(MultipleIntervalsSpanning,
                                      _SingleOrderGroupString):
    pass


class MultipleIntervalsSpanningGroupby(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid"""
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       end,    1],
                [4,     1,       noise,  0],
                [5,     2,       start,  1],
                [6,     2,       noise,  1],
                [7,     2,       end,    1],
                [8,     2,       noise,  0]]

        return data


class MultipleIntervalsSpanningGroupbyExtended(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid"""
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       end,    1],
                [4,     1,       noise,  0],

                [5,     2,       start,  1],
                [6,     2,       noise,  1],
                [7,     2,       end,    1],
                [8,     2,       noise,  0],
                [9,     2,       noise,  0],
                [10,    2,       start,  2],
                [11,    2,       noise,  2],
                [12,    2,       end,    2],
                [13,    2,       start,  3],
                [14,    2,       end,    3]]

        return data


class MultipleIntervalsSpanningGroupbyExtendedTriple(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid"""
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       end,    1],
                [4,     1,       noise,  0],

                [5,     2,       start,  1],
                [6,     2,       noise,  1],
                [7,     2,       end,    1],
                [8,     2,       noise,  0],
                [9,     2,       noise,  0],

                [10,    3,       start,  1],
                [11,    3,       noise,  1],
                [12,    3,       end,    1],
                [13,    3,       start,  2],
                [14,    3,       end,    2]]

        return data


class MultipleIntervalsUnsorted(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid"""
        data = [[6,     1,       noise,  2],
                [3,     1,       end,    1],
                [4,     1,       noise,  0],
                [8,     1,       noise,  0],
                [1,     1,       noise,  0],
                [7,     1,       end,    2],
                [2,     1,       start,  1],
                [5,     1,       start,  2]]

        return data


class MultipleIntervalsMissing(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid"""
        data = [[1,     1,       NULL,   0],
                [2,     1,       start,  1],
                [3,     1,       end,    1],
                [4,     1,       noise,  0],
                [5,     1,       start,  2],
                [6,     1,       end,    2],
                [7,     1,       NULL,   0]]

        return data


class InvalidStartsWithEnd(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       end,    0],
                [2,     1,       end,    0],
                [3,     1,       end,    0],
                [4,     1,       noise,  0],
                [5,     1,       start,  1],
                [6,     1,       end,    1]]

        return data


class InvalidEndsWithStart(_SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       end,    1],
                [4,     1,       start,  0],
                [5,     1,       start,  0],
                [6,     1,       start,  0]]

        return data


class MultipleOrderGroupby(_MultiOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order1, order2, groupby1, groupby2, marker, iid"""
        data = [[1,      1,      1,        1,        noise,  0],
                [1,      2,      1,        1,        start,  1],
                [2,      1,      1,        1,        end,    1],
                [2,      2,      1,        1,        noise,  0],

                [3,      1,      1,        2,        start,  1],
                [3,      2,      1,        2,        noise,  1],
                [4,      1,      1,        2,        end,    1],
                [4,      2,      1,        2,        noise,  0],

                [1,      1,      2,        1,        start,  1],
                [1,      2,      2,        1,        end,    1],
                [2,      1,      2,        1,        start,  2],
                [2,      2,      2,        1,        end,    2],

                [3,      1,      2,        2,        start,  1],
                [3,      2,      2,        2,        noise,  1],
                [4,      1,      2,        2,        end,    1],
                [4,      2,      2,        2,        noise,  0]]

        return data


class MultipleOrderGroupbyReverse(_MultiOrderGroupReverse):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order1, order2, groupby1, groupby2, marker, iid"""
        data = [[1,      1,      1,        1,        end,    2],
                [1,      2,      1,        1,        start,  2],
                [2,      1,      1,        1,        end,    1],
                [2,      2,      1,        1,        start,  1],

                [3,      1,      1,        2,        start,  0],
                [3,      2,      1,        2,        end,    1],
                [4,      1,      1,        2,        noise,  1],
                [4,      2,      1,        2,        start,  1],

                [1,      1,      2,        1,        start,  0],
                [1,      2,      2,        1,        end,    1],
                [2,      1,      2,        1,        start,  1],
                [2,      2,      2,        1,        end,    0],

                [3,      1,      2,        2,        start,  0],
                [3,      2,      2,        2,        noise,  0],
                [4,      1,      2,        2,        end,    0],
                [4,      2,      2,        2,        noise,  0]]

        return data


class MultipleOrderGroupbyMissing(_MultiOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order1, order2, groupby1, groupby2, marker, iid"""
        data = [[1,      1,      1,        1,        NULL,   0],
                [1,      2,      1,        1,        start,  1],
                [2,      1,      1,        1,        end,    1],
                [2,      2,      1,        1,        noise,  0],

                [3,      1,      1,        2,        start,  1],
                [3,      2,      1,        2,        noise,  1],
                [4,      1,      1,        2,        end,    1],
                [4,      2,      1,        2,        NULL,   0],

                [5,      1,      1,        2,        noise,  0],
                [5,      2,      1,        2,        NULL,   0],
                [5,      3,      1,        2,        start,  2],
                [5,      4,      1,        2,        end,    2],

                [3,      1,      2,        2,        start,  1],
                [3,      2,      2,        2,        end,    1],
                [4,      1,      2,        2,        NULL,   0],
                [4,      2,      2,        2,        noise,  0]]

        return data


class MultipleOrderGroupbyMissingUnsorted(_MultiOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order1, order2, groupby1, groupby2, marker, iid"""
        data = [[1,      1,      1,        1,        NULL,   0],
                [4,      1,      1,        2,        end,    1],
                [5,      3,      1,        2,        start,  2],
                [3,      1,      2,        2,        start,  1],
                [2,      1,      1,        1,        end,    1],
                [3,      1,      1,        2,        start,  1],
                [5,      1,      1,        2,        noise,  0],
                [3,      2,      1,        2,        noise,  1],
                [1,      2,      1,        1,        start,  1],
                [4,      2,      1,        2,        NULL,   0],
                [2,      2,      1,        1,        noise,  0],
                [5,      2,      1,        2,        NULL,   0],
                [4,      1,      2,        2,        NULL,   0],
                [5,      4,      1,        2,        end,    2],
                [3,      2,      2,        2,        end,    1],
                [4,      2,      2,        2,        noise,  0]]

        return data


CollectionGeneral = TestCollection([
    NoInterval,
    NoIntervalInvalidStart,
    NoIntervalInvalidEnd,
    SingleInterval,
    SingleIntervalStartsWith,
    SingleIntervalEndsWith,
    SingleIntervalSpanning,
    SingleIntervalSpanningGroupby,
    SingleIntervalUnsorted,
    SingleIntervalMissings,
    MultipleIntervals,
    MultipleIntervalsReverse,
    MultipleIntervalsSpanning,
    MultipleIntervalsSpanningFloat,
    MultipleIntervalsSpanningFloatNAN,
    MultipleIntervalsSpanningString,
    MultipleIntervalsSpanningGroupby,
    MultipleIntervalsSpanningGroupbyExtended,
    MultipleIntervalsSpanningGroupbyExtendedTriple,
    MultipleIntervalsUnsorted,
    MultipleIntervalsMissing,
    InvalidStartsWithEnd,
    InvalidEndsWithStart,
    MultipleOrderGroupby,
    MultipleOrderGroupbyReverse,
    MultipleOrderGroupbyMissing,
    MultipleOrderGroupbyMissingUnsorted
], test_kwargs={"marker_use": MARKER_USE})


class _IdenticalStartEnd:
    marker_start = 1
    marker_end = 1


class IdenticalStartEndSingleInterval(_IdenticalStartEnd, _SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       noise,  1],
                [4,     1,       noise,  1]]

        return data


class IdenticalStartEndMultipleInterval(_IdenticalStartEnd, _SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       noise,  1],
                [4,     1,       start,  2],
                [5,     1,       noise,  2],
                [6,     1,       start,  3],
                [7,     1,       noise,  3],
                [8,     1,       noise,  3]]

        return data


class IdenticalStartEndMultipleIntervalReversed(_IdenticalStartEnd,
                                                _SingleOrderGroupReverse):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       noise,  3],
                [2,     1,       start,  3],
                [3,     1,       noise,  2],
                [4,     1,       start,  2],
                [5,     1,       noise,  1],
                [6,     1,       start,  1],
                [7,     1,       noise,  0],
                [8,     1,       noise,  0]]

        return data


class IdenticalStartEndMultipleIntervalMissing(_IdenticalStartEnd,
                                               _SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       NULL,   0],
                [2,     1,       start,  1],
                [3,     1,       noise,  1],
                [4,     1,       start,  2],
                [5,     1,       NULL,   2],
                [6,     1,       start,  3],
                [7,     1,       noise,  3],
                [8,     1,       NULL,   3]]

        return data


class IdenticalStartEndMultipleIntervalMissingUnsorted(_IdenticalStartEnd,
                                                       _SingleOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise

        # cols:  order, groupby, marker, iid
        data = [[5,     1,       NULL,   2],
                [2,     1,       start,  1],
                [6,     1,       start,  3],
                [4,     1,       start,  2],
                [8,     1,       NULL,   3],
                [1,     1,       NULL,   0],
                [3,     1,       noise,  1],
                [7,     1,       noise,  3]]

        return data


class IdenticalStartEndMultipleOrderGroupbyMissing(_IdenticalStartEnd,
                                                   _MultiOrderGroup):
    def data(self):
        start = self.marker_start
        noise = self.marker_noise

        # cols:  order1, order2, groupby1, groupby2, marker, iid"""
        data = [[1,      1,      1,        1,        NULL,   0],
                [1,      2,      1,        1,        start,  1],
                [2,      1,      1,        1,        NULL,   1],
                [2,      2,      1,        1,        noise,  1],

                [3,      1,      1,        2,        start,  1],
                [3,      2,      1,        2,        noise,  1],
                [4,      1,      1,        2,        start,  2],
                [4,      2,      1,        2,        NULL,   2],

                [5,      1,      1,        2,        noise,  2],
                [5,      2,      1,        2,        NULL,   2],
                [5,      3,      1,        2,        start,  3],
                [5,      4,      1,        2,        noise,  3],

                [3,      1,      2,        2,        start,  1],
                [3,      2,      2,        2,        start,  2],
                [4,      1,      2,        2,        NULL,   2],
                [4,      2,      2,        2,        noise,  2]]

        return data


CollectionIdenticalStartEnd = TestCollection([
    IdenticalStartEndSingleInterval,
    IdenticalStartEndMultipleInterval,
    IdenticalStartEndMultipleIntervalReversed,
    IdenticalStartEndMultipleIntervalMissing,
    IdenticalStartEndMultipleIntervalMissingUnsorted,
    IdenticalStartEndMultipleOrderGroupbyMissing
])


class FirstStartFirstEndValid(_FirstStartFirstEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       start,  1],
                [2,     1,       start,  1],
                [3,     1,       start,  1],
                [4,     1,       noise,  1],
                [5,     1,       start,  1],
                [6,     1,       noise,  1],
                [7,     1,       start,  1],
                [8,     1,       noise,  1],
                [9,     1,       end,    1],
                [10,    1,       noise,  0],
                [11,    1,       end,    0],
                [12,    1,       noise,  0],
                [13,    1,       end,    0],
                [14,    1,       end,    0],
                [15,    1,       end,    0]]

        return data


class FirstStartFirstEndInvalid(_FirstStartFirstEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       end,    0],
                [2,     1,       end,    0],
                [3,     1,       end,    0],
                [4,     1,       noise,  0],
                [5,     1,       end,    0],
                [6,     1,       noise,  0],
                [7,     1,       start,  1],
                [8,     1,       start,  1],
                [9,     1,       noise,  1],
                [10,    1,       start,  1],
                [11,    1,       noise,  1],
                [12,    1,       end,    1],
                [13,    1,       end,    0],
                [14,    1,       noise,  0],
                [15,    1,       end,    0],
                [16,    1,       noise,  0],
                [17,    1,       start,  0],
                [18,    1,       noise,  0],
                [19,    1,       start,  0],
                [20,    1,       start,  0],
                [21,    1,       start,  0]]

        return data


class FirstStartFirstEndInvalidMissing(_FirstStartFirstEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       end,    0],
                [2,     1,       end,    0],
                [3,     1,       end,    0],
                [4,     1,       NULL,   0],
                [5,     1,       end,    0],
                [6,     1,       NULL,   0],
                [7,     1,       start,  1],
                [8,     1,       start,  1],
                [9,     1,       NULL,   1],
                [10,    1,       start,  1],
                [11,    1,       NULL,   1],
                [12,    1,       end,    1],
                [13,    1,       end,    0],
                [14,    1,       NULL,   0],
                [15,    1,       end,    0],
                [16,    1,       NULL,   0],
                [17,    1,       start,  0],
                [18,    1,       NULL,   0],
                [19,    1,       start,  0],
                [20,    1,       start,  0],
                [21,    1,       start,  0]]

        return data


class FirstStartLastEndValid(_FirstStartLastEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       start,  1],
                [2,     1,       start,  1],
                [3,     1,       start,  1],
                [4,     1,       noise,  1],
                [5,     1,       start,  1],
                [6,     1,       noise,  1],
                [7,     1,       start,  1],
                [8,     1,       noise,  1],
                [9,     1,       end,    1],
                [10,    1,       noise,  1],
                [11,    1,       end,    1],
                [12,    1,       noise,  1],
                [13,    1,       end,    1],
                [14,    1,       end,    1],
                [15,    1,       end,    1]]

        return data


class FirstStartLastEndInvalid(_FirstStartLastEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       end,    0],
                [2,     1,       end,    0],
                [3,     1,       end,    0],
                [4,     1,       noise,  0],
                [5,     1,       end,    0],
                [6,     1,       noise,  0],
                [7,     1,       start,  1],
                [8,     1,       start,  1],
                [9,     1,       noise,  1],
                [10,    1,       start,  1],
                [11,    1,       noise,  1],
                [12,    1,       end,    1],
                [13,    1,       end,    1],
                [14,    1,       noise,  1],
                [15,    1,       end,    1],
                [16,    1,       noise,  0],
                [17,    1,       start,  0],
                [18,    1,       noise,  0],
                [19,    1,       start,  0],
                [20,    1,       start,  0],
                [21,    1,       start,  0]]

        return data


class FirstStartLastEndInvalidMissing(_FirstStartLastEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       end,    0],
                [2,     1,       end,    0],
                [3,     1,       end,    0],
                [4,     1,       NULL,   0],
                [5,     1,       end,    0],
                [6,     1,       NULL,   0],
                [7,     1,       start,  1],
                [8,     1,       start,  1],
                [9,     1,       NULL,   1],
                [10,    1,       start,  1],
                [11,    1,       NULL,   1],
                [12,    1,       end,    1],
                [13,    1,       end,    1],
                [14,    1,       NULL,   1],
                [15,    1,       end,    1],
                [16,    1,       NULL,   0],
                [17,    1,       start,  0],
                [18,    1,       NULL,   0],
                [19,    1,       start,  0],
                [20,    1,       start,  0],
                [21,    1,       start,  0]]

        return data


class LastStartFirstEndValid(_LastStartFirstEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       start,  0],
                [2,     1,       start,  0],
                [3,     1,       start,  0],
                [4,     1,       noise,  0],
                [5,     1,       start,  0],
                [6,     1,       noise,  0],
                [7,     1,       start,  1],
                [8,     1,       noise,  1],
                [9,     1,       end,    1],
                [10,    1,       noise,  0],
                [11,    1,       end,    0],
                [12,    1,       noise,  0],
                [13,    1,       end,    0],
                [14,    1,       end,    0],
                [15,    1,       end,    0]]

        return data


class LastStartFirstEndInvalid(_LastStartFirstEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       end,    0],
                [2,     1,       end,    0],
                [3,     1,       end,    0],
                [4,     1,       noise,  0],
                [5,     1,       end,    0],
                [6,     1,       noise,  0],
                [7,     1,       start,  0],
                [8,     1,       start,  0],
                [9,     1,       noise,  0],
                [10,    1,       start,  1],
                [11,    1,       noise,  1],
                [12,    1,       end,    1],
                [13,    1,       end,    0],
                [14,    1,       noise,  0],
                [15,    1,       end,    0],
                [16,    1,       noise,  0],
                [17,    1,       start,  0],
                [18,    1,       noise,  0],
                [19,    1,       start,  0],
                [20,    1,       start,  0],
                [21,    1,       start,  0]]

        return data


class LastStartFirstEndInvalidMissing(_LastStartFirstEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       end,    0],
                [2,     1,       end,    0],
                [3,     1,       end,    0],
                [4,     1,       NULL,   0],
                [5,     1,       end,    0],
                [6,     1,       NULL,   0],
                [7,     1,       start,  0],
                [8,     1,       start,  0],
                [9,     1,       NULL,   0],
                [10,    1,       start,  1],
                [11,    1,       NULL,   1],
                [12,    1,       end,    1],
                [13,    1,       end,    0],
                [14,    1,       NULL,   0],
                [15,    1,       end,    0],
                [16,    1,       NULL,   0],
                [17,    1,       start,  0],
                [18,    1,       NULL,   0],
                [19,    1,       start,  0],
                [20,    1,       start,  0],
                [21,    1,       start,  0]]

        return data


class LastStartLastEndValid(_LastStartLastEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       start,  0],
                [2,     1,       start,  0],
                [3,     1,       start,  0],
                [4,     1,       noise,  0],
                [5,     1,       start,  0],
                [6,     1,       noise,  0],
                [7,     1,       start,  1],
                [8,     1,       noise,  1],
                [9,     1,       end,    1],
                [10,    1,       noise,  1],
                [11,    1,       end,    1],
                [12,    1,       noise,  1],
                [13,    1,       end,    1],
                [14,    1,       end,    1],
                [15,    1,       end,    1]]

        return data


class LastStartLastEndInvalid(_LastStartLastEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       end,    0],
                [2,     1,       end,    0],
                [3,     1,       end,    0],
                [4,     1,       noise,  0],
                [5,     1,       end,    0],
                [6,     1,       noise,  0],
                [7,     1,       start,  0],
                [8,     1,       start,  0],
                [9,     1,       noise,  0],
                [10,    1,       start,  1],
                [11,    1,       noise,  1],
                [12,    1,       end,    1],
                [13,    1,       end,    1],
                [14,    1,       noise,  1],
                [15,    1,       end,    1],
                [16,    1,       noise,  0],
                [17,    1,       start,  0],
                [18,    1,       noise,  0],
                [19,    1,       start,  0],
                [20,    1,       start,  0],
                [21,    1,       start,  0]]

        return data


class LastStartLastEndInvalidMissing(_LastStartLastEnd, _SingleOrderGroup):

    def data(self):
        start = self.marker_start
        end = self.marker_end

        # cols:  order, groupby, marker, iid
        data = [[1,     1,       end,    0],
                [2,     1,       end,    0],
                [3,     1,       end,    0],
                [4,     1,       NULL,   0],
                [5,     1,       end,    0],
                [6,     1,       NULL,   0],
                [7,     1,       start,  0],
                [8,     1,       start,  0],
                [9,     1,       NULL,   0],
                [10,    1,       start,  1],
                [11,    1,       NULL,   1],
                [12,    1,       end,    1],
                [13,    1,       end,    1],
                [14,    1,       NULL,   1],
                [15,    1,       end,    1],
                [16,    1,       NULL,   0],
                [17,    1,       start,  0],
                [18,    1,       NULL,   0],
                [19,    1,       start,  0],
                [20,    1,       start,  0],
                [21,    1,       start,  0]]

        return data


CollectionMarkerSpecifics = TestCollection([
    LastStartLastEndValid,
    LastStartLastEndInvalid,
    LastStartLastEndInvalidMissing,
    LastStartFirstEndValid,
    LastStartFirstEndInvalid,
    LastStartFirstEndInvalidMissing,
    FirstStartLastEndValid,
    FirstStartLastEndInvalid,
    FirstStartLastEndInvalidMissing,
    FirstStartFirstEndValid,
    FirstStartFirstEndInvalid,
    FirstStartFirstEndInvalidMissing,
])


class ResultTypeRawIids(_SingleOrderGroup):

    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid"""
        data = [[1,     1,       noise,  0],
                [2,     1,       start,  1],
                [3,     1,       end,    1],
                [4,     1,       noise,  2],
                [5,     1,       start,  3],
                [6,     1,       noise,  3],
                [7,     1,       end,    3],
                [8,     1,       noise,  4],
                [9,     1,       noise,  4],
                [10,    1,       start,  5],
                [11,    1,       noise,  5],
                [12,    1,       end,    5],
                [13,    1,       start,  6],
                [14,    1,       end,    6]]

        return data


class ResultTypeValidIids(_SingleOrderGroup):

    def data(self):
        start = self.marker_start
        noise = self.marker_noise
        end = self.marker_end

        # cols:  order, groupby, marker, iid"""
        data = [[1,     1,       end,    0],
                [2,     1,       noise,  0],
                [3,     1,       end,    0],
                [4,     1,       start,  5],
                [5,     1,       noise,  5],
                [6,     1,       end,    5],
                [7,     1,       noise,  0],
                [8,     1,       start,  4],
                [9,     1,       end,    4],
                [10,    1,       start,  9],
                [11,    1,       noise,  9],
                [12,    1,       end,    9],
                [13,    1,       start,  0],
                [14,    1,       start,  0]]

        return data


MISSING_ORDER_GROUP_BY = {'no_order': {'groupby_columns': 'groupby',
                                       'orderby_columns': None,
                                       'ascending': None},
                          'no_groupby': {'groupby_columns': None,
                                         'orderby_columns': 'order'},
                          'no_order_no_groupby': {'groupby_columns': None,
                                                  'orderby_columns': None,
                                                  'ascending': None}}

CollectionNoOrderGroupBy = TestCollection([
    NoInterval,
    NoIntervalInvalidStart,
    NoIntervalInvalidEnd,
    SingleInterval,
    SingleIntervalStartsWith,
    SingleIntervalEndsWith,
    SingleIntervalSpanning,
    SingleIntervalSpanningGroupby]
    , test_kwargs={"missing_order_group_by": MISSING_ORDER_GROUP_BY})
