import numpy as np
import pytest

from tests.test_data.interval_identifier import (
    end_marker_begins,
    ends_with_single_interval,
    groupby_multiple_intervals,
    groupby_multiple_intervals_reverse,
    groupby_multiple_more_intervals,
    groupby_single_intervals,
    invalid_end,
    invalid_start,
    multiple_groupby_order_columns,
    multiple_groupby_order_columns_reverse,
    multiple_groupby_order_columns_with_invalids,
    multiple_intervals,
    multiple_intervals_spanning,
    multiple_intervals_spanning_unsorted,
    no_interval,
    single_interval,
    single_interval_spanning,
    start_marker_left_open,
    starts_with_single_interval,
    identical_start_end_with_invalids)

from pywrangler.pandas.wranglers.interval_identifier import (
    NaiveIterator,
    VectorizedCumSum
)

# ensure backwards compatibility beyond pandas 0.20.0
try:
    from pandas.testing import assert_frame_equal
except ImportError:
    from pandas.util.testing import assert_frame_equal

pytestmark = pytest.mark.pandas

MARKER_TYPES = {"string": {"start": "start",
                           "end": "end",
                           "noise": "noise"},

                "int": {"start": 1,
                        "end": 2,
                        "noise": 3},

                "float": {"start": 1.1,
                          "end": 1.2,
                          "noise": 1.3},
                "nan": {"start": 1,
                        "end": 2,
                        "noise": np.NaN}}

MARKERS = MARKER_TYPES.values()
MARKERS_IDS = list(MARKER_TYPES.keys())
MARKERS_KWARGS = dict(argnames='marker',
                      argvalues=MARKERS,
                      ids=MARKERS_IDS)

WRANGLER = (NaiveIterator, VectorizedCumSum)
WRANGLER_IDS = [x.__name__ for x in WRANGLER]
WRANGLER_KWARGS = dict(argnames='wrangler',
                       argvalues=WRANGLER,
                       ids=WRANGLER_IDS)

SHUFFLE_KWARGS = dict(argnames='shuffle',
                      argvalues=(False, True),
                      ids=('Ordered', 'Shuffled'))

TEST_CASES = (no_interval, single_interval, single_interval_spanning,
              starts_with_single_interval, ends_with_single_interval,
              multiple_intervals, multiple_intervals_spanning,
              multiple_intervals_spanning_unsorted, groupby_multiple_intervals,
              groupby_single_intervals, groupby_multiple_more_intervals,
              multiple_groupby_order_columns, invalid_end, invalid_start,
              multiple_groupby_order_columns_with_invalids,
              groupby_multiple_intervals_reverse,
              multiple_groupby_order_columns_reverse, start_marker_left_open,
              end_marker_begins)
TEST_CASE_IDS = [x.__name__ for x in TEST_CASES]
TEST_CASE_KWARGS = dict(argnames='test_case',
                        argvalues=TEST_CASES,
                        ids=TEST_CASE_IDS)

TEST_CASES_NO_ORDER_GROUP = (no_interval, single_interval,
                             single_interval_spanning,
                             starts_with_single_interval,
                             ends_with_single_interval,
                             multiple_intervals, multiple_intervals_spanning,
                             invalid_end, invalid_start,
                             start_marker_left_open,
                             end_marker_begins)
TEST_CASES_NO_ORDER_GROUP_IDS = [x.__name__ for x in TEST_CASES_NO_ORDER_GROUP]
TEST_CASES_NO_ORDER_GROUP_KWARGS = dict(argnames='test_case',
                                        argvalues=TEST_CASES_NO_ORDER_GROUP,
                                        ids=TEST_CASES_NO_ORDER_GROUP_IDS)

GROUPBY_ORDER_TYPES = {'no_order': {'groupby_columns': 'groupby'},
                       'no_groupby': {'order_columns': 'order'},
                       'no_order_no_groupby': {}}
GROUPBY_ORDER = GROUPBY_ORDER_TYPES.values()
GROUPBY_ORDER_IDS = list(GROUPBY_ORDER_TYPES.keys())
GROUPBY_ORDER_KWARGS = dict(argnames='groupby_order',
                            argvalues=GROUPBY_ORDER,
                            ids=GROUPBY_ORDER_IDS)


@pytest.mark.parametrize(**SHUFFLE_KWARGS)
@pytest.mark.parametrize(**MARKERS_KWARGS)
@pytest.mark.parametrize(**TEST_CASE_KWARGS)
@pytest.mark.parametrize(**WRANGLER_KWARGS)
def test_groupby_order_columns(test_case, wrangler, marker, shuffle):
    """Tests against all available wranglers and test cases for different input
    types and shuffled data.

    Parameters
    ----------
    test_case: function
        Generates test data for given test case. Refers to `TEST_CASES`.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.
    marker: dict
        Defines the type of data which is used to generate test data. See
        `MARKERS`.
    shuffle: bool
        Define if the data gets shuffled or not.

    """

    # generate test_input and expected result
    test_input, expected = test_case(start=marker["start"],
                                     end=marker["end"],
                                     noise=marker["noise"],
                                     target_column_name="iids",
                                     shuffle=shuffle)

    # determine sort order, if test_case ends with 'reverse', than switch
    if test_case.__name__.endswith("reverse"):
        sort_order = [False]
    else:
        sort_order = [True]

    # determine correct order and groupby columns dependant on test data shape
    n_cols = test_input.shape[1]
    if n_cols == 3:
        kwargs = {"order_columns": "order",
                  "groupby_columns": "groupby",
                  "ascending": sort_order}

    elif n_cols == 5:
        kwargs = {"order_columns": ("order1", "order2"),
                  "groupby_columns": ("groupby1", "groupby2"),
                  "ascending": sort_order * 2}

    else:
        raise ValueError("Incorrect number of columns for test data. "
                         "See module test_data/interval_identifier.py")

    # instantiate actual wrangler_instance
    wrangler_instance = wrangler(marker_column="marker",
                                 marker_start=marker["start"],
                                 marker_end=marker["end"],
                                 **kwargs)

    test_output = wrangler_instance.fit_transform(test_input)
    assert_frame_equal(test_output, expected, check_dtype=False)


@pytest.mark.parametrize(**GROUPBY_ORDER_KWARGS)
@pytest.mark.parametrize(**TEST_CASES_NO_ORDER_GROUP_KWARGS)
@pytest.mark.parametrize(**WRANGLER_KWARGS)
def test_no_groupby_order_columns(test_case, wrangler, groupby_order):
    """Tests wranglers while not defining order or/and groupby columns on a
    subset of test cases which do not have a specific order or groupby.

    Parameters
    ----------
    test_case: function
        Generates test data for given test case. Refers to
        `TEST_CASES_NO_ORDER_GROUP`.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.
    groupby_order dict
        Defines the order and groupby columns. See `GROUPBY_ORDER_TYPES`.

    """

    marker = MARKER_TYPES["int"]

    # generate test_input and expected result
    test_input, expected = test_case(start=marker["start"],
                                     end=marker["end"],
                                     noise=marker["noise"],
                                     target_column_name="iids",
                                     shuffle=False)

    # instantiate actual wrangler_instance
    wrangler_instance = wrangler(marker_column="marker",
                                 marker_start=marker["start"],
                                 marker_end=marker["end"],
                                 **groupby_order)

    test_output = wrangler_instance.fit_transform(test_input)

    assert_frame_equal(test_output, expected)


@pytest.mark.parametrize(**MARKERS_KWARGS)
def test_identical_start_end_marker(marker):
    test_input, expected = identical_start_end_with_invalids(start=marker["start"],
                                                             noise=marker["noise"],
                                                             target_column_name="iids")

    wrangler_instance = VectorizedCumSum(marker_column="marker",
                                         marker_start=marker["start"],
                                         order_columns="order",
                                         groupby_columns="groupby")

    test_output = wrangler_instance.fit_transform(test_input)

    assert_frame_equal(test_output, expected, check_dtype=False)
