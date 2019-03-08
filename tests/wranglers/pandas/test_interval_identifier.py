
import pytest

import pandas as pd

from pywrangler.wranglers.pandas.interval_identifier import NaiveIterator

from ..test_data.interval_identifier import (
    begin_marker_left_open,
    close_marker_starts,
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
    starts_with_single_interval
)

MARKER_TYPES = {"string": {"begin": "begin",
                           "close": "close",
                           "noise": "noise"},

                "int": {"begin": 1,
                        "close": 2,
                        "noise": 3},

                "float": {"begin": 1.1,
                          "close": 1.2,
                          "noise": 1.3}}

MARKERS = MARKER_TYPES.values()
MARKERS_IDS = list(MARKER_TYPES.keys())
MARKERS_KWARGS = dict(argnames='marker',
                      argvalues=MARKERS,
                      ids=MARKERS_IDS)

WRANGLER = (NaiveIterator,)
WRANGLER_IDS = [x.__name__ for x in WRANGLER]
WRANGLER_KWARGS = dict(argnames='algorithm',
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
              multiple_groupby_order_columns_reverse, begin_marker_left_open,
              close_marker_starts)
TEST_IDS = [x.__name__ for x in TEST_CASES]
TEST_KWARGS = dict(argnames='test_case',
                   argvalues=TEST_CASES,
                   ids=TEST_IDS)


@pytest.mark.parametrize(**SHUFFLE_KWARGS)
@pytest.mark.parametrize(**MARKERS_KWARGS)
@pytest.mark.parametrize(**TEST_KWARGS)
@pytest.mark.parametrize(**WRANGLER_KWARGS)
def test_pandas_interval_identifier(test_case, algorithm, marker, shuffle):
    """Tests against all available algorithms and test cases.

    Parameters
    ----------
    test_case: function
        Generates test data for given test case. Refers to `TEST_CASES`.
    algorithm: pywrangler.wrangler.interfaces.IntervalIdentifier
        Refers to the actual wrangler begin tested. See `WRANGLER`.
    marker: dict
        Defines the type of data which is used to generate test data. See
        `MARKERS`.
    shuffle: bool
        Define if the data gets shuffled or not.

    """

    # generate test_input and expected result
    test_input, expected = test_case(begin=marker["begin"],
                                     close=marker["close"],
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
        kwargs = {"order_columns": ("order1" ,"order2"),
                  "groupby_columns": ("groupby1", "groupby2"),
                  "ascending": sort_order*2}

    else:
        raise ValueError("Incorrect number of columns for test data. "
                         "See module test_data/interval_identifier.py")

    # instantiate actual wrangler
    wrangler = algorithm(marker_column="marker",
                         marker_start=marker["begin"],
                         marker_end=marker["close"],
                         **kwargs)

    test_output = wrangler.fit_transform(test_input)
    pd.testing.assert_frame_equal(test_output, expected)
