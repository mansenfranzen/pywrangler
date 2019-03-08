
import pytest

import pandas as pd

from pywrangler.wranglers.pandas.interval_identifier import NaiveIterator

from ..test_data.interval_identifier import (
    ends_with_single_interval,
    groupby_multiple_intervals,
    groupby_multiple_more_intervals,
    groupby_single_intervals,
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

ALGORITHMS = (NaiveIterator, )
ALGORITHMS_IDS = [x.__name__ for x in ALGORITHMS]
ALGORITHMS_KWARGS = dict(argnames='algorithm',
                         argvalues=ALGORITHMS,
                         ids=ALGORITHMS_IDS)

SHUFFLE_KWARGS = dict(argnames='shuffle',
                      argvalues=(False, True),
                      ids=('Ordered', 'Shuffled'))

TEST_CASES = (no_interval, single_interval, single_interval_spanning,
              starts_with_single_interval, ends_with_single_interval,
              multiple_intervals, multiple_intervals_spanning,
              multiple_intervals_spanning_unsorted, groupby_multiple_intervals,
              groupby_single_intervals, groupby_multiple_more_intervals)
TEST_IDS = [x.__name__ for x in TEST_CASES]
TEST_KWARGS = dict(argnames='test_case',
                   argvalues=TEST_CASES,
                   ids=TEST_IDS)


@pytest.mark.parametrize(**TEST_KWARGS)
@pytest.mark.parametrize(**ALGORITHMS_KWARGS)
@pytest.mark.parametrize(**MARKERS_KWARGS)
@pytest.mark.parametrize(**SHUFFLE_KWARGS)
def test_interval_identifier(test_case, algorithm, marker, shuffle):
    """Tests against all available algorithms and test cases.

    """

    wrangler = algorithm("marker", marker["begin"], marker["close"],
                         "order", "groupby")

    test_input, expected = test_case(begin=marker["begin"],
                                     close=marker["close"],
                                     noise=marker["noise"],
                                     target_column_name="iids",
                                     shuffle=shuffle)

    test_output = wrangler.fit_transform(test_input)

    pd.testing.assert_frame_equal(test_output, expected)
