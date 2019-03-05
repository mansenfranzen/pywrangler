"""This module contains the test data relevant for the interval identifier
wrangler.

"""

import pytest

import pandas as pd

COLUMNS_STD = ("order", "groupby", "marker")

MARKER_TYPES = {

    "string": {"begin": "begin",
               "close": "close",
               "noise": "noise"},

    "int": {"begin": 1,
            "close": 2,
            "noise": 3},

    "float": {"begin": 1.1,
              "close": 1.2,
              "noise": 1.3}
}


def _unpack_marker(marker_mapping):
    """Helper function to extract begin, close and noise marker.

    """

    begin = marker_mapping["begin"]
    close = marker_mapping["close"]
    noise = marker_mapping["noise"]

    return begin, close, noise


def _return_dfs(input, output, columns=COLUMNS_STD, index=None):
    """Helper function to return input and output dataframes.

    """

    df_in = pd.DataFrame(input, columns=columns, index=index)
    df_out = pd.DataFrame(output, index=index)

    return df_in, df_out


@pytest.fixture(params=MARKER_TYPES.values(),
                ids=MARKER_TYPES.keys())
def no_interval(request):
    """Contains no identifiable interval.

    """

    begin, close, noise = _unpack_marker(request.param)

    input = [[1, 1, noise],
             [2, 1, noise],
             [3, 1, noise],
             [4, 1, noise]]

    output = [0, 0, 0, 0]

    return _return_dfs(input, output)


@pytest.fixture(params=MARKER_TYPES.values(),
                ids=MARKER_TYPES.keys())
def single_interval(request):
    """Most basic example containing a single interval.

    """

    begin, close, noise = _unpack_marker(request.param)

    input = [[1, 1, noise],
             [2, 1, begin],
             [3, 1, close],
             [4, 1, noise]]

    output = [0, 1, 1, 0]

    return _return_dfs(input, output)


@pytest.fixture(params=MARKER_TYPES.values(),
                ids=MARKER_TYPES.keys())
def single_interval_spanning(request):
    """Most basic example containing a single interval.

    """

    begin, close, noise = _unpack_marker(request.param)

    input = [[1, 1, noise],
             [2, 1, begin],
             [3, 1, noise],
             [4, 1, close],
             [5, 1, noise]]

    output = [0, 1, 1, 1, 0]

    return _return_dfs(input, output)
