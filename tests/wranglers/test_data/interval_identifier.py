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


@pytest.fixture(params=MARKER_TYPES.values(),
                ids=MARKER_TYPES.keys())
def empty_result(request):
    """Contains no identifiable interval.

    """

    begin, close, noise = _unpack_marker(request.param)

    data = [[1, 1, noise],
            [2, 1, close],
            [3, 1, noise],
            [4, 1, begin]]

    result = [0, 0, 0, 0]

    df_in = pd.DataFrame(data, columns=COLUMNS_STD)
    df_out = pd.DataFrame(result)

    return df_in, df_out
