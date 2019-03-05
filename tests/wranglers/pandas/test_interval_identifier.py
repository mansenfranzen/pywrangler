
import pandas as pd

from pywrangler.wranglers.pandas.interval_identifier import NaiveIterator

from ..test_data.interval_identifier import empty_result


def test_empty_result(empty_result):

    test_input, test_output = empty_result

    wrangler = NaiveIterator("marker", "start", "end", "order", "groupby")

    pd.testing.assert_frame_equal(wrangler.fit_transform(test_input),
                                  test_output)
