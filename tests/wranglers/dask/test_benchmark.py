"""This module contains tests for dask benchmarks.

isort:skip_file
"""

import time

import pytest
import pandas as pd
import numpy as np

pytestmark = pytest.mark.dask  # noqa: E402
dask = pytest.importorskip("dask")  # noqa: E402

from dask import dataframe as dd

from pywrangler.wranglers.dask.benchmark import DaskTimeProfiler
from pywrangler.wranglers.dask.base import DaskSingleNoFit

SLEEP = 0.0001


@pytest.fixture
def wrangler_sleeps():
    class DummyWrangler(DaskSingleNoFit):
        def transform(self, df):
            time.sleep(SLEEP)
            return df

    return DummyWrangler


def test_dask_time_profiler_fastest(wrangler_sleeps):
    """Basic test for dask time profiler ensuring fastest timing is slower
    than forced sleep.

    """

    df_input = dd.from_pandas(pd.DataFrame(np.random.rand(10, 10)), 2)

    time_profiler = DaskTimeProfiler(wrangler_sleeps(), 1).profile(df_input)

    assert time_profiler.best >= SLEEP
