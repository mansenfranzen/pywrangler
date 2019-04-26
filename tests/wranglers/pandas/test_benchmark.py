"""This module contains tests for pandas benchmarks.

"""

import time

import numpy as np
import pandas as pd

from pywrangler.benchmark import allocate_memory
from pywrangler.wranglers.pandas.base import PandasSingleNoFit
from pywrangler.wranglers.pandas.benchmark import (
    PandasMemoryProfiler,
    PandasTimeProfiler
)

MIB = 2 ** 20


def test_pandas_memory_profiler_memory_usage_dfs():
    df1 = pd.DataFrame(np.random.rand(10))
    df2 = pd.DataFrame(np.random.rand(10))

    test_input = [df1, df2]
    test_output = int(df1.memory_usage(index=True, deep=True).sum() +
                      df2.memory_usage(index=True, deep=True).sum())

    assert PandasMemoryProfiler._memory_usage_dfs(*test_input) == test_output


def test_pandas_memory_profiler_return_self():
    class DummyWrangler(PandasSingleNoFit):
        def transform(self, df):
            return pd.DataFrame()

    memory_profiler = PandasMemoryProfiler(DummyWrangler())

    assert memory_profiler is memory_profiler.profile(pd.DataFrame())


def test_pandas_memory_profiler_usage_increases_mean():
    empty_df = pd.DataFrame()

    class DummyWrangler(PandasSingleNoFit):
        def transform(self, df):
            return pd.DataFrame(allocate_memory(30))

    memory_profiler = PandasMemoryProfiler(DummyWrangler())

    assert memory_profiler.profile(empty_df).median > 29 * MIB


def test_pandas_memory_profiler_usage_input_output():
    df_input = pd.DataFrame(np.random.rand(1000))
    df_output = pd.DataFrame(np.random.rand(10000))

    test_df_input = df_input.memory_usage(index=True, deep=True).sum()
    test_df_output = df_output.memory_usage(index=True, deep=True).sum()

    class DummyWrangler(PandasSingleNoFit):
        def transform(self, df):
            return df_output

    memory_profiler = PandasMemoryProfiler(DummyWrangler()).profile(df_input)

    assert memory_profiler.input == test_df_input
    assert memory_profiler.output == test_df_output


def test_pandas_memory_profiler_usage_ratio():
    usage_mib = 30
    df_input = pd.DataFrame(np.random.rand(1000000))
    usage_input = df_input.memory_usage(index=True, deep=True).sum()
    test_output = ((usage_mib - 1) * MIB) / usage_input

    class DummyWrangler(PandasSingleNoFit):
        def transform(self, df):
            return pd.DataFrame(allocate_memory(usage_mib))

    memory_profiler = PandasMemoryProfiler(DummyWrangler())

    assert memory_profiler.profile(df_input).ratio > test_output


def test_pandas_time_profiler_fastest():
    """Basic test for pandas time profiler ensuring fastest timing is slower
    than forced sleep.

    """

    sleep = 0.0001
    df_input = pd.DataFrame()

    class DummyWrangler(PandasSingleNoFit):
        def transform(self, df):
            time.sleep(sleep)
            return df

    time_profiler = PandasTimeProfiler(DummyWrangler(), 1).profile(df_input)

    assert time_profiler.best >= sleep
