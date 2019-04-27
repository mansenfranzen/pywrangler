"""This module contains tests for pandas benchmarks.

"""

import time

import pytest

import numpy as np
import pandas as pd

from pywrangler.benchmark import allocate_memory
from pywrangler.wranglers.pandas.base import PandasSingleNoFit
from pywrangler.wranglers.pandas.benchmark import (
    PandasMemoryProfiler,
    PandasTimeProfiler
)

MIB = 2 ** 20


@pytest.fixture
def test_wrangler():
    """Helper fixture to generate PandasWrangler instances with parametrization
    of transform output and sleep.

    """

    def create_wrangler(size=None, result=None, sleep=0):
        """Return instance of PandasWrangler.

        Parameters
        ----------
        size: float
            Memory size in MiB to allocate during transform step.
        result: pd.DataFrame
            Define extact return value of transform step.
        sleep: float
            Define sleep interval.

        """

        class DummyWrangler(PandasSingleNoFit):
            def transform(self, df):
                if size is not None:
                    df_out = pd.DataFrame(allocate_memory(size))
                else:
                    df_out = pd.DataFrame(result)

                time.sleep(sleep)
                return df_out

        return DummyWrangler()

    return create_wrangler


def test_pandas_memory_profiler_memory_usage_dfs():
    df1 = pd.DataFrame(np.random.rand(10))
    df2 = pd.DataFrame(np.random.rand(10))

    test_input = [df1, df2]
    test_output = int(df1.memory_usage(index=True, deep=True).sum() +
                      df2.memory_usage(index=True, deep=True).sum())

    assert PandasMemoryProfiler._memory_usage_dfs(*test_input) == test_output


def test_pandas_memory_profiler_return_self(test_wrangler):
    memory_profiler = PandasMemoryProfiler(test_wrangler())

    assert memory_profiler is memory_profiler.profile(pd.DataFrame())


@pytest.mark.xfail(reason="Succeeds locally but sometimes fails remotely due "
                          "to non deterministic memory management.")
def test_pandas_memory_profiler_usage_median(test_wrangler):
    wrangler = test_wrangler(size=30, sleep=0.01)
    memory_profiler = PandasMemoryProfiler(wrangler)

    assert memory_profiler.profile(pd.DataFrame()).median > 29 * MIB


def test_pandas_memory_profiler_usage_input_output(test_wrangler):
    df_input = pd.DataFrame(np.random.rand(1000))
    df_output = pd.DataFrame(np.random.rand(10000))

    test_df_input = df_input.memory_usage(index=True, deep=True).sum()
    test_df_output = df_output.memory_usage(index=True, deep=True).sum()

    wrangler = test_wrangler(result=df_output)
    memory_profiler = PandasMemoryProfiler(wrangler).profile(df_input)

    assert memory_profiler.input == test_df_input
    assert memory_profiler.output == test_df_output


@pytest.mark.xfail(reason="Succeeds locally but sometimes fails remotely due "
                          "to non deterministic memory management.")
def test_pandas_memory_profiler_ratio(test_wrangler):
    usage_mib = 30
    df_input = pd.DataFrame(np.random.rand(1000000))
    usage_input = df_input.memory_usage(index=True, deep=True).sum()
    test_output = ((usage_mib - 1) * MIB) / usage_input

    wrangler = test_wrangler(size=usage_mib, sleep=0.01)

    memory_profiler = PandasMemoryProfiler(wrangler)

    assert memory_profiler.profile(df_input).ratio > test_output


def test_pandas_time_profiler_best(test_wrangler):
    """Basic test for pandas time profiler ensuring fastest timing is slower
    than forced sleep.

    """

    sleep = 0.0001
    wrangler = test_wrangler(sleep=sleep)
    time_profiler = PandasTimeProfiler(wrangler, 1).profile(pd.DataFrame())

    assert time_profiler.best >= sleep
