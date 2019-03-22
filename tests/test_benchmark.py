"""This module contains tests for the benchmark utilities.

"""

import sys

import pytest

import numpy as np
import pandas as pd

from pywrangler.benchmark import (
    BaseProfiler,
    MemoryProfiler,
    PandasMemoryProfiler,
    allocate_memory
)
from pywrangler.exceptions import NotProfiledError
from pywrangler.wranglers.pandas.base import PandasSingleNoFit

MIB = 2 ** 20


def test_allocate_memory_empty():
    memory_holder = allocate_memory(0)

    assert memory_holder is None


def test_allocate_memory_5mb():
    memory_holder = allocate_memory(5)

    assert sys.getsizeof(memory_holder) == 5 * (2 ** 20)


def test_base_profiler_not_implemented():
    base_profiler = BaseProfiler()

    for will_raise in ('profile', 'profile_report'):
        with pytest.raises(NotImplementedError):
            getattr(base_profiler, will_raise)()


def test_base_profiler_check_is_profiled():
    base_profiler = BaseProfiler()
    base_profiler._not_set = None
    base_profiler._is_set = "value"

    with pytest.raises(NotProfiledError):
        base_profiler._check_is_profiled(['_not_set'])

    base_profiler._check_is_profiled(['_is_set'])


def test_base_profiler_mb_to_bytes():
    assert BaseProfiler._mb_to_bytes(1) == 1048576
    assert BaseProfiler._mb_to_bytes(1.5) == 1572864
    assert BaseProfiler._mb_to_bytes(0.33) == 346030


def test_memory_profiler_return_self():
    def dummy():
        pass

    memory_profiler = MemoryProfiler(dummy)
    assert memory_profiler.profile() is memory_profiler


def test_memory_profiler_properties():
    def dummy():
        pass

    memory_profiler = MemoryProfiler(dummy)
    memory_profiler._baselines = [0, 1, 2, 3]
    memory_profiler._max_usages = [4, 5, 7, 8]

    assert memory_profiler.max_usages == memory_profiler._max_usages
    assert memory_profiler.baselines == memory_profiler._baselines
    assert memory_profiler.increases == [4, 4, 5, 5]
    assert memory_profiler.increases_mean == 4.5
    assert memory_profiler.increases_std == 0.5
    assert memory_profiler.baseline_change == 1


def test_memory_profiler_no_side_effect():
    def no_side_effect():
        dummy = 5
        return dummy

    assert MemoryProfiler(no_side_effect).profile().baseline_change < 0.5 * MIB


def test_memory_profiler_side_effect():
    side_effect_container = []

    def side_effect():
        memory_holder = allocate_memory(5)
        side_effect_container.append(memory_holder)

        return memory_holder

    assert MemoryProfiler(side_effect).profile().baseline_change > 4.9 * MIB


def test_memory_profiler_no_increase():
    def no_increase():
        pass

    assert MemoryProfiler(no_increase).profile().increases_mean < 0.1 * MIB
    assert MemoryProfiler(no_increase).profile().increases_std < 0.1 * MIB


def test_memory_profiler_increase():
    def increase():
        memory_holder = allocate_memory(30)
        return memory_holder

    assert MemoryProfiler(increase).profile().increases_mean > 29 * MIB


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

    assert memory_profiler.profile(empty_df).usage_increases_mean > 29 * MIB


def test_pandas_memory_profiler_usage_input_output():
    df_input = pd.DataFrame(np.random.rand(1000))
    df_output = pd.DataFrame(np.random.rand(10000))

    test_df_input = df_input.memory_usage(index=True, deep=True).sum()
    test_df_output = df_output.memory_usage(index=True, deep=True).sum()

    class DummyWrangler(PandasSingleNoFit):
        def transform(self, df):
            return df_output

    memory_profiler = PandasMemoryProfiler(DummyWrangler()).profile(df_input)

    assert memory_profiler.usage_input == test_df_input
    assert memory_profiler.usage_output == test_df_output


def test_pandas_memory_profiler_usage_ratio():
    usage_mib = 30
    df_input = pd.DataFrame(np.random.rand(1000000))
    usage_input = df_input.memory_usage(index=True, deep=True).sum()
    test_output = ((usage_mib - 1) * MIB) / usage_input

    class DummyWrangler(PandasSingleNoFit):
        def transform(self, df):
            return pd.DataFrame(allocate_memory(usage_mib))

    memory_profiler = PandasMemoryProfiler(DummyWrangler())

    assert memory_profiler.profile(df_input).usage_ratio > test_output
