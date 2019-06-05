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

from pywrangler.benchmark import allocate_memory
from pywrangler.wranglers.dask.benchmark import (
    DaskTimeProfiler,
    DaskMemoryProfiler,
    DaskBaseProfiler
)
from pywrangler.wranglers.dask.base import DaskSingleNoFit


@pytest.fixture
def mean_wranger():
    class DummyWrangler(DaskSingleNoFit):
        def transform(self, df):
            return df.mean()

    return DummyWrangler()


@pytest.fixture
def test_wrangler():
    """Helper fixture to generate DaskWrangler instances with parametrization
    of transform output and sleep.

    """

    def create_wrangler(size=None, result=None, sleep=0):
        """Return instance of DaskWrangler.

        Parameters
        ----------
        size: float
            Memory size in MiB to allocate during transform step.
        result: Dask DataFrame
            Define extact return value of transform step.
        sleep: float
            Define sleep interval.

        """

        class DummyWrangler(DaskSingleNoFit):
            def transform(self, df):
                if size is not None:
                    pdf = pd.DataFrame(allocate_memory(size))
                    df_out = dd.from_pandas(pdf)
                elif result is not None:
                    df_out = result
                else:
                    df_out = dd.from_pandas(pd.DataFrame([0]), 1)

                time.sleep(sleep)
                return df_out

        return DummyWrangler()

    return create_wrangler


def test_dask_base_profiler_wrap_fit_transform(test_wrangler):
    pdf = pd.DataFrame(np.random.rand(50, 5))
    df = dd.from_pandas(pdf, 5).max().max()

    profiler = DaskTimeProfiler(wrangler=test_wrangler(result=df),
                                repetitions=1)

    wrapped = profiler._wrap_fit_transform()

    assert callable(wrapped)
    assert wrapped(df) == pdf.max().max()


def test_dask_base_profiler_cache_input():
    class MockPersist:
        def persist(self):
            self.persist_called = True
            return self

    dask_mocks = [MockPersist(), MockPersist()]

    persisted = DaskBaseProfiler._cache_input(dask_mocks)

    assert all([x.persist_called for x in persisted])


def test_dask_base_profiler_clear_cache_input():
    pdf = pd.DataFrame(np.random.rand(50, 5))

    with pytest.warns(None) as record:
        DaskBaseProfiler._clear_cached_input([dd.from_pandas(pdf, 5)])
        assert len(record) == 0

    df = dd.from_pandas(pdf, 5)
    ref = df  # noqa: F841

    with pytest.warns(ResourceWarning):
        DaskBaseProfiler._clear_cached_input([df])


def test_dask_time_profiler_fastest(test_wrangler):
    """Basic test for dask time profiler ensuring fastest timing is slower
    than forced sleep.

    """

    sleep = 0.001

    df_input = dd.from_pandas(pd.DataFrame(np.random.rand(10, 10)), 2)

    time_profiler = DaskTimeProfiler(wrangler=test_wrangler(sleep=sleep),
                                     repetitions=1,
                                     cache_input=True)

    assert time_profiler.profile(df_input).best >= sleep


def test_dask_time_profiler_profile_return_self(test_wrangler):
    df_input = dd.from_pandas(pd.DataFrame(np.random.rand(10, 10)), 2)

    time_profiler = DaskTimeProfiler(wrangler=test_wrangler(),
                                     repetitions=1)

    assert time_profiler.profile(df_input) is time_profiler


def test_dask_time_profiler_cached_faster(mean_wranger):
    pdf = pd.DataFrame(np.random.rand(1000000, 10))
    df_input = dd.from_pandas(pdf, 2).mean()

    time_profiler_no_cache = DaskTimeProfiler(wrangler=mean_wranger,
                                              repetitions=5,
                                              cache_input=False)

    time_profiler_cache = DaskTimeProfiler(wrangler=mean_wranger,
                                           repetitions=5,
                                           cache_input=True)

    no_cache_time = time_profiler_no_cache.profile(df_input).median
    cache_time = time_profiler_cache.profile(df_input).median

    assert no_cache_time > cache_time


def test_dask_memory_profiler_profile_return_self(test_wrangler):
    df_input = dd.from_pandas(pd.DataFrame(np.random.rand(10, 10)), 2)

    mem_profiler = DaskMemoryProfiler(wrangler=test_wrangler(),
                                      repetitions=1)

    assert mem_profiler.profile(df_input) is mem_profiler
    assert mem_profiler.runs == 1


@pytest.mark.xfail(reason="Succeeds locally but sometimes fails remotely due "
                          "to non deterministic memory management.")
def test_dask_memory_profiler_cached_lower_usage(mean_wranger):
    pdf = pd.DataFrame(np.random.rand(1000000, 10))
    df_input = dd.from_pandas(pdf, 5).mean()

    mem_profiler_no_cache = DaskMemoryProfiler(wrangler=mean_wranger,
                                               repetitions=5,
                                               cache_input=False,
                                               interval=0.00001)

    mem_profiler_cache = DaskMemoryProfiler(wrangler=mean_wranger,
                                            repetitions=5,
                                            cache_input=True,
                                            interval=0.00001)

    no_cache_usage = mem_profiler_no_cache.profile(df_input).median
    cache_usage = mem_profiler_cache.profile(df_input).median

    assert no_cache_usage > cache_usage
